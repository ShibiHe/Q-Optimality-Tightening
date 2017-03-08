#! /usr/bin/env python
__author__ = 'frankhe'

"""
DQN agents
"""
import time
import os
import logging
import numpy as np
import cPickle

import ale_data_set
import sys
sys.setrecursionlimit(10000)

recording_size = 0


class OptimalityTightening(object):
    def __init__(self, q_network, epsilon_start, epsilon_min,
                 epsilon_decay, replay_memory_size, exp_pref, update_frequency,
                 replay_start_size, rng, transitions_sequence_length, transition_range, penalty_method,
                 weight_min, weight_max, weight_decay_length, two_train=False, late2=True, close2=True, verbose=False,
                 double=False, save_pkl=True):
        self.double_dqn = double
        self.network = q_network
        self.num_actions = q_network.num_actions
        self.epsilon_start = epsilon_start
        self.update_frequency = update_frequency

        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay_memory_size = replay_memory_size
        self.exp_dir = exp_pref + '_' + str(weight_max) + '_' + str(weight_min)
        if late2:
            self.exp_dir += '_l2'
        if close2:
            self.exp_dir += '_close2'
        else:
            self.exp_dir += '_len' + str(transitions_sequence_length) + '_r' + str(transition_range)
        if two_train:
            self.exp_dir += '_TTR'

        self.replay_start_size = replay_start_size
        self.rng = rng
        self.transition_len = transitions_sequence_length
        self.two_train = two_train
        self.verbose = verbose
        if verbose > 0:
            print "Using verbose", verbose
            self.exp_dir += '_vb' + str(verbose)

        self.phi_length = self.network.num_frames
        self.image_width = self.network.input_width
        self.image_height = self.network.input_height
        self.penalty_method = penalty_method
        self.batch_size = self.network.batch_size
        self.discount = self.network.discount
        self.transition_range = transition_range
        self.late2 = late2
        self.close2 = close2
        self.same_update = False
        self.save_pkl = save_pkl

        self.start_index = 0
        self.terminal_index = None

        self.weight_max = weight_max
        self.weight_min = weight_min
        self.weight = self.weight_max
        self.weight_decay_length = weight_decay_length
        self.weight_decay = (self.weight_max - self.weight_min) / self.weight_decay_length

        try:
            os.stat(self.exp_dir)
        except OSError:
            os.makedirs(self.exp_dir)

        self.data_set = ale_data_set.DataSet(width=self.image_width,
                                             height=self.image_height,
                                             rng=rng,
                                             max_steps=self.replay_memory_size,
                                             phi_length=self.phi_length,
                                             discount=self.discount,
                                             batch_size=self.batch_size,
                                             transitions_len=self.transition_len)

        # just needs to be big enough to create phi's
        self.test_data_set = ale_data_set.DataSet(width=self.image_width,
                                                  height=self.image_height,
                                                  rng=rng,
                                                  max_steps=self.phi_length * 2,
                                                  phi_length=self.phi_length)
        self.epsilon = self.epsilon_start
        if self.epsilon_decay != 0:
            self.epsilon_rate = ((self.epsilon_start - self.epsilon_min) /
                                 self.epsilon_decay)
        else:
            self.epsilon_rate = 0

        self.testing = False

        self._open_results_file()
        self._open_learning_file()
        self._open_recording_file()

        self.step_counter = None
        self.episode_reward = None
        self.start_time = None
        self.loss_averages = None
        self.total_reward = None

        self.episode_counter = 0
        self.batch_counter = 0

        self.holdout_data = None

        # In order to add an element to the data set we need the
        # previous state and action and the current reward.  These
        # will be used to store states and actions.
        self.last_img = None
        self.last_action = None

        # Exponential moving average of runtime performance.
        self.steps_sec_ema = 0.
        self.program_start_time = None
        self.last_count_time = None
        self.epoch_time = None
        self.total_time = None

    def time_count_start(self):
        self.last_count_time = self.program_start_time = time.time()

    def _open_results_file(self):
        logging.info("OPENING " + self.exp_dir + '/results.csv')
        self.results_file = open(self.exp_dir + '/results.csv', 'w', 0)
        self.results_file.write(\
            'epoch,num_episodes,total_reward,reward_per_epoch,mean_q, epoch time, total time\n')
        self.results_file.flush()

    def _open_learning_file(self):
        self.learning_file = open(self.exp_dir + '/learning.csv', 'w', 0)
        self.learning_file.write('mean_loss,epsilon\n')
        self.learning_file.flush()

    def _update_results_file(self, epoch, num_episodes, holdout_sum):
        out = "{},{},{},{},{},{},{}\n".format(epoch, num_episodes,
                                              self.total_reward, self.total_reward / float(num_episodes),
                                              holdout_sum, self.epoch_time, self.total_time)
        self.last_count_time = time.time()
        self.results_file.write(out)
        self.results_file.flush()

    def _update_learning_file(self):
        out = "{},{}\n".format(np.mean(self.loss_averages),
                               self.epsilon)
        self.learning_file.write(out)
        self.learning_file.flush()

    def _open_recording_file(self):
        self.recording_tot = 0
        self.recording_file = open(self.exp_dir + '/recording.csv', 'w', 0)
        self.recording_file.write('nn_output, q_return, history_return, loss')
        self.recording_file.write('\n')
        self.recording_file.flush()

    def _update_recording_file(self, nn_output, q_return, history_return, loss):
        if self.recording_tot > recording_size:
            return
        self.recording_tot += 1
        out = "{},{},{},{}".format(nn_output, q_return, history_return, loss)
        self.recording_file.write(out)
        self.recording_file.write('\n')
        self.recording_file.flush()

    def start_episode(self, observation):
        """
        This method is called once at the beginning of each episode.
        No reward is provided, because reward is only available after
        an action has been taken.

        Arguments:
           observation - height x width numpy array

        Returns:
           An integer action
        """

        self.step_counter = 0
        self.batch_counter = 0
        self.episode_reward = 0

        # We report the mean loss for every epoch.
        self.loss_averages = []

        self.start_time = time.time()
        return_action = self.rng.randint(0, self.num_actions)

        self.last_action = return_action

        self.last_img = observation

        return return_action

    def _choose_action(self, data_set, epsilon, cur_img, reward):
        """
        Add the most recent data to the data set and choose
        an action based on the current policy.
        """

        data_set.add_sample(self.last_img, self.last_action, reward, False, start_index=self.start_index)
        if self.step_counter >= self.phi_length:
            phi = data_set.phi(cur_img)
            action = self.network.choose_action(phi, epsilon)
        else:
            action = self.rng.randint(0, self.num_actions)

        return action

    def _do_training(self):
        """
        Returns the average loss for the current batch.
        May be overridden if a subclass needs to train the network
        differently.
        """
        if self.close2:
            self.data_set.random_close_transitions_batch(self.batch_size, self.transition_len)
        else:
            self.data_set.random_transitions_batch(self.batch_size, self.transition_len, self.transition_range)

        target_q_imgs = np.append(self.data_set.forward_imgs, self.data_set.backward_imgs, axis=1)
        target_q_table = self.network.q_target_vals(target_q_imgs)
        target_double_q_table = None
        if self.double_dqn:
            target_double_q_table = self.network.q_target(target_q_imgs)
        q_values = self.network.q_s_a_batch_vals(self.data_set.center_imgs, self.data_set.center_actions)

        states1 = np.zeros((self.batch_size, self.phi_length, self.image_height, self.image_width), dtype='uint8')
        actions1 = np.zeros((self.batch_size, 1), dtype='int32')
        targets1 = np.zeros((self.batch_size, 1), dtype='float32')
        states2 = np.zeros((self.batch_size, self.phi_length, self.image_height, self.image_width), dtype='uint8')
        actions2 = np.zeros((self.batch_size, 1), dtype='int32')
        targets2 = np.zeros((self.batch_size, 1), dtype='float32')
        """
            0 1 2 3* 4 5 6 7 8 V_R
            0 1 2 4  5 6 7 8 V_R
            V0 = r3 + y*Q4; V1 = r3 +y*r4 + y^2*Q5
            Q2 -r2 = Q3*y; Q1 - r1 - y*r2  = y^2*Q3
            V-1 = (Q2 - r2) / y; V-2 = (Q1 - r1 - y*r2)/y^2; V-3 = (Q0 -r0 -y*r1 - y^2*r2)/y^3
            r1 + y*r2 = R1 - y^2*R3
            Q1 = r1+y*r2 + y^2*Q3
        """
        for i in xrange(self.batch_size):
            q_value = q_values[i]
            if self.two_train:
                # does nothing first
                states2[i] = self.data_set.center_imgs[i]
                actions2[i] = self.data_set.center_actions[i]
                targets2[i] = q_value
            center_position = int(self.data_set.center_positions[i])
            if self.data_set.terminal.take(center_position, mode='wrap'):
                states1[i] = self.data_set.center_imgs[i]
                actions1[i] = self.data_set.center_actions[i]
                targets1[i] = self.data_set.center_return_values[i]
                continue
            forward_targets = np.zeros(self.transition_len, dtype=np.float32)
            backward_targets = np.zeros(self.transition_len, dtype=np.float32)
            for j in xrange(self.transition_len):
                if j > 0 and self.data_set.forward_positions[i, j] == center_position + 1:
                    forward_targets[j] = q_value
                else:
                    if not self.double_dqn:
                        forward_targets[j] = self.data_set.center_return_values[i] - \
                                             self.data_set.forward_return_values[i, j] * self.data_set.forward_discounts[i, j] + \
                                             self.data_set.forward_discounts[i, j] * \
                                             np.max(target_q_table[i, j])
                    else:
                        forward_targets[j] = self.data_set.center_return_values[i] - \
                                             self.data_set.forward_return_values[i, j] * self.data_set.forward_discounts[i, j] + \
                                             self.data_set.forward_discounts[i, j] * target_double_q_table[i, j]
                    """ for integrity"""
                    if self.verbose == 1:
                        end = self. data_set.forward_positions[i, j]
                        discount = 1.0
                        cumulative_reward = 0.0
                        for k in range(center_position, end):
                            cumulative_reward += discount * self.data_set.rewards.take(k, mode='wrap')
                            discount *= self.discount
                        cumulative_reward += discount * np.max(target_q_table[i, j])
                        if not np.isclose(cumulative_reward, forward_targets[j], atol=0.000001):
                            print self.data_set.backward_positions[i], self.data_set.center_positions[i], self.data_set.forward_positions[i]
                            print self.data_set.start_index.take(k, mode='wrap'), self.data_set.terminal_index.take(k, mode='wrap')
                            print 'center return=', self.data_set.center_return_values[i], 'forward return=', \
                                self.data_set.forward_return_values[i,j], 'forward discount=', self.data_set.forward_discounts[i, j]
                            end = self.data_set.forward_positions[i, j]
                            discount = 1.0
                            cumulative_reward = 0.0
                            for k in range(center_position, end):
                                cumulative_reward += discount * self.data_set.rewards.take(k, mode='wrap')
                                print k, 'cumulative=', cumulative_reward, 'discount=', discount, 'reward=', self.data_set.rewards.take(k, mode='wrap'), \
                                    'return=', self.data_set.return_value.take(k, mode='wrap')
                                print '\t start=', self.data_set.start_index.take(k, mode='wrap'), 'terminal=', self.data_set.terminal_index.take(k, mode='wrap')
                                discount *= self.discount
                            cumulative_reward += discount * np.max(target_q_table[i, j])
                            print 'final cumulative=', cumulative_reward, 'target=', forward_targets[j], \
                                'maxQ=', np.max(target_q_table[i, j])
                            raw_input()

                if self.data_set.backward_positions[i, j] == center_position + 1:
                    backward_targets[j] = q_value
                else:
                    backward_targets[j] = (-self.data_set.backward_return_values[i, j] +
                                           self.data_set.backward_discounts[i, j] * self.data_set.center_return_values[i] +
                                           target_q_table[i, self.transition_len + j, self.data_set.backward_actions[i, j]]) /\
                                          self.data_set.backward_discounts[i, j]
                    """ for integrity"""
                    if self.verbose == 1:
                        end = self.data_set.backward_positions[i, j]
                        discount = 1.0
                        cumulative_reward = 0.0
                        for k in range(end, center_position):
                            cumulative_reward += self.data_set.rewards.take(k, mode='wrap') * discount
                            discount *= self.discount
                        cumulative_reward = (-cumulative_reward + target_q_table[i, self.transition_len + j, self.data_set.actions.take(end, mode='wrap')])/discount
                        if not np.isclose(cumulative_reward, backward_targets[j], atol=0.000001):
                            print self.data_set.backward_positions[i], self.data_set.center_positions[i], self.data_set.forward_positions[i]
                            print self.data_set.start_index.take(k, mode='wrap'), self.data_set.terminal_index.take(k, mode='wrap')
                            print 'center return=', self.data_set.center_return_values[i], 'backward return=', \
                                self.data_set.backward_return_values[i,j], 'backward discount=', self.data_set.backward_discounts[i, j]
                            end = self.data_set.backward_positions[i, j]
                            discount = 1.0
                            cumulative_reward = 0.0
                            for k in range(end, center_position):
                                cumulative_reward += self.data_set.rewards.take(k, mode='wrap') * discount
                                print k, 'cumulative=', cumulative_reward, 'discount=', discount, 'reward=', self.data_set.rewards.take(k, mode='wrap'),\
                                    'return=', self.data_set.return_value.take(k, mode='wrap')
                                print '\t start=', self.data_set.start_index.take(k, mode='wrap'), 'terminal=', self.data_set.terminal_index.take(k, mode='wrap')
                                discount *= self.discount
                            cumulative_reward = (-cumulative_reward + target_q_table[i, self.transition_len + j, self.data_set.actions.take(end, mode='wrap')])/discount
                            print 'final cumulative=', cumulative_reward, 'target=', backward_targets[j], \
                                'Q=', target_q_table[i, self.transition_len + j, self.data_set.backward_actions[i, j]]
                            raw_input()

            forward_targets = np.append(forward_targets, self.data_set.center_return_values[i])
            v0 = v1 = forward_targets[0]
            if self.penalty_method == 'max':
                v_max = np.max(forward_targets[1:])
                v_min = np.min(backward_targets)
                if self.two_train and v_min < q_value:
                    v_min_index = np.argmin(backward_targets)
                    states2[i] = self.data_set.backward_imgs[i, v_min_index]
                    actions2[i] = self.data_set.backward_actions[i, v_min_index]
                    targets2[i] = self.data_set.backward_return_values[i, v_min_index] - \
                                  self.data_set.backward_discounts[i, v_min_index] * self.data_set.center_return_values[i] + \
                                  self.data_set.backward_discounts[i, v_min_index] * q_value
                if ((self.late2 and self.weight == self.weight_min) or (not self.late2)) \
                        and (v_max - 0.1 > q_value > v_min + 0.1):
                    v1 = v_max * 0.5 + v_min * 0.5
                elif v_max - 0.1 > q_value:
                    v1 = v_max
                elif ((self.late2 and self.weight == self.weight_min) or (not self.late2)) and v_min + 0.1 < q_value:
                    v1 = v_min

            states1[i] = self.data_set.center_imgs[i]
            actions1[i] = self.data_set.center_actions[i]
            targets1[i] = v0 * self.weight + (1-self.weight) * v1

        if self.two_train:
            if self.same_update:
                self.network.train(states2, actions2, targets2)
            else:
                self.network.train2(states2, actions2, targets2)
        loss = self.network.train(states1, actions1, targets1)
        # if self.recording_tot < recording_size:
        #     pass
            # for i in range(self.network.batch_size):
            #     self._update_recording_file(output[i], target[i], return_value[i], loss)
        return loss

    def step(self, reward, observation):
        """
        This method is called each time step.

        Arguments:
           reward      - Real valued reward.
           observation - A height x width numpy array

        Returns:
           An integer action.

        """

        self.step_counter += 1
        self.episode_reward += reward

        # TESTING---------------------------
        if self.testing:
            action = self._choose_action(self.test_data_set, 0.05,
                                         observation, np.clip(reward, -1, 1))

        # NOT TESTING---------------------------
        else:
            if len(self.data_set) > self.replay_start_size:
                self.epsilon = max(self.epsilon_min,
                                   self.epsilon - self.epsilon_rate)
                self.weight = max(self.weight_min,
                                  self.weight - self.weight_decay)

                action = self._choose_action(self.data_set, self.epsilon,
                                             observation,
                                             np.clip(reward, -1, 1))

                if self.step_counter % self.update_frequency == 0:
                    loss = self._do_training()
                    self.batch_counter += 1
                    self.loss_averages.append(loss)

            else:  # Still gathering initial random data...
                action = self._choose_action(self.data_set, self.epsilon,
                                             observation,
                                             np.clip(reward, -1, 1))

        self.last_action = action
        self.last_img = observation

        return action

    def end_episode(self, reward, terminal=True):
        """
        This function is called once at the end of an episode.

        Arguments:
           reward      - Real valued reward.
           terminal    - Whether the episode ended intrinsically
                         (ie we didn't run out of steps)
        Returns:
            None
        """

        self.episode_reward += reward
        self.step_counter += 1
        total_time = time.time() - self.start_time

        if self.testing:
            # If we run out of time, only count the last episode if
            # it was the only episode.
            if terminal or self.episode_counter == 0:
                self.episode_counter += 1
                self.total_reward += self.episode_reward
        else:
            # Store the latest sample.
            self.data_set.add_sample(self.last_img,
                                     self.last_action,
                                     np.clip(reward, -1, 1),
                                     True, start_index=self.start_index)
            """update"""
            if terminal:
                q_return = 0.
            else:
                phi = self.data_set.phi(self.last_img)
                q_return = np.mean(self.network.q_vals(phi))
            # last_q_return = -1.0
            self.start_index = self.data_set.top
            self.terminal_index = index = (self.start_index-1) % self.data_set.max_steps
            while True:
                q_return = q_return * self.network.discount + self.data_set.rewards[index]
                self.data_set.return_value[index] = q_return
                self.data_set.terminal_index[index] = self.terminal_index
                index = (index-1) % self.data_set.max_steps
                if self.data_set.terminal[index] or index == self.data_set.bottom:
                    break

            rho = 0.98
            self.steps_sec_ema *= rho
            self.steps_sec_ema += (1. - rho) * (self.step_counter/total_time)

            logging.info("steps/second: {:.2f}, avg: {:.2f}".format(
                self.step_counter/total_time, self.steps_sec_ema))

            if self.batch_counter > 0:
                self._update_learning_file()
                logging.info("average loss: {:.4f}".format(\
                                np.mean(self.loss_averages)))

    def finish_epoch(self, epoch):
        if self.save_pkl:
            net_file = open(self.exp_dir + '/network_file_' + str(epoch) + '.pkl', 'w')
            cPickle.dump(self.network, net_file, -1)
            net_file.close()
        this_time = time.time()
        self.total_time = this_time-self.program_start_time
        self.epoch_time = this_time-self.last_count_time

    def start_testing(self):
        self.testing = True
        self.total_reward = 0
        self.episode_counter = 0

    def finish_testing(self, epoch):
        self.testing = False
        holdout_size = 3200

        if self.holdout_data is None and len(self.data_set) > holdout_size:
            imgs = self.data_set.random_imgs(holdout_size)
            self.holdout_data = imgs[:, :self.phi_length]

        holdout_sum = 0
        if self.holdout_data is not None:
            for i in range(holdout_size):
                holdout_sum += np.max(
                    self.network.q_vals(self.holdout_data[i]))

        self._update_results_file(epoch, self.episode_counter,
                                  holdout_sum / holdout_size)

