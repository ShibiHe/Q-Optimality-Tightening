#! /usr/bin/env python
__author__ = 'frankhe'

import lasagne
import numpy as np
import theano
import theano.tensor as T
from updates import deepmind_rmsprop


class DeepQLearner:
    def __init__(self, input_width, input_height, num_actions,
                 num_frames, discount, learning_rate, rho,
                 rms_epsilon, momentum, clip_delta, freeze_interval,
                 batch_size, network_type, update_rule,
                 batch_accumulator, rng, input_scale=255.0,
                 double=False, transition_length=4):

        if double:
            print 'USING DOUBLE DQN'
        self.input_width = input_width
        self.input_height = input_height
        self.num_actions = num_actions
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.discount = discount
        self.rho = rho
        self.lr = learning_rate
        self.rms_epsilon = rms_epsilon
        self.momentum = momentum
        self.clip_delta = clip_delta
        self.freeze_interval = freeze_interval
        self.rng = rng

        lasagne.random.set_rng(self.rng)

        self.update_counter = 0

        self.l_out = self.build_network(network_type, input_width, input_height,
                                        num_actions, num_frames, batch_size)
        if self.freeze_interval > 0:
            self.next_l_out = self.build_network(network_type, input_width,
                                                 input_height, num_actions,
                                                 num_frames, batch_size)
            self.reset_q_hat()

        states = T.tensor4('states_t')
        actions = T.icol('actions_t')
        target = T.col('evaluation_t')

        self.states_shared = theano.shared(
            np.zeros((batch_size, num_frames, input_height, input_width),
                     dtype=theano.config.floatX))
        self.actions_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))
        self.target_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self.states_transition_shared = theano.shared(
            np.zeros((batch_size, transition_length * 2, num_frames, input_height, input_width),
                     dtype=theano.config.floatX))
        self.states_one_shared = theano.shared(
            np.zeros((num_frames, input_height, input_width),
                     dtype=theano.config.floatX))

        q_vals = lasagne.layers.get_output(self.l_out, states / input_scale)

        """get Q(s)   batch_size = 1 """
        q1_givens = {
            states: self.states_one_shared.reshape((1,
                                                    self.num_frames,
                                                    self.input_height,
                                                    self.input_width))
        }
        self._q1_vals = theano.function([], q_vals[0], givens=q1_givens)
        """get Q(s)   batch_size = batch size """
        q_batch_givens = {
            states: self.states_shared.reshape((self.batch_size,
                                                self.num_frames,
                                                self.input_height,
                                                self.input_width))
        }
        self._q_batch_vals = theano.function([], q_vals, givens=q_batch_givens)

        action_mask = T.eq(T.arange(num_actions).reshape((1, -1)),
                           actions.reshape((-1, 1))).astype(theano.config.floatX)

        q_s_a = (q_vals * action_mask).sum(axis=1).reshape((-1, 1))
        """ get Q(s,a)   batch_size = batch size """
        q_s_a_givens = {
            states: self.states_shared.reshape((self.batch_size,
                                                self.num_frames,
                                                self.input_height,
                                                self.input_width)),
            actions: self.actions_shared
        }
        self._q_s_a_vals = theano.function([], q_s_a, givens=q_s_a_givens)

        if self.freeze_interval > 0:
            q_target_vals = lasagne.layers.get_output(self.next_l_out,
                                                      states / input_scale)
        else:
            q_target_vals = lasagne.layers.get_output(self.l_out,
                                                      states / input_scale)
            q_target_vals = theano.gradient.disconnected_grad(q_target_vals)

        if not double:
            q_target = T.max(q_target_vals, axis=1)
        else:
            greedy_actions = T.argmax(q_vals, axis=1)
            q_target_mask = T.eq(T.arange(num_actions).reshape((1, -1)),
                                 greedy_actions.reshape((-1, 1)).astype(theano.config.floatX))
            q_target = (q_target_vals * q_target_mask).sum(axis=1).reshape((-1, 1))
        """get Q target Q'(s,a') for a batch of transitions  batch size = batch_size * transition length"""
        q_target_transition_givens = {
            states: self.states_transition_shared.reshape(
                (batch_size * transition_length * 2, self.num_frames, self.input_height, self.input_width))
        }
        self._q_target = theano.function([], q_target.reshape((batch_size, transition_length * 2)),
                                         givens=q_target_transition_givens)
        """get Q target_vals Q'(s) for a batch of transitions  batch size = batch_size * transition length"""
        self._q_target_vals = theano.function([], q_target_vals.reshape(
            (batch_size, transition_length * 2, num_actions)), givens=q_target_transition_givens)

        diff = q_s_a - target

        if self.clip_delta > 0:
            # If we simply take the squared clipped diff as our loss,
            # then the gradient will be zero whenever the diff exceeds
            # the clip bounds. To avoid this, we extend the loss
            # linearly past the clip point to keep the gradient constant
            # in that regime.
            # 
            # This is equivalent to declaring d loss/d q_vals to be
            # equal to the clipped diff, then backpropagating from
            # there, which is what the DeepMind implementation does.
            quadratic_part = T.minimum(abs(diff), self.clip_delta)
            linear_part = abs(diff) - quadratic_part
            loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        else:
            loss = 0.5 * diff ** 2

        if batch_accumulator == 'sum':
            loss = T.sum(loss)
        elif batch_accumulator == 'mean':
            loss = T.mean(loss)
        else:
            raise ValueError("Bad accumulator: {}".format(batch_accumulator))

        params = lasagne.layers.helper.get_all_params(self.l_out)

        if update_rule == 'deepmind_rmsprop':
            updates = deepmind_rmsprop(loss, params, self.lr, self.rho,
                                       self.rms_epsilon)
        elif update_rule == 'rmsprop':
            updates = lasagne.updates.rmsprop(loss, params, self.lr, self.rho,
                                              self.rms_epsilon)
        elif update_rule == 'sgd':
            updates = lasagne.updates.sgd(loss, params, self.lr)
        else:
            raise ValueError("Unrecognized update: {}".format(update_rule))

        if self.momentum > 0:
            updates = lasagne.updates.apply_momentum(updates, None,
                                                     self.momentum)
        """Q(s,a) target train()"""
        train_givens = {
            states: self.states_shared,
            actions: self.actions_shared,
            target: self.target_shared
        }
        self._train = theano.function([], [loss], updates=updates, givens=train_givens, on_unused_input='warn')

        self._train2 = theano.function([], [loss], updates=updates, givens=train_givens, on_unused_input='warn')

    def q_vals(self, single_state):
        self.states_one_shared.set_value(single_state)
        return self._q1_vals()

    def q_batch_vals(self, states):
        self.states_shared.set_value(states)
        return self._q_batch_vals()

    def q_s_a_batch_vals(self, states, actions):
        self.states_shared.set_value(states)
        self.actions_shared.set_value(actions)
        return self._q_s_a_vals()

    def q_target(self, batch_transition_states):
        self.states_transition_shared.set_value(batch_transition_states)
        return self._q_target()

    def q_target_vals(self, batch_transition_states):
        self.states_transition_shared.set_value(batch_transition_states)
        return self._q_target_vals()

    def train(self, states, actions, target):
        self.states_shared.set_value(states)
        self.actions_shared.set_value(actions)
        self.target_shared.set_value(target)
        if self.freeze_interval > 0 and self.update_counter % self.freeze_interval == 0:
            self.reset_q_hat()
        loss = self._train()
        self.update_counter += 1
        return np.sqrt(loss)

    def train2(self, states, actions, target):
        self.states_shared.set_value(states)
        self.actions_shared.set_value(actions)
        self.target_shared.set_value(target)
        if self.freeze_interval > 0 and self.update_counter % self.freeze_interval == 0:
            self.reset_q_hat()
        loss = self._train2()
        return np.sqrt(loss)

    def build_network(self, network_type, input_width, input_height,
                      output_dim, num_frames, batch_size):
        if network_type == "nature_cuda":
            return self.build_nature_network(input_width, input_height,
                                             output_dim, num_frames, batch_size)
        if network_type == "nature_dnn":
            return self.build_nature_network_dnn(input_width, input_height,
                                                 output_dim, num_frames,
                                                 batch_size)
        elif network_type == "linear":
            return self.build_linear_network(input_width, input_height,
                                             output_dim, num_frames, batch_size)
        else:
            raise ValueError("Unrecognized network: {}".format(network_type))

    def choose_action(self, state, epsilon):
        if self.rng.rand() < epsilon:
            return self.rng.randint(0, self.num_actions)
        q_vals = self.q_vals(state)
        return np.argmax(q_vals)

    def reset_q_hat(self):
        all_params = lasagne.layers.helper.get_all_param_values(self.l_out)
        lasagne.layers.helper.set_all_param_values(self.next_l_out, all_params)

    def build_nature_network(self, input_width, input_height, output_dim,
                             num_frames, batch_size):
        """
        Build a large network consistent with the DeepMind Nature paper.
        """
        from lasagne.layers import cuda_convnet

        l_in = lasagne.layers.InputLayer(
            shape=(None, num_frames, input_width, input_height)
        )

        l_conv1 = cuda_convnet.Conv2DCCLayer(
            l_in,
            num_filters=32,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(), # Defaults to Glorot
            b=lasagne.init.Constant(.1),
            dimshuffle=True
        )

        l_conv2 = cuda_convnet.Conv2DCCLayer(
            l_conv1,
            num_filters=64,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            dimshuffle=True
        )

        l_conv3 = cuda_convnet.Conv2DCCLayer(
            l_conv2,
            num_filters=64,
            filter_size=(3, 3),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            dimshuffle=True
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv3,
            num_units=512,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=output_dim,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        return l_out

    def build_nature_network_dnn(self, input_width, input_height, output_dim,
                                 num_frames, batch_size):
        """
        Build a large network consistent with the DeepMind Nature paper.
        """
        from lasagne.layers import dnn

        l_in = lasagne.layers.InputLayer(
            shape=(None, num_frames, input_width, input_height)
        )

        l_conv1 = dnn.Conv2DDNNLayer(
            l_in,
            num_filters=32,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_conv2 = dnn.Conv2DDNNLayer(
            l_conv1,
            num_filters=64,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_conv3 = dnn.Conv2DDNNLayer(
            l_conv2,
            num_filters=64,
            filter_size=(3, 3),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv3,
            num_units=512,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=output_dim,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        return l_out

    def build_linear_network(self, input_width, input_height, output_dim,
                             num_frames, batch_size):
        """
        Build a simple linear learner.  Useful for creating
        tests that sanity-check the weight update code.
        """

        l_in = lasagne.layers.InputLayer(
            shape=(None, num_frames, input_width, input_height)
        )

        l_out = lasagne.layers.DenseLayer(
            l_in,
            num_units=output_dim,
            nonlinearity=None,
            W=lasagne.init.Constant(0.0),
            b=None
        )

        return l_out
