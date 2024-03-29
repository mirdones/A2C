import numpy as np
import tensorflow as tf


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        # sample a random categorical action from given logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        # no tf.get_variable(), just simple Keras API
        self.hidden1 = tf.keras.layers.Dense(128, activation='relu')
        self.hidden2 = tf.keras.layers.Dense(128, activation='relu')
        self.value = tf.keras.layers.Dense(1, name='value')
        # logits are unnormalized log probabilities
        self.logits = tf.keras.layers.Dense(num_actions, name='policy_logits')
        # self.dist = ProbabilityDistribution()

    def call(self, inputs, training=None, mask=None):
        # inputs is a numpy array, convert to Tensor
        x = tf.convert_to_tensor(inputs)
        # separate hidden layers from the same input tensor
        hidden_logs = self.hidden1(x)
        hidden_vals = self.hidden2(x)
        return self.logits(hidden_logs), self.value(hidden_vals)

    def action_value(self, obs):
        # executes call() under the hood
        logits, value = self.predict(obs)
        # action = self.dist.predict(logits)
        # a simpler option, will become clear later why we don't use it
        action = tf.random.categorical(logits, 1)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)
