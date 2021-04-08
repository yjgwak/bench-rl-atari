from typing import Tuple
import tensorflow as tf
from tensorflow.python.layers import layers


class Actor(tf.keras.Model):
    def __init__(self, num_actions, num_hidden_units=128):
        super(Actor, self).__init__()
        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.actor = layers.Dense(num_actions)

    def call(self, inputs):
        x = self.common(inputs)
        return self.actor(x)


class Critic(tf.keras.Model):
    def __init__(self, num_hidden_units=128):
        super(Critic, self).__init__()
        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.critic = layers.Dense(1)

    def call(self, inputs):
        x = self.common(inputs)
        return self.critic(x)


class ActorCritic(tf.keras.Model):
    def __init__(self, num_actions, num_hidden_units=128):
        super(ActorCritic, self).__init__()
        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)

    def call(self, inputs):
        x = self.common(inputs)
        return self.actor(x), self.critic(x)
