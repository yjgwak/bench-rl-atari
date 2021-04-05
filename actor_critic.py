from typing import Tuple, List

import gym
import tqdm as tqdm
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.python.layers import layers

env = gym.make("CartPole-v0")

seed = 42
env.seed(seed)
tf.random.set_seed(seed)
np.random.seed()

eps = np.finfo(np.float32).eps.item()


class Actor(tf.keras.Model):
    def __init__(self, num_actions: int, num_hidden_units: int):
        super(Actor, self).__init__()
        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.actor = layers.Dense(num_actions)

    def call(self, inputs):
        x = self.common(inputs)
        return self.actor(x)


class Critic(tf.keras.Model):
    def __init__(self, num_hidden_units: int):
        super(Critic, self).__init__()
        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.critic = layers.Dense(1)

    def call(self, inputs):
        x = self.common(inputs)
        return self.critic(x)


class ActorCritic(tf.keras.Model):
    def __init__(self,
                 num_actions: int,
                 num_hidden_units: int):
        super(ActorCritic, self).__init__()
        self.common = layers.Dense(num_hidden_units, activation="relu")
        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)


num_actions = env.action_space.n # 2
num_hidden_units = 128


def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    state, reward, done, _ = env.step(action)
    return (state.astype(np.float32),
            np.array(reward, np.int32),
            np.array(done, np.int32))


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(env_step, [action], [tf.float32, tf.int32, tf.int32])


def run_episde(
        initial_state: tf.Tensor,
        actor: tf.keras.Model,
        critic: tf.keras.Model,
        max_steps: int) -> List[tf.Tensor]:

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        state = tf.expand_dims(state, 0)
        action_logits_t = actor(state)
        value = critic(state)

        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)

        values = values.write(t, tf.squeeze(value))

        action_probs = action_probs.write(t, action_probs_t[0, action])

        state, reward, done = tf_env_step(action)
        state.set_shape(initial_state_shape)

        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break

    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()


    return action_probs, values, rewards


def get_expected_return(
        rewards: tf.Tensor,
        gamma: float,
        standarize: bool = True) -> tf.Tensor:

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standarize:
        returns = ((returns - tf.math.reduce_mean(returns)) /
                (tf.math.reduce_std(returns) + eps))
    return returns

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_loss(
        action_probs: tf.Tensor,
        values: tf.Tensor,
        returns: tf.Tensor) -> tf.Tensor:

    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = huber_loss(values, returns)

    return actor_loss, critic_loss


@tf.function
def train_step(
        initial_state: tf.Tensor,
        actor: tf.keras.Model,
        critic: tf.keras.Model,
        opt1: tf.keras.optimizers.Optimizer,
        opt2: tf.keras.optimizers.Optimizer,
        gamma: float,
        max_steps_per_episode: int) -> tf.Tensor:

    with tf.GradientTape(persistent=True) as tape:
        action_probs, values, rewards = run_episde(
            initial_state, actor, critic, max_steps_per_episode)

        returns = get_expected_return(rewards, gamma)

        action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

        loss_actor, loss_critic = compute_loss(action_probs, values, returns)

    grads = tape.gradient(loss_actor, actor.trainable_variables)
    opt1.apply_gradients(zip(grads, actor.trainable_variables))
    grads = tape.gradient(loss_critic, critic.trainable_variables)
    opt2.apply_gradients(zip(grads, critic.trainable_variables))
    del tape
    episode_reward = tf.math.reduce_sum(rewards)
    return episode_reward


max_episodes = 10000
max_steps_per_episode = 1000
reward_threshold = 195
gamma = 0.99

running_reward = 0
model = ActorCritic(num_actions, num_hidden_units)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

actor = Actor(num_actions, num_hidden_units)
critic = Critic(num_hidden_units)

opt1 = tf.keras.optimizers.Adam(learning_rate=0.01)
opt2 = tf.keras.optimizers.Adam(learning_rate=0.01)

with tqdm.trange(max_episodes) as t:
    for i in t:
        initial_state = tf.constant(env.reset(), dtype=tf.float32)
        episode_reward = int(train_step(initial_state, actor, critic, opt1, opt2, gamma, max_steps_per_episode))

        running_reward = episode_reward*0.01 + running_reward * .99

        t.set_description(f'Episode {i}')
        t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)
        if running_reward > reward_threshold:
            break

    print(f'\n Solved at episode {i}: average reward: {running_reward: .2f}!')

