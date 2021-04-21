from typing import Tuple, List

import gym
import tqdm as tqdm
import tensorflow as tf
import numpy as np

from common import compute_loss, get_mc_return
from models.Simple import ActorCritic

env = gym.make("CartPole-v0")

seed = 42
env.seed(seed)
tf.random.set_seed(seed)
np.random.seed()

num_actions = env.action_space.n # 2
num_hidden_units = 128


def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    state, reward, done, _ = env.step(action)
    return (state.astype(np.float32),
            np.array(reward, np.int32),
            np.array(done, np.int32))


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(env_step, [action], [tf.float32, tf.int32, tf.int32])


def run_episde(initial_state, model, max_steps):
    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        state = tf.expand_dims(state, 0)
        action_logits_t, value = model(state)

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


@tf.function
def train_step(initial_state, model, optimizer, gamma, max_steps_per_episode):
    with tf.GradientTape() as tape:
        action_probs, values, rewards = run_episde(initial_state, model, max_steps_per_episode)

        returns = get_mc_return(rewards, gamma)
        action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]
        loss = compute_loss(action_probs, values, returns)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_reward = tf.math.reduce_sum(rewards)
    return episode_reward


max_episodes = 10000
max_steps_per_episode = 1000
reward_threshold = 195
gamma = 0.99

running_reward = 0
model = ActorCritic(num_actions, num_hidden_units)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

writer = tf.summary.create_file_writer("log/cartpole")
with tqdm.trange(max_episodes) as t:
    for i in t:
        initial_state = tf.constant(env.reset(), dtype=tf.float32)
        episode_reward = int(train_step(initial_state, model, optimizer, gamma, max_steps_per_episode))

        running_reward = episode_reward*0.01 + running_reward * .99

        t.set_description(f'Episode {i}')
        t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)
        with writer.as_default():
            tf.summary.scalar("reward_ep", episode_reward, step=i)
            tf.summary.scalar("reward_rn", running_reward, step=i)
        if running_reward > reward_threshold:
            break

    print(f'\n Solved at episode {i}: average reward: {running_reward: .2f}!')

