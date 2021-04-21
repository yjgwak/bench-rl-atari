import gym
import numpy as np
import tensorflow as tf
import tqdm
from common import get_mc_return, compute_loss, get_td_return
from models.Simple import ActorCritic


class Agent:
    def __init__(self, env_name="CartPole-v0", seed=42):
        self.env = gym.make(env_name)
        self.set_seed(seed)

        num_actions = self.env.action_space.n  # 2
        num_hidden_units = 128
        self.core = ActorCritic(num_actions, num_hidden_units)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

    def env_step(self, action):
        state, reward, done, _ = self.env.step(action)
        return state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.int32)

    def tf_env_step(self, action):
        return tf.numpy_function(self.env_step, [action], [tf.float32, tf.int32, tf.int32])

    def set_seed(self, seed):
        self.env.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed()

    def run_episode(self, initial_state, max_steps):
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps):
            state = tf.expand_dims(state, 0)
            action_logits_t, value = self.core(state)

            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)

            values = values.write(t, tf.squeeze(value))

            action_probs = action_probs.write(t, action_probs_t[0, action])

            state, reward, done = self.tf_env_step(action)
            state.set_shape(initial_state_shape)

            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()
        return action_probs, values, rewards

    @tf.function
    def train_step(self, initial_state, gamma, max_steps_per_episode):
        """
            I don't know why but the initial_state should be given as a parameter.
            It's because of the annotation tf.function
        """
        with tf.GradientTape() as tape:
            action_probs, values, rewards = self.run_episode(initial_state, max_steps_per_episode)
            returns = get_td_return(rewards, values, gamma)
            action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]
            loss = compute_loss(action_probs, values, returns)

        grads = tape.gradient(loss, self.core.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.core.trainable_variables))
        episode_reward = tf.math.reduce_sum(rewards)
        return episode_reward

    def train(self, reward_threshold=195):
        max_episodes = 10000
        max_steps_per_episode = 1000
        gamma = 0.99

        ema_reward = 0
        writer = tf.summary.create_file_writer("log/cartpole")
        with tqdm.trange(max_episodes) as t:
            for i in t:
                initial_state = tf.constant(self.env.reset(), dtype=tf.float32)
                reward = int(self.train_step(initial_state, gamma, max_steps_per_episode))
                ema_reward = reward * 0.01 + ema_reward * .99

                # decoration
                t.set_description(f'Episode {i}')
                t.set_postfix(reward=reward, ema_reward=ema_reward)

                with writer.as_default():
                    tf.summary.scalar("reward", reward, step=i)
                if ema_reward > reward_threshold:
                    break

            print(f'\n Solved at episode {i}: average reward: {ema_reward: .2f}!')

        pass


if __name__ == "__main__":
    agent = Agent()
    agent.train()

