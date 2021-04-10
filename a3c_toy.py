import tensorflow as tf
import gym
import os
import threading
import numpy as np
from models.Simple import ActorCritic
from a2c_toy import Agent


class MasterAgent:
    def __init__(self, save_dir="save", env_name='CartPole-v0'):
        self.env_name = env_name
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        env = gym.make(self.env_name)
        self.num_actions = env.action_space.n

        self.global_model = ActorCritic(self.num_actions)  # global network
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3, use_locking=True)
        self.global_model(tf.constant(np.random.random((1, 4)), dtype=tf.float32))

    def run(self):
        for i in range(16):
            local_agent = Worker(i, self.global_model, self.optimizer, self.env_name)
            local_agent.start()

    def play(self, load=False):
        env = gym.make(self.env_name).unwrapped
        state = env.reset()
        model = self.global_model
        if load:
            model_path = os.path.join(self.save_dir, 'model_{}.h5'.format(self.env_name))
            print('Loading model from: {}'.format(model_path))
            model.load_weights(model_path)
        done = False
        step_counter = 0
        reward_sum = 0

        try:
            while not done:
                env.render(mode='rgb_array')
                policy, value = model(tf.constant(state[None, :], dtype=tf.float32))
                policy = tf.nn.softmax(policy)
                action = np.argmax(policy)
                state, reward, done, _ = env.step(action)
                reward_sum += reward
                print("{}. Reward: {}, action: {}".format(step_counter, reward_sum, action))
                step_counter += 1
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            env.close()


class Memory:
    def __init__(self):
        self.step = 0
        self.action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        self.values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        self.rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    def write(self, action_prob, value, reward):
        self.action_probs = self.action_probs.write(self.step, action_prob)
        self.values = self.values.write(self.step, value)
        self.rewards = self.rewards.write(self.step, reward)
        self.step += 1

    def stack(self):
        return self.action_probs.stack(), self.values.stack(), self.rewards.stack()


class Worker(threading.Thread):
    # Set up global variables across different threads
    global_episode = 0

    def __init__(self, idx, global_model, optimizer, env_name='CartPole-v0'):
        super(Worker, self).__init__()
        self.idx = idx
        self.global_model = global_model
        self.optimizer = optimizer

        self.env_name = env_name
        self.env = gym.make(self.env_name).unwrapped
        self.num_actions = self.env.action_space.n
        self.local_model = ActorCritic(self.num_actions)

    def run(self):
        max_episodes = 10000
        max_steps_per_episode = 1000
        gamma = 0.99

        done = False
        while Worker.global_episode < max_episodes and not done:
            initial_state = tf.constant(self.env.reset(), dtype=tf.float32)
            with tf.GradientTape() as tape:
                action_probs, values, rewards = self._run_episode(initial_state, max_steps_per_episode)
                returns = Agent.get_expected_return(rewards, gamma)
                action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]
                loss = Agent.compute_loss(action_probs, values, returns)

            grads = tape.gradient(loss, self.local_model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.global_model.trainable_variables))
            self.local_model.set_weights(self.global_model.get_weights())
            Worker.global_episode += 1
            print(f"Episode: {Worker.global_episode}, Reward: {sum(rewards)}")

    def _run_episode(self, initial_state, max_steps_per_episode):
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps_per_episode):
            state = tf.expand_dims(state, 0)
            action_logits_t, value = self.local_model(state)

            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)

            values = values.write(t, tf.squeeze(value))

            action_probs = action_probs.write(t, action_probs_t[0, action])

            state, reward, done = self._tf_env_step(action)
            state.set_shape(initial_state_shape)

            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_probs, values, rewards

    def _env_step(self, action):
        state, reward, done, _ = self.env.step(action)
        return state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.int32)

    def _tf_env_step(self, action):
        return tf.numpy_function(self._env_step, [action], [tf.float32, tf.int32, tf.int32])


if __name__ == "__main__":
    agent = MasterAgent()
    agent.run()
    # agent.play()
