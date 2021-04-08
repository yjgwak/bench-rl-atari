import tensorflow as tf
import gym
import os
import threading


class MasterAgent:
    def __init__(self, save_dir="save", env_name='CartPole-v0'):
        self.env_name = env_name
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        env = gym.make(self.env_name)
        self.action_size = env.action_space.n
        self.opt = tf.train.AdamOptimizer(learning_rate=0.001, use_locking=True)

        self.global_model = ActorCritic(self.action_size)  # global network
        self.global_model(tf.constant(np.random.random((1, self.state_size)), dtype=tf.float32))

    def play(self):
        env = gym.make(self.env_name).unwrapped
        state = env.reset()
        model = self.global_model
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
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []


class Worker(threading.Thread):
    # Set up global variables across different threads
    global_episode = 0

    # Moving average reward
    global_moving_average_reward = 0
    best_score = 0
    save_lock = threading.Lock()

    def __init__(self, global_model, opt, result_queue, idx, env_name='CartPole-v0', save_dir='tmp'):
        super(Worker, self).__init__()
        self.global_model = global_model
        self.opt = opt
        self.result_queue = result_queue
        self.worker_idx = idx
        self.env_name = env_name

        self.env = gym.make(self.env_name).unwrapped
        self.action_size = self.env.action_space.n

        self.save_dir = save_dir

        self.local_model = ActorCritic(self.action_size)
        self.ep_loss = 0.0

    def run(self):
        total_step = 1
        mem = Memory()

        while Worker.global_episode < args.max_eps:
            current_state = self.env.reset()
            mem.clear()
            ep_reward = 0.
            ep_steps = 0
            self.ep_loss = 0

            time_count = 0
            done = False
            while not done:
                action_logits, _ = self.local_model(tf.constant(current_state[None, :], dtype=tf.float32))
                action = tf.random.categorical(action_logits_t, 1)[0, 0]
                probs = tf.nn.softmax(action_logits)

                action = np.random.choice(self.action_size, p=probs.numpy()[0])
                new_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = -1
                ep_reward += reward
                mem.store(current_state, action, reward)

                if time_count == args.update_freq or done:
                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape
                    with tf.GradientTape() as tape:
                        total_loss = self.compute_loss(done, new_state, mem, args.gamma)
                    self.ep_loss += total_loss
                    # Calculate local gradients
                    grads = tape.gradient(total_loss, self.local_model.trainable_weights)
                    # Push local gradients to global model
                    self.opt.apply_gradients(zip(grads, self.global_model.trainable_weights))
                    # Update local model with new weights
                    self.local_model.set_weights(self.global_model.get_weights())

                    mem.clear()
                    time_count = 0

                    if done:  # done and print information
                        Worker.global_moving_average_reward = \
                            record(Worker.global_episode, ep_reward, self.worker_idx,
                                   Worker.global_moving_average_reward, self.result_queue,
                                   self.ep_loss, ep_steps)
                        # We must use a lock to save our model and to print to prevent data races.
                        if ep_reward > Worker.best_score:
                            with Worker.save_lock:
                                print("Saving best model to {}, "
                                      "episode score: {}".format(self.save_dir, ep_reward))
                                self.global_model.save_weights(
                                    os.path.join(self.save_dir,
                                                 'model_{}.h5'.format(self.env_name))
                                )
                                Worker.best_score = ep_reward
                        Worker.global_episode += 1
                ep_steps += 1

                time_count += 1
                current_state = new_state
                total_step += 1
        self.result_queue.put(None)

    def compute_loss(self, done, new_state, memory, gamma=0.99):
        if done:
            reward_sum = 0.  # terminal
        else:
            reward_sum = self.local_model(
                tf.constant(new_state[None, :],
                                     dtype=tf.float32))[-1].numpy()[0]

        # Get discounted rewards
        discounted_rewards = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        logits, values = self.local_model(
            tf.constant(np.vstack(memory.states),
                                 dtype=tf.float32))
        # Get our advantages
        advantage = tf.constant(np.array(discounted_rewards)[:, None],
                                         dtype=tf.float32) - values
        # Value loss
        value_loss = advantage ** 2

        # Calculate our policy loss
        actions_one_hot = tf.one_hot(memory.actions, self.action_size, dtype=tf.float32)

        policy = tf.nn.softmax(logits)
        entropy = tf.reduce_sum(policy * tf.log(policy + 1e-20), axis=1)

        policy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=actions_one_hot,
                                                                 logits=logits)
        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= 0.01 * entropy
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss
