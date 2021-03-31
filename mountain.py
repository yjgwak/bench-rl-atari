import tensorflow as tf
import numpy as np
import gym  # requires OpenAI gym installed
import tqdm as tqdm
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.losses import Huber
from tensorflow.python.keras.optimizer_v2.adam import Adam
import tensorflow_probability as tfp

env = gym.make("MountainCarContinuous-v0")
eps = np.finfo(np.float32).eps.item()


def env_step(action):  # np.ndarray):# -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    state, reward, done, _ = env.step(action)
    return (state.astype(np.float32),
            np.array(reward, np.float32),
            np.array(done, np.int32))


def tf_env_step(action):  # tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(env_step, [action], [tf.float32, tf.float32, tf.int32])


class ActorCritic(tf.keras.Model):
    def __init__(self, param: dict, action_range):
        super().__init__()

        self.actor_pre = tf.keras.Sequential([
            Dense(param['actor'][0], activation='relu'),
            Dense(param['actor'][1], activation='relu')
        ])
        self.mu = Dense(param['actor'][2])
        self.sigma = Dense(param['actor'][3], activation='relu')

        self.critic = tf.keras.Sequential([
            Dense(param['critic'][0], activation='relu'),
            Dense(param['critic'][1], activation='relu'),
            Dense(param['critic'][2])
        ])

        self.action_range = action_range

    def call(self, inputs, **kwargs):
        x = self.actor_pre(inputs)
        normal_dist = tfp.distributions.Normal(self.mu(x), self.sigma(x) + eps)
        action_norm = tf.squeeze(normal_dist.sample(1), axis=0)
        action_clipped = tf.clip_by_value(action_norm, *self.action_range)

        return action_clipped, self.critic(inputs), normal_dist


def train_step(initial_state: tf.Tensor,
               model: tf.keras.Model,
               optimizer: tf.keras.optimizers.Optimizer):
    max_steps = int(1e8)

    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    state = initial_state
    state = tf.expand_dims(state, 0)
    gamma = 0.99

    for t in tf.range(max_steps):
        with tf.GradientTape() as tape:
            action, value, norm = model(state)
            norm = norm[0, 0]

            state, reward, done = tf_env_step(action[0])

            state = tf.expand_dims(state, 0)
            _, next_value, _ = model(state)

            target = reward + gamma * next_value[0, 0]
            td_error = target - value[0, 0]

            action_log_prob = tf.math.log(tf.clip_by_value(norm.prob(action[0, 0]), eps, 1.0))
            loss_actor = -action_log_prob * td_error
            loss_critic = huber_loss(value, target)
            # loss = loss_critic
            loss = loss_actor + loss_critic

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        losses = losses.write(t, norm.mean())

        if tf.cast(done, tf.bool):
            break

    rewards = rewards.stack()
    episode_reward = tf.math.reduce_sum(rewards)
    return episode_reward


if __name__ == "__main__":

    layer_param = {'actor': [40, 40, 1, 1], 'critic': [400, 400, 1]}

    action_range = [env.action_space.low.item(), env.action_space.high.item()]
    model = ActorCritic(layer_param, action_range)

    optimizer = Adam(learning_rate=0.000001)
    huber_loss = Huber(reduction=tf.keras.losses.Reduction.SUM)

    # Training loop
    gamma = 0.99  # discount factor
    max_episodes = 100
    running_reward = 0
    with tqdm.trange(max_episodes) as t:
        for i in t:
            initial_state = tf.constant(env.reset(), dtype=tf.float32)
            episode_reward = float(train_step(initial_state, model, optimizer))
            running_reward = episode_reward * .01 + running_reward * .99

            t.set_description(f'Episode {i}')
            t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)
            #
            #
            #
            # for step in range(max_steps):
            #     state = tf.expand_dims(state, 0)
            #     action, value, norm = model(state)
            #     action_prob = norm.prob(action) + eps
            #
            #     values = values.write(step, tf.squeeze(value))
            #     action_probs = action_probs.write(step, action_prob[0])
            #     state, reward, done = tf_env_step(action)
            #     rewards = rewards.write(t, reward)
            #
            #     if tf.cast(done, tf.bool):
            #         break
            #
            # t.set_description(f'Episode {i}')
            # # t.set_postfix()
            #
            # # next_state, reward, done = tf_env_step()
            # while (not done):
            #     # Sample action according to current policy
            #     # action.shape = (1,1)
            #     action = sess.run(action_tf_var, feed_dict={
            #         state_placeholder: scale_state(state)})
            #     # Execute action and observe reward & next state from E
            #     # next_state shape=(2,)
            #     # env.step() requires input shape = (1,)
            #     next_state, reward, done, _ = env.step(
            #         np.squeeze(action, axis=0))
            #     steps += 1
            #     reward_total += reward
            #     # V_of_next_state.shape=(1,1)
            #     V_of_next_state = sess.run(V, feed_dict=
            #     {state_placeholder: scale_state(next_state)})
            #     # Set TD Target
            #     # target = r + gamma * V(next_state)
            #     target = reward + gamma * np.squeeze(V_of_next_state)
            #
            #     # td_error = target - V(s)
            #     # needed to feed delta_placeholder in actor training
            #     td_error = target - np.squeeze(sess.run(V, feed_dict=
            #     {state_placeholder: scale_state(state)}))
            #
            #     # Update actor by minimizing loss (Actor training)
            #     _, loss_actor_val = sess.run(
            #         [training_op_actor, loss_actor],
            #         feed_dict={action_placeholder: np.squeeze(action),
            #                    state_placeholder: scale_state(state),
            #                    delta_placeholder: td_error})
            #     # Update critic by minimizinf loss  (Critic training)
            #     _, loss_critic_val = sess.run(
            #         [training_op_critic, loss_critic],
            #         feed_dict={state_placeholder: scale_state(state),
            #                    target_placeholder: target})
            #
            #     state = next_state
            #     # end while
            # episode_history.append(reward_total)
            # print("Episode: {}, Number of Steps : {}, Cumulative reward: {:0.2f}".format(
            #     i, steps, reward_total))
            #
            # if np.mean(episode_history[-100:]) > 90 and len(episode_history) >= 101:
            #     print("****************Solved***************")
            #     print("Mean cumulative reward over 100 episodes:{:0.2f}".format(
            #         np.mean(episode_history[-100:])))
