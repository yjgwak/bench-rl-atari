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
        self.sigma = Dense(param['actor'][3], activation='softplus')

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
        action_prob = tf.clip_by_value(normal_dist.prob(action_clipped), eps, 1.0)

        return action_clipped, action_prob, self.critic(inputs)

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
        discounted_sum = reward + gamma + discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standarize:
        returns = ((returns - tf.math.reduce_mean(returns)) /
                (tf.math.reduce_std(returns) + eps))
    return returns


def train_step(initial_state: tf.Tensor,
               model: tf.keras.Model,
               optimizer: tf.keras.optimizers.Optimizer):

    max_steps = int(1e8)
    gamma = 0.99

    huber_loss = Huber(reduction=tf.keras.losses.Reduction.SUM)

    with tf.GradientTape() as tape:
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        state = initial_state
        for t in tf.range(max_steps):
            state = tf.expand_dims(tf.squeeze(state), 0)
            action, action_prob, value = model(state)

            values = values.write(t, tf.squeeze(value))

            action_probs = action_probs.write(t, tf.squeeze(action_prob))

            state, reward, done = tf_env_step(action)

            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        returns = get_expected_return(rewards, gamma)

        action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

        advantages = returns - values
        action_log_probs = tf.math.log(action_probs)
        loss_actor = -tf.math.reduce_sum(action_log_probs * advantages)
        loss_critic = huber_loss(value, returns)
        loss = loss_actor + loss_critic

    grads = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward


if __name__ == "__main__":

    layer_param = {'actor': [40, 40, 1, 1], 'critic': [400, 400, 1]}

    action_range = [env.action_space.low.item(), env.action_space.high.item()]
    model = ActorCritic(layer_param, action_range)

    optimizer = Adam(learning_rate=0.000001)

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
