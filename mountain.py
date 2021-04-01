import tensorflow as tf
import numpy as np
import gym  # requires OpenAI gym installed
import tqdm as tqdm
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.losses import Huber
from tensorflow.python.keras.optimizer_v2.adam import Adam
import tensorflow_probability as tfp


env = gym.make("MountainCarContinuous-v0")

seed = 42
env.seed(seed)
tf.random.set_seed(seed)
np.random.seed()

eps = np.finfo(np.float32).eps.item()


def env_step(action): # np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    state, reward, done, _ = env.step(action)
    return (state.astype(np.float32),
            np.array(reward, np.float32),
            np.array(done, np.int32))


def tf_env_step(action): # tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(env_step, [action], [tf.float32, tf.float32, tf.int32])


class Actor(tf.keras.Model):
    def __init__(self, params, action_range):
        super().__init__()
        self.seq = tf.keras.Sequential([Dense(p, activation='relu') for p in params])
        self.mean = Dense(1)
        self.stddev = Dense(1, activation='softplus')
        self.action_range = action_range

    def call(self, inputs, **kwargs):
        x = self.seq(inputs)
        n = tfp.distributions.Normal(self.mean(x), self.stddev(x))
        action = tf.clip_by_value(n.sample(1), *self.action_range)
        action_log_prob = n.log_prob(action)
        return action, action_log_prob


class Critic(tf.keras.Model):
    def __init__(self, params):
        super().__init__()
        self.seq = tf.keras.Sequential([Dense(p, activation='relu') for p in params])
        self.critic = Dense(1)

    def call(self, inputs, **kwargs):
        x = self.seq(inputs)
        return self.critic(x)


class ActorCritic(tf.keras.Model):
    def __init__(self, action_range):
        super().__init__()

        self.common = tf.keras.Sequential([
            Dense(128, activation='relu')
        ])
        self.mu = Dense(1)
        self.sigma = Dense(1, activation='softplus')
        self.critic = Dense(1)

        self.action_range = action_range

    def call(self, inputs, **kwargs):
        x = self.common(inputs)
        n = tfp.distributions.Normal(self.mu(x), self.sigma(x))
        action = tf.clip_by_value(n.sample(1), *self.action_range)
        action_log_prob = n.log_prob(action)

        return action, action_log_prob, self.critic(x)


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
               actor: tf.keras.Model,
               critic: tf.keras.Model,
               opt1: tf.keras.optimizers.Optimizer,
               opt2: tf.keras.optimizers.Optimizer):

    max_steps = int(1e8)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    state = initial_state
    for t in tf.range(max_steps):
        with tf.GradientTape(persistent=True) as tape:
            # Run
            state = tf.expand_dims(tf.squeeze(state), 0)
            action, action_log_prob = actor(state)
            value = critic(state)
            state, reward, done = tf_env_step(action)

            # return(advantage), Q, TD
            state = tf.expand_dims(tf.squeeze(state), 0)
            next_value = critic(state)

            loss_actor, loss_critic = calc_loss(action_log_prob, next_value, reward, value)

        grads = tape.gradient(loss_actor, actor.trainable_variables)
        opt1.apply_gradients(zip(grads, actor.trainable_variables))
        grads = tape.gradient(loss_critic, critic.trainable_variables)
        opt2.apply_gradients(zip(grads, critic.trainable_variables))

        rewards = rewards.write(t, reward)
        del tape
        if tf.cast(done, tf.bool):
            break

    rewards = rewards.stack()
    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward


def calc_loss(action_log_prob, next_value, reward, value):
    target = reward + 0.99 * next_value  # r +\gamma * V(s_{t+1})
    td1_error = target - value
    loss_actor = -tf.math.reduce_sum(action_log_prob * td1_error)
    loss_critic = tf.keras.losses.MeanSquaredError()(value, target)
    return loss_actor, loss_critic

def main():
    action_range = [env.action_space.low.item(), env.action_space.high.item()]
    actor = Actor([128], action_range)
    critic = Critic([128])
    # model = ActorCritic(action_range)
    # Training loop

    max_episodes = 100
    running_reward = 0
    opt1 = Adam(learning_rate=0.001)
    opt2 = Adam(learning_rate=0.001)

    with tqdm.trange(max_episodes) as t:
        for i in t:
            initial_state = tf.constant(env.reset(), dtype=tf.float32)
            episode_reward = float(train_step(initial_state, actor, critic, opt1, opt2))
            running_reward = episode_reward * .01 + running_reward * .99
            t.set_description(f'Episode {i}')
            t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)


if __name__ == "__main__":
    main()
