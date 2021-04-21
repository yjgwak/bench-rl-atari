import numpy as np
import tensorflow as tf

eps = np.finfo(np.float32).eps.item()


def get_mc_return(rewards, values, gamma, standarize=True):
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
        returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps))
    return returns


def get_td_return(rewards, values, gamma):
    values = tf.concat([values[1:], tf.constant([0], dtype=tf.float32)], axis=0)
    rewards = tf.cast(rewards, dtype=tf.float32)
    returns = rewards + gamma * values
    return returns


def get_tdl_return(rewards, values, gamma, ld=0.99, standarize=True):
    returns_td = get_td_return(rewards, values, gamma)[::-1]

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    rewards = tf.cast(rewards[::-1], dtype=tf.float32)

    returns = returns.write(0, rewards[0])
    reward_constant = tf.constant(0.0, dtype=tf.float32)

    discounted_sum = rewards[0]
    discounted_sum_shape = discounted_sum.shape

    for i in tf.range(1, n):
        reward_constant = ld + ld * reward_constant
        discounted_sum = returns_td[i] + ld * gamma * discounted_sum + reward_constant * rewards[i-1]
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = (1-ld) * returns.stack()[::-1]

    if standarize:
        returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps))
    return returns


def compute_loss(action_probs, values, returns):
    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    critic_loss = huber_loss(values, returns)

    return actor_loss + critic_loss


