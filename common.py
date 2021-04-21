import numpy as np
import tensorflow as tf

eps = np.finfo(np.float32).eps.item()
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


def compute_loss(action_probs, values, returns):
    advantage = returns - values
    action_log_probs = tf.math.log(action_probs)

    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
    critic_loss = huber_loss(values, returns)

    return actor_loss + critic_loss


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
        returns = ((returns - tf.math.reduce_mean(returns)) /
                (tf.math.reduce_std(returns) + eps))
    return returns


def get_td_return(rewards, values, gamma):
    values = values[1:]
    values = tf.concat([values, tf.constant([0], dtype=tf.float32)], axis=0)

    rewards = tf.cast(rewards, dtype=tf.float32)
    returns = rewards + (gamma * values)
    return returns


def get_tdl_return(rewards, values, gamma, ld=0.99):
    # TODO: express as matrix form
    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)
