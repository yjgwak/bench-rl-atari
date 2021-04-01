import tensorflow as tf
from tf.keras.layers import Dense, Lambda, Input
import numpy as np



class Agent:
    def __init__(self, env_name="MountainCarContinuous-v0"):
        self.env = gym.make(env_name)

    def _env_step(self, action):
        state, reward, done, _ = env.step(action)
        return state.astype(np.float32), np.array(reward, np.float32), np.array(done, np.bool)

    def _tf_env_step(self, action):
        return tf.numpy_function(self._env_step, [action], [tf.float32, tf.float32, tf.bool])

    def build_actor(self, param):
        action_range = [env.action_space.low.item(), env.action_space.high.item()]
        ActorCritic(param, action_range)
        init_normal = tf.keras.initializers.GlorotNormal()
        self.actor_pre = tf.keras.Sequential([
            Dense(param['actor'][0], activation='relu', kernel_initializer=init_normal),
            Dense(param['actor'][1], activation='relu', kernel_initializer=init_normal)
        ])
        self.mu = Dense(param['actor'][2], activation='tanh', kernel_initializer=init_normal)
        self.sigma = Dense(param['actor'][2], activation='sigmoid', kernel_initializer=init_normal)

        pass

    def build_optimizers(self):
        pass


class ActorCritic(tf.keras.Model):
    def __init__(self, param: dict, action_range):
        super().__init__()

        init_normal = tf.keras.initializers.GlorotNormal()
        self.actor_pre = tf.keras.Sequential([
            Dense(param['actor'][0], activation='relu', kernel_initializer=init_normal),
            Dense(param['actor'][1], activation='relu', kernel_initializer=init_normal)
        ])
        self.mu = Dense(param['actor'][2], activation='tanh', kernel_initializer=init_normal)
        self.sigma = Dense(param['actor'][2], activation='sigmoid', kernel_initializer=init_normal)

        self.critic = tf.keras.Sequential([
            Dense(param['critic'][0], activation='relu', kernel_initializer=init_normal),
            Dense(param['critic'][1], activation='relu', kernel_initializer=init_normal),
            Dense(param['critic'][2], kernel_initializer=init_normal)
        ])

        self.action_range = action_range

    def call(self, inputs, **kwargs):
        x = self.actor_pre(inputs)
        normal_dist = tfp.distributions.Normal(self.mu(x), self.sigma(x) + eps)
        action_norm = tf.squeeze(normal_dist.sample(1), axis=0)
        action_clipped = tf.clip_by_value(action_norm, *self.action_range)
        action_prob = normal_dist.prob(action_clipped) + eps

        return action_clipped, action_prob, self.critic(inputs)
