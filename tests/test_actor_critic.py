from unittest import TestCase
# from mountain import ActorCritic
import tensorflow as tf
import numpy as np
import gym  # requires OpenAI gym installed
# import tqdm as tqdm
from tensorflow.python.keras.layers import Dense
# from tensorflow.python.keras.losses import Huber
# from tensorflow.python.keras.optimizer_v2.adam import Adam
import tensorflow_probability as tfp

import common


class Test(TestCase):
    def test_env(self):
        env = gym.make("MountainCarContinuous-v0")
        eps = common.eps.item()

        class ActorCritic(tf.keras.Model):
            def __init__(self, param: dict):
                super(ActorCritic, self).__init__()

                self.actor_pre = tf.keras.Sequential([
                    Dense(param['actor'][0], activation='relu'),
                    Dense(param['actor'][1], activation='relu')
                ])
                self.mu = Dense(param['actor'][2])
                self.sigma = Dense(param['actor'][3])

                self.critic = tf.keras.Sequential([
                    Dense(param['critic'][0], activation='relu'),
                    Dense(param['critic'][1], activation='relu'),
                    Dense(param['critic'][2])
                ])

                self.action_range = [env.action_space.low.item(), env.action_space.high.item()]

            def call(self, inputs):
                x = self.actor_pre(inputs)
                normal_dist = tfp.distributions.Normal(self.mu(x), self.sigma(x) + eps)
                action_norm = tf.squeeze(normal_dist.sample(1), axis=0)
                action_clipped = tf.clip_by_value(action_norm, *self.action_range)

                return action_clipped, self.critic(inputs), normal_dist

        def env_step(action):  # np.ndarray):# -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            state, reward, done, _ = env.step(action)
            return (state.astype(np.float32),
                    np.array(reward, np.float32),
                    np.array(done, np.int32))

        def tf_env_step(action):  # tf.Tensor) -> List[tf.Tensor]:
            return tf.numpy_function(env_step, [action], [tf.float32, tf.float32, tf.int32])

        layer_param = {'actor': [40, 40, 1, 1], 'critic': [400, 400, 1]}
        model = ActorCritic(layer_param)
        state = env.reset()
        state = tf.expand_dims(state, 0)
        action, value, norm = model(state)
        action = action[0]
        print(model(state))
        print(tf_env_step(action))
        self.fail()
