import cv2
import gym
import numpy as np
from tensorflow.python.framework.tensor_spec import BoundedTensorSpec, TensorSpec
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.typing.types import Tensor

ENV_NAME = 'BreakoutDeterministic-v4'


class GameEnvironment:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.state = None
        self.last_lives = 0

    def reset(self):
        self.state = process_frame(self.env.reset())
        self.last_lives = 0

    def step(self, action, render_mode=None):
        state, reward, terminal, info = self.env.step(action)

        if info['ale.lives'] < self.last_lives:
            life_lost = True
        else:
            life_lost = terminal
        self.last_lives = info['ale.lives']

        self.state = process_frame(state)

        if render_mode == 'rgb_array':
            return self.state, reward, terminal, life_lost, self.env.render(render_mode)
        elif render_mode == 'human':
            self.env.render()

        return self.state, reward, terminal, life_lost

    def observation_spec(self):
        return TensorSpec(shape=self.env.observation_space.shape, dtype=self.env.observation_space.dtype)

    def action_spec(self):
        return BoundedTensorSpec(shape=(), dtype=self.env.action_space.dtype,
                                 minimum=0, maximum=self.env.action_space.n-1)

    # def time_step_spec(self):
    #     return TimeStep(step_type=Tensor, reward=)


def process_frame(frame, shape=(84, 84)):
    frame = frame.astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame[34:34+160, :160]
    frame = cv2.resize(frame, shape, interpolation=cv2.ITER_NEAREST)
    frame = frame.reshape((*shape, 1))

    return frame


if __name__ == "__main__":
    env = gym.make()
    env.render()

