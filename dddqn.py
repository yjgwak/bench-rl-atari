import tensorflow as tf
import tf_agents
import os
import numpy as np
from random import random
from tensorflow import TensorSpec
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.losses import Huber
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tf_agents.replay_buffers.py_uniform_replay_buffer import PyUniformReplayBuffer
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer

from game import GameEnvironment


def build_q(n_actions, input_shape=(84, 84)):
    frame = Input(shape=input_shape)
    x = ((frame / 255.) - 0.5) * 2
    x = Conv2D(32, (8, 8), strides=4, activation='relu')(x)
    x = Conv2D(64, (4, 4), strides=2, activation='relu')(x)
    x = Conv2D(64, (3, 3), strides=1, activation='relu')(x)
    x = Conv2D(1024, (7, 7), strides=1, activation='relu')(x)
    x = LSTM(1024)(x)

    val_x, adv_x = tf.split(x, 2, 3)
    val_x = Flatten()(val_x)
    val = Dense(1)(val_x)

    adv_x = Flatten()(adv_x)
    adv = Dense(n_actions)(adv_x)
    q_vals = val + (adv - tf.reduce_mean(adv, axis=1, keepdims=True))
    model = Model(frame, q_vals)
    model.compile(optimizer=Adam(), loss=Huber())
    return model


class MyReplayBuffer(tf_agents.replay_buffers.replay_buffer.ReplayBuffer):

    def __init__(self, data_spec, capacity, stateful_dataset=False):
        """Initializes the replay buffer.

        Args:
          data_spec: A spec or a list/tuple/nest of specs describing a single item
            that can be stored in this buffer
          capacity: number of elements that the replay buffer can hold.
          stateful_dataset: whether the dataset contains stateful ops or not.
        """
        super(MyReplayBuffer, self).__init__(data_spec, capacity, stateful_dataset)

    def _num_frames(self):
        """Returns the number of frames in the replay buffer."""
        raise NotImplementedError

    def _add_batch(self, items):
        pass

    def _get_next(self, sample_batch_size, num_steps, time_stacked):
        """Returns an item or batch of items from the buffer."""
        raise NotImplementedError

    def _gather_all(self):
        """Returns all the items in buffer."""
        raise NotImplementedError

    def _clear(self):
        """Clears the replay buffer."""
        raise NotImplementedError

    def _as_dataset(self,
                    sample_batch_size,
                    num_steps,
                    sequence_preprocess_fn,
                    num_parallel_calls):
        """Creates and returns a dataset that returns entries from the buffer."""
        raise NotImplementedError


class ReplayBuffer:
    """Replay Buffer to store transitions.
    This implementation was heavily inspired by Fabio M. Graetz's replay buffer
    here: https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb"""
    def __init__(self, size=1000000, input_shape=(84, 84), history_length=4, use_per=True):
        """
        Arguments:
            size: Integer, Number of stored transitions
            input_shape: Shape of the preprocessed frame
            history_length: Integer, Number of frames stacked together to create a state for the agent
            use_per: Use PER instead of classic experience replay
        """
        self.size = size
        self.input_shape = input_shape
        self.history_length = history_length
        self.count = 0  # total index of memory written to, always less than self.size
        self.current = 0  # index to write to

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.input_shape[0], self.input_shape[1]), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)
        self.priorities = np.zeros(self.size, dtype=np.float32)

        self.use_per = use_per

    def add_experience(self, action, frame, reward, terminal, clip_reward=True):
        """Saves a transition to the replay buffer
        Arguments:
            action: An integer between 0 and env.action_space.n - 1
                determining the action the agent perfomed
            frame: A (84, 84, 1) frame of the game in grayscale
            reward: A float determining the reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if frame.shape != self.input_shape:
            raise ValueError('Dimension of frame is wrong!')

        if clip_reward:
            reward = np.sign(reward)

        # Write memory
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.priorities[self.current] = max(self.priorities.max(), 1)  # make the most recent experience important
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size

    def get_minibatch(self, batch_size=32, priority_scale=0.0):
        """Returns a minibatch of self.batch_size = 32 transitions
        Arguments:
            batch_size: How many samples to return
            priority_scale: How much to weight priorities. 0 = completely random, 1 = completely based on priority
        Returns:
            A tuple of states, actions, rewards, new_states, and terminals
            If use_per is True:
                An array describing the importance of transition. Used for scaling gradient steps.
                An array of each index that was sampled
        """

        if self.count < self.history_length:
            raise ValueError('Not enough memories to get a minibatch')

        # Get sampling probabilities from priority list
        if self.use_per:
            scaled_priorities = self.priorities[self.history_length:self.count-1] ** priority_scale
            sample_probabilities = scaled_priorities / sum(scaled_priorities)

        # Get a list of valid indices
        indices = []
        for i in range(batch_size):
            while True:
                # Get a random number from history_length to maximum frame written with probabilities based on priority weights
                if self.use_per:
                    index = np.random.choice(np.arange(self.history_length, self.count-1), p=sample_probabilities)
                else:
                    index = random.randint(self.history_length, self.count - 1)

                # We check that all frames are from same episode with the two following if statements.  If either are True, the index is invalid.
                if index >= self.current and index - self.history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.history_length:index].any():
                    continue
                break
            indices.append(index)

        # Retrieve states from memory
        states = []
        new_states = []
        for idx in indices:
            states.append(self.frames[idx-self.history_length:idx, ...])
            new_states.append(self.frames[idx-self.history_length+1:idx+1, ...])

        states = np.transpose(np.asarray(states), axes=(0, 2, 3, 1))
        new_states = np.transpose(np.asarray(new_states), axes=(0, 2, 3, 1))

        if self.use_per:
            # Get importance weights from probabilities calculated earlier
            importance = 1/self.count * 1/sample_probabilities[[index - self.history_length for index in indices]]
            importance = importance / importance.max()

            return (states, self.actions[indices], self.rewards[indices], new_states, self.terminal_flags[indices]), importance, indices
        else:
            return states, self.actions[indices], self.rewards[indices], new_states, self.terminal_flags[indices]

    def set_priorities(self, indices, errors, offset=0.1):
        """Update priorities for PER
        Arguments:
            indices: Indices to update
            errors: For each index, the error between the target Q-vals and the predicted Q-vals
        """
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset

    def save(self, folder_name):
        """Save the replay buffer to a folder"""

        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        np.save(folder_name + '/actions.npy', self.actions)
        np.save(folder_name + '/frames.npy', self.frames)
        np.save(folder_name + '/rewards.npy', self.rewards)
        np.save(folder_name + '/terminal_flags.npy', self.terminal_flags)

    def load(self, folder_name):
        """Loads the replay buffer from a folder"""
        self.actions = np.load(folder_name + '/actions.npy')
        self.frames = np.load(folder_name + '/frames.npy')
        self.rewards = np.load(folder_name + '/rewards.npy')
        self.terminal_flags = np.load(folder_name + '/terminal_flags.npy')


class Agent:
    def __init__(self, replay_buffer):
        eps_initial = 1
        eps_final = 0.1
        eps_final_frame = 0.01
        eps_evaluation = 0.0
        eps_annealing_frames = 1000000
        replay_buffer_start_size = 50000
        max_frames = 25000000

        self.rb = MyReplayBuffer(data_spec=data_spec, batch_size=batch_size, max_length=int(1e5))
        self.replay_buffer_start_size = replay_buffer_start_size
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames
        self.max_frames = max_frames

        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope * self.replay_buffer_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame) / (
                self.max_frames - self.eps_annealing_frames - self.replay_buffer_start_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2 * self.max_frames

    def learn(self):
        pass

    def get_epsilon(self, frame_number):
        if frame_number < self.replay_buffer_start_size:
            return self.eps_initial
        elif self.replay_buffer_start_size <= frame_number < self.replay_buffer_start_size + self.eps_annealing_frames:
            return self.slope * frame_number + self.intercept
        elif frame_number >= self.replay_buffer_start_size + self.eps_annealing_frames:
            return self.slope_2 * frame_number + self.intercept_2


if __name__ == "__main__":

    data_spec = (
        TensorSpec((84, 84), dtype=tf.uint8, name='state_t'),
        TensorSpec((), dtype=tf.uint8, name='action'),
        TensorSpec((), dtype=tf.float32, name='reward'),
        TensorSpec((84, 84), dtype=tf.uint8, name='state_t1'),
        TensorSpec((), dtype=tf.bool, name='terminal_flag')
    )
    capacity = 10000

    batch_size = 32

    MAX_EPOCHS = 100
    MAX_EPOCH_FRAME = 10000
    MAX_EPISODE_LENGTH = 1000
    ENV_NAME = "hello world"
    env = GameEnvironment(ENV_NAME)


    agent = Agent()
    for epoch in range(MAX_EPOCHS):
        epoch_frame = 0
        while epoch_frame < MAX_EPOCH_FRAME:
            frame_number = 0
            env.reset()
            action = agent.get_action(frame_number)
            state, reward, terminal, life_lost = agent.step(action)
            frame_number += 1
            epoch_frame += 1
            agent.add_experience(action=action,
                                 frame=state[:, :, 0],
                                 reward=reward, clip_reward=CLIP_REWARD,
                                 terminal=life_lost)
            agent.learn()
            agent.update()
            if terminal:
                break








