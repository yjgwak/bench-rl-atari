import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.losses import Huber
from tensorflow.python.keras.optimizer_v2.adam import Adam
import numpy as np


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


class ReplayBuffer:
    def __init__(self, size=int(1e5), input_shape=(84, 84)):
        self.size = size
        self.input_shape = input_shape
        self.count = 0
        self.current = 0

        # Allocate Memory
        self.actions = np.empty(self.size, dtype=np.uint8)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, *self.input_shape), dtype=np.uint8)
        self.terminal_falgs = np.empty(self.size, dtype=np.bool)

