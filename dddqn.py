import tensorflow as tf
from tensorflow import TensorSpec
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.losses import Huber
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer


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


class ReplayBuffer(TFUniformReplayBuffer):
    def __init__(self, **kwargs):
        super(ReplayBuffer, self).__init__(**kwargs)


class Agent:
    def __init__(self):
        rb = ReplayBuffer(data_spec=data_spec, batch_size=batch_size, max_length=int(1e5))


    def learn(self):
        pass


if __name__ == "__main__":
    data_spec = (
        TensorSpec((), dtype=tf.uint8, name='action'),
        TensorSpec((), dtype=tf.float32, name='reward'),
        TensorSpec((84, 84), dtype=tf.uint8, name='frame'),
        TensorSpec((), dtype=tf.bool, name='terminal_flag')
    )

    batch_size = 32
