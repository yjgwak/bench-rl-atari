import gym
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils.common import element_wise_squared_loss

from game import GameEnvironment

env_name = 'BreakoutDeterministic-v4'
train_env = GameEnvironment(env_name)
optimizer = Adam()

q_net = q_network.QNetwork(
  train_env.observation_spec(),
  train_env.action_spec(),
  fc_layer_params=(100,))

agent = dqn_agent.DqnAgent(
  train_env.time_step_spec(),
  train_env.action_spec(),
  q_network=q_net,
  optimizer=optimizer,
  td_errors_loss_fn=element_wise_squared_loss,
  train_step_counter=tf.Variable(0))

agent.initialize()
