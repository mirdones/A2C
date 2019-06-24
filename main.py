import gym
import matplotlib.pyplot as plt
from Model import *
from A2CAgent import *

logging.getLogger().setLevel(logging.INFO)

env = gym.make('CartPole-v0')
model = Model(num_actions=env.action_space.n)
agent = A2CAgent(model)
agent.model.build(input_shape=(1, 4))
print(agent.model.summary())
model.load_weights('./checkpoints/my_checkpoint')
rewards_history = agent.train(env)
agent.model.save_weights('./checkpoints/my_checkpoint')
print("Finished training.")
print("Total Episode Reward: %d out of 200" % agent.test(env, True))

plt.style.use('seaborn')
plt.plot(np.arange(0, len(rewards_history), 25), rewards_history[::25])
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
