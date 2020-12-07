import gym
import gym_se

env = gym.make('se-dummy-v0')
env.reset()
for _ in range(1000):
    env.render()
    observation, _, _, _ = env.step(env.action_space.sample())
    print(observation)
env.close()
