import gym

env = gym.make('Blackjack-v0')

for i in range(20):
    obs = env.reset()
    for t in range(10):
        # env.render()
        print(obs)
        a = env.action_space.sample()
        print(a)
        obs, reward, done, info = env.step(a)
        print(reward)
        # print(info)
        if done:
            print('finished------')
            break

env.close()

