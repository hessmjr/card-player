# -*- coding: utf-8 -*-
import operator
import random

import gym

env = gym.make("HotterColder-v0")

# Q-table
q_table = {}

# Hyperparameters (alpha,gamma,epsilon)
alpha = 0.1
gamma = 0.8
epsilon = 0.1

# Plotting metrix
reward_list = []

episode_number = 10000
for i in range(1, episode_number):
    # Initialize enviroment
    state = env.reset()
    reward_count = 0

    while True:
        # Episode vs Explore to find action

        if random.uniform(0, 1) < epsilon or state not in q_table:
            action = env.action_space.sample()[0]
        else:
            action = max(q_table[state].items(), key=operator.itemgetter(1))[0]

        # print('state: ' + str(state))
        # print('action: ' + str(action))

        # action process take reward/observation
        next_state, reward, done, info = env.step([action])
        sr = state * reward

        # Q-Learning Function
        old_value = 0.0
        if sr in q_table and action in q_table[sr]:
            old_value = q_table[sr][action]

        if next_state in q_table:
            next_max = max(q_table[next_state].keys())
        else:
            next_max = 0.0
        next_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        # Q_table update
        if sr not in q_table:
            q_table[sr] = {}
        q_table[sr][action] = next_value

        # update state
        state = next_state

        if done:
            reward_list.append(reward_count)

            if i % 10 == 0:
                print("Episode: {} , Reward: {} ".format(i, sum(reward_list)))
            break

print("Final reward: {}".format(sum(reward_list)))
