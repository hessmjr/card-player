# -*- coding: utf-8 -*-
import operator
import random

import gym

env = gym.make("Blackjack-v0")

# Q-table
q_table = {}

# Hyperparameters (alpha,gamma,epilson)
alpha = 0.1
gamma = 0.8
epsilon = 0.2
epsilon_decay = 0.999

# Plotting metrix
reward_list = []

episode_number = 1000001
for i in range(0, episode_number):
    # Initialize enviroment
    state = env.reset()
    reward_count = 0

    while True:
        # Episode vs Explore to find action

        if random.uniform(0, 1) < epsilon or state not in q_table:
            action = env.action_space.sample()
            epsilon *= epsilon_decay
        else:
            # action = np.argmax(q_table[state])
            action = max(q_table[state].items(), key=operator.itemgetter(1))[0]

        # print('state: ' + str(state))
        # print('action: ' + str(action))

        # action process take reward/observation
        next_state, reward, done, info = env.step(action)

        # Q-Learning Function
        old_value = 0.0
        if state in q_table and action in q_table[state]:
            old_value = q_table[state][action]

        if next_state in q_table:
            next_max = max(q_table[next_state].keys())
        else:
            next_max = 0.0
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        # Q_table update
        if state not in q_table:
            q_table[state] = {}
        q_table[state][action] = new_value

        # update state
        state = next_state
        reward_count += reward

        if done:
            reward_list.append(reward_count)

            if i % 100 == 0:
                print("Episode: {} , Reward: {} ".format(i, sum(reward_list)))
            break

print("Final reward: {}".format(sum(reward_list)))
print(q_table)
