# https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

import sys
import random
import pprint
from time import sleep

import gym
import numpy as np

# Configuration
np.set_printoptions(threshold=np.inf, suppress=True)
np.set_printoptions(threshold=sys.maxsize, suppress=True)

env = gym.make('Taxi-v2').env


def get_info():
    env.reset()
    print(env.render())

    print('Action Space {}'.format(env.action_space))
    print('State Space {}'.format(env.observation_space))


# Initiating environment's state
state = env.encode(3, 1, 2, 0)
env.s = state  # 328
print(env.s)

# The reward matrix for the state 328
pprint.pp(env.P[328])


def solve_without_rl():
   
    epochs = 0
    penalties, rewards = 0, 0

    frames = []  # for animation

    done = False

    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        # Put each rendered frame into dict for animation
        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
        })

        epochs += 1

    for i, frame in enumerate(frames):
        print(frame['frame'].getvalue())
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.2)

    print("Timesteps taken: {}".format(epochs))
    print("Penalties incurred: {}".format(penalties))


def solve_q_learning():
    # Training the agent

    # Init Q-table 500 x 6
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # Hyperparameters
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1

    for i in range(1, 100001):
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_table[state])  # Exploit learned values

            next_state, reward, done, info = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1

        if i % 100 == 0:
            print(f"Episode: {i}")

    print("Training finished.\n")
    print(q_table[328])

    # Evaluate agent's performance after Q-learning
    total_epochs, total_penalties = 0, 0
    episodes = 100

    for _ in range(episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0

        done = False

        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)

            if reward == -10:
                penalties += 1

            epochs += 1

        total_penalties += penalties
        total_epochs += epochs

    print(f"Results after {episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
    print(f"Average penalties per episode: {total_penalties / episodes}")


# solve_without_rl()
solve_q_learning()

env.close()
