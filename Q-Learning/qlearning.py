
import numpy as np
import pandas as pd
import time


N_STATES = 40   # the length of the 1 dimensional world
ACTIONS = ['left', 'right']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 13   # maximum episodes
FRESH_TIME = 0    # fresh time for one move


def init_q_table(n_states, actions):
    qtable = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    return qtable


def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name


def agent_get_feedback_from_environment(S, A):
    # The agent will get environment feedback
    # Only the most-right side has reward. 
    if A == 'right':    # move right
        if S == N_STATES - 2:   # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:   # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R


def update_environment(S, episode, total_steps):
    # This is how environment be updated, so that you can see how the environment look like from python console output 
    env_list =['[']+ ['-']*(N_STATES-1) + [']']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, total_steps)
        print('\r{}'.format(interaction), end='')
        time.sleep(0.1)
        print('\r\n                                ', end='\n')
    else:
        env_list[S+1] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def q_learning_algorithm():
    # main part of Q-learning loop
    q_table = init_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        total_steps = 0
        S = 0
        is_terminated = False
        update_environment(S, episode, total_steps)
        while not is_terminated:

            A = choose_action(S, q_table)
            S_, R = agent_get_feedback_from_environment(S, A)  # take action & get next state and reward
            q_predict = q_table.ix[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.ix[S, A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next state

            update_environment(S, episode, total_steps+1)
            total_steps += 1
    return q_table


if __name__ == "__main__":
    q_table = q_learning_algorithm()
    print('\r\nQ-table:\n')
    print(q_table)
