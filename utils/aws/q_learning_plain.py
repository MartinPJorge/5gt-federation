import time
import argparse
import json
import numpy as np
import datetime
import pandas as pd
import random
import logging
import math
from aws_env import AWS_env
import matplotlib as mpl
from matplotlib import pyplot as plt
import sys

FIND_BEST_Q=False # runs all combinations of alpha and discount
FEDERATED=True # federated domain to find best combinations in Q-learning
FEDERATED_MULTIPLIER=8 # federated_res=xFEDERATED_MULTIPLIER*local_res
BEST_FILE='/tmp/alpha-combs-x8.json' # file to store best combs

def initialize_q_table():
    qtable = {
    # ( local, federate, arrival, reward, price): {
    #  "local": 0,
    #  "federate": 0,
    #  "reject": 0
    # }

    ( local, federate, arrival, reward, price): [0, 0, 0]

    for local in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for federate in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for arrival in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for reward in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for price in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    }
    return qtable

def adapt_number(number):
    if number > 1:
        return 1

    # number = number*1000
    x = math.ceil(number*100)*0.01
    y = x*1000%10
    b = 1 if y>=5 else 0
    result = int(x*10)/10+(b*0.1)
    return result


def state_to_row(state):
    local = adapt_number((state[0]+state[1] + state[2])/3)
    federate = adapt_number((state[3]+state[4] + state[5])/3)
    arrival = adapt_number(max(state[6], state[7], state[8]))
    reward = adapt_number(state[9])
    price = adapt_number(state[10])

    result = ( local, federate, arrival, reward, price)
    
    return result

# def state_to_vector(qtable, state):
#    return [qtable[state_to_row(state)]['local'], qtable[state_to_row(state)]['federate'], qtable[state_to_row(state)]['reject']]    

def calculate_total_states(cpu, memory, disk, f_cpu, f_disk, f_memory):

    return tot_states


def q_learning(env, alpha, discount, episodes,
               epsilon_start=None, epsilon_end=None, out= None):

    qtable = initialize_q_table()
    tot_actions = 3

    episodes_rewards = []
    actions = []

    for episode in range(episodes):
        env.reset()
        curr_state, next_state = env.get_state(), env.get_state()
        sim_active = True
        episode_reward = 0
        t = 0

        logging.info('EPISODE=%d' % episode)

        # Derive the episode epsilon
        if epsilon_start != None and epsilon_end != None:
            epsilon = (episode+1) / episodes * (epsilon_end - epsilon_start)\
                      + epsilon_start

        while next_state != None:
            start_interval = time.time()
            t = t + 1
            logging.debug(f'\nt={t}\t')
            logging.debug(f'Q-network at t={t}')
            action = 0

            if epsilon_start != None and epsilon_end != None:
                action = np.argmax(qtable[state_to_row(curr_state)])\
                         if np.random.random() < epsilon else\
                         np.random.randint(tot_actions)
            else:
                action = np.argmax(qtable[state_to_row(curr_state)] +\
                                   np.random.randn(1, tot_actions) * (1 / float(episode + 1)))
            logging.debug(f'Action taken={action}')
            start_action = time.time()
            reward, next_state = env.take_action(action)
            episode_reward += reward
            # print(f'time action = {time.time() - start_action}')
            # print(f'action={action},reward={reward}, current_state={curr_state},next_state={next_state}')
            # print(f'time interval = {time.time() - start_interval}')
            if next_state == None:
                logging.debug("Qtable(current_state):", qtable[state_to_row(curr_state)])
                break
            else:
                qtable[state_to_row(curr_state)][action] += alpha * (reward + discount * np.max(qtable[state_to_row(next_state)]) - qtable[state_to_row(curr_state)][action])
            curr_state = next_state

        episodes_rewards.append(episode_reward)
    
    if out != None:
        logging.info('Storing the q-table')
        logging.info(out)
        with open(out, 'w') as fp:
            json.dump({str(st): qtable[st] for st in qtable}, fp)
    
    
    return episodes_rewards



def test_q_table(qtable, env):
    # qtable: the Q-table to test
    # env: environmnet to take actions
    #
    # returns: the SAR, i.e, sequences, actions, rewards

    env.reset()
    rewards, actions = [], []
    state = env.get_state()
    state_sequence = [state]

    t = 0
    while state != None:
        logging.info(f't={t} test')
        # Select the maximum action for the Q(·)
        action = np.argmax(qtable[state_to_row(state)])
        actions += [action]

        # execute selected action in the environment
        st_a = time.time()
        reward, state = env.take_action(actions[-1])
        logging.info(f'it takes {time.time() - st_a} seconds to act')
        rewards += [reward]
        state_sequence += [state]
        t += 1

    return state_sequence[-(len(actions)+1):-1], actions, rewards
    


if __name__ == '__main__':
    #Parse arguments
    parser = argparse.ArgumentParser(description='Plot EC2 spot prices')
    parser.add_argument('prices_csvs', type=str,
                        help='ec2_aws_csv_1|ec2_aws_csv2|... list of CSVs' +\
                                'for testing is just one CSV')
    parser.add_argument('instance_types', type=str,
                        help='|-separated list of instances: ' +\
                            't3a.nano|t3a.small|...\n' +\
                            'or * wildcard to plot all')
    parser.add_argument('arrivals', type=str,
                        help='path to CSV with arrivals dataframe')
    parser.add_argument('domains', type=str,
                        help='path to JSON with local|federated resources')
    # Training arguments
    parser.add_argument('--epsilon_start', type=float,
                        help='epsilon-greedy 1st value for the off-policy')
    parser.add_argument('--epsilon_end', type=float,
                        help='epsilon-greedy last value for the off-policy')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--gamma', type=float,
                        help='discounted factor for the reward')
    parser.add_argument('--alpha', type=float,
                        help='learning rate of optimizer')
    parser.add_argument('--M', type=int, help='number of episodes')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--in_model', type=str, default='/tmp/model',
                        help='Path where the trained DQN model is stored')
    parser.add_argument('--out_model', type=str, default='/tmp/model',
                        help='Path where the trained DQN model is stored')
    args = parser.parse_args()


    # Set the logger
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    logging.info(args.epsilon_start, args.epsilon_end)
    # Check arguments
    logging.info(f'traIN={args.train}')
    if args.train == True:
        if args.gamma > 1 or args.gamma < 0:
            logging.info(f'gamma={args.gamma}, but it must belong to [0,1]')
            sys.exit(1)
        if args.alpha > 1 or args.alpha < 0:
            logging.info(f'gamma={args.alpha}, but it must belong to [0,1]')
            sys.exit(1)    
    else:
        if not args.in_model:
            logging.info(f'in_model parameter missing, required for testing')
            sys.exit(1)

    
    # Get the instances
    instances = list(pd.read_csv(args.prices_csvs)['InstanceType'].unique())\
            if args.instance_types == '*'\
            else args.instance_types.split('|')

    # Load AWS prices CSVs
    prices_dfs = []
    for prices_csv in args.prices_csvs.split('|'):
        prices_df = pd.read_csv(prices_csv)
        prices_df['Timestamp'] = pd.to_datetime(prices_df['Timestamp'])
        # Filter the asked instances
        prices_df = prices_df[prices_df['InstanceType'].isin(instances)]
        prices_dfs.append(prices_df)

   
    # Load domains JSON with their resources
    with open(args.domains) as fp:
        domain = json.load(fp)

    # Filter out those arrivals of non-specified instances
    arrivals = pd.read_csv(args.arrivals)
    arrivals = arrivals[arrivals['instance'].isin(instances)]

    # Create the environment
    env = AWS_env(cpu=domain['local']['cpu'], memory=domain['local']['memory'],
            disk=domain['local']['disk'], f_cpu=domain ['federated']['cpu'],
            f_disk=domain['federated']['disk'],
            f_memory=domain['federated']['memory'],
            arrivals=arrivals,
            spot_prices=prices_df)
    
    # number = 0.799999999999999999999
    # print(math.ceil(number))
    # b =  math.ceil(number*1000)*0.001
    # print(b)

    # print(adapt_number(number))


    if args.train == True:
        episode_reward = q_learning(env=env, alpha= args.alpha,
                discount=args.gamma, episodes= args.M, out= args.out_model,
                epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end)
    else:
        with open(args.in_model, 'r') as fp:
            qtable = json.load(fp)
        qtable = {eval(st): qtable[st] for st in qtable.keys()}
        states, actions, rewards = test_q_table(qtable, env)
        logging.info('State|action|reward')
        for t in range(len(states)):
            logging.info(f'{states[t]}|{actions[t]}|{rewards[t]}')
        sys.exit(0)

    max_profit = max(episode_reward)
   

    logging.info("--------------- MAXIMUM PROFITS ---------------")

    logging.info("\tQ learning Federation: ", max(episode_reward))

    x = np.arange(0, len(episode_reward), 1)
    fig, ax = plt.subplots()
    plt.grid(linestyle='--', linewidth=0.5)
    plt.xlabel('episodes')
    plt.ylabel('normalized episode profit')
    plt.plot(x, [er/max_profit for er in episode_reward], label='Q-learning',
            color='C0', linewidth=4)

    plt.legend(loc='best', handlelength=4)
    
    logging.info("Total rewards: ", episode_reward)
    # filename = "../../../results/result.png"
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    # plt.savefig(filename)

    # plt.show()
    
