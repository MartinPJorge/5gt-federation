import time
import argparse
import json
import numpy as np
import datetime
import pandas as pd
import random
import math
from aws_env import AWS_env
import matplotlib as mpl
from matplotlib import pyplot as plt

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

    # number = number*1000
    x = math.ceil(number*100)*0.01
    y = x*1000%10
    b = 1 if y>=5 else 0
    result = int(x*10)/10+(b*0.1)
    return result


def state_return_action(state):
    action = 0
    local = []
    federate = []

    local = [a - b for a, b in zip(state[:3],state[6:])]
    federate = [a - b for a, b in zip(state[3:6],state[6:9])]
    if local[np.argmax(local)] <= 1.0 and local[np.argmin(local)]>0.0:
        action = 0
    elif federate[np.argmax(federate)] <= 1.0 and federate[np.argmin(federate)]>0.0:
        action = 1
    else:
        action = 2
    
    return action
    
def greedy(env, out= None):

    episode_reward = 0
    tot_actions = 3
    actions = []
    t = 0
    env.reset()
    next_state = env.get_state()


    while next_state != None:
        start_interval = time.time()
        t = t + 1
        print(f'\nt={t}\t')
        # print(f'Greedy-network at t={t}')
        # print(f'State:{next_state}')
        
        action = 0
        action = state_return_action(next_state)
        actions.append(action)
        print(f'Action taken={action}')
        start_action = time.time()
        
        reward, next_state = env.take_action(action)
        episode_reward += reward
        # print(f'time action = {time.time() - start_action}')
        # print(f'action={action},reward={reward}, current_state={curr_state},next_state={next_state}')
        # print(f'time interval = {time.time() - start_interval}')
        if next_state == None:
            print("Finished")
            break
        
     
    
    unique, counts = np.unique(actions, return_counts=True)
    total_actions_count = dict(zip(unique, counts))
    
    return episode_reward, total_actions_count

    


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
    args = parser.parse_args()

     # Get the instances
    instances = list(prices_df['InstanceType'].unique()) if args.instance_types == '*' else args.instance_types.split('|')

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
    
    reward, actions_count = greedy(env=env, out= None)

    print("--------------- MAXIMUM PROFITS ---------------")

    print("\tGreedy Federation: ", reward)
    print("\tActions count: ", actions_count)

    # x = np.arange(0, len(episode_reward), 1)
    # fig, ax = plt.subplots()
    # plt.grid(linestyle='--', linewidth=0.5)
    # plt.xlabel('episodes')
    # plt.ylabel('normalized episode profit')
    # plt.plot(x, [er/max_profit for er in episode_reward], label='Q-learning',
    #         color='C0', linewidth=4)

    # plt.legend(loc='best', handlelength=4)
    
    # print("Total rewards: ", episode_reward)
    # filename = "../../../results/result.png"
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    # plt.savefig(filename)

    # plt.show()
    
