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
import sys

FIND_BEST_Q=False # runs all combinations of alpha and discount
FEDERATED=True # federated domain to find best combinations in Q-learning
FEDERATED_MULTIPLIER=8 # federated_res=xFEDERATED_MULTIPLIER*local_res
BEST_FILE='/tmp/alpha-combs-x8.json' # file to store best combs



def take_actions(actions, env):
    # actions: [0, 2, 1, 0, ...] 
    # env: environmnet to take actions
    #
    # returns: the SAR, i.e, sequences, actions, rewards

    env.reset()
    rewards = []
    state = env.get_state()
    state_sequence = [state]

    t, i = 0, 0
    while state != None:
        logging.info(f't={t} test')

        # execute selected action in the environment
        st_a = time.time()
        reward, state = env.take_action(actions[i])
        logging.debug(f'it takes {time.time() - st_a} seconds to act')
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
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--ampl_out', type=str, default='/tmp/model',
                        help='file with timestamp|instance|local|fed|reject')
    args = parser.parse_args()


    # Set the logger
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    # Check arguments
    if not args.ampl_out:
        logging.info(f'please provide AMPL decisions')
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

    # Load the AMPL dataframe
    ampl_df = pd.read_csv(args.ampl_out)
    ampl_df = ampl_df[ampl_df['instance'].isin(instances)]
    ampl_df.sort_values(by='timestamp', inplace=True, ascending=True)


    # Take AMPL actions
    actions = []
    for _, row in ampl_df.iterrows():
       action = AWS_env.A_LOCAL
       if row['federate'] == 1:
           action = AWS_env.A_FEDERATE
       elif row['reject'] == 1:
           action = AWS_env.A_REJECT
       actions.append(action)
    states, actions, rewards = take_actions(actions, env)
    logging.info(f'total reward = {sum(rewards)}')
    logging.info('State|action|reward')
    for t in range(len(states)):
        logging.info(f'{states[t]}|{actions[t]}|{rewards[t]}')
    logging.info(f'total reward = {sum(rewards)}')
    sys.exit(0)



