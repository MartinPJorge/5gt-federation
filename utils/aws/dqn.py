import argparse
import json
import numpy as np
import datetime
import pandas as pd
import tensorflow as tf
from tensorflow import keras


class AWS_env():
    def __init__(self, cpu, memory, disk,
                 f_cpu=math.inf, f_memory=math.inf, f_disk=math.inf,
                 arrivals):
        self.cpu = cpu
        self.disk = disk
        self.memory = memory
        self.f_cpu = f_cpu
        self.f_disk = f_disk
        self.f_memory = f_memory
        self.time = float(0)
        self.capacity = [int(cpu), int(memory), int(disk)]
        self.f_capacity = [int(f_cpu), int(f_memory), int(f_disk)] if\
            f_cpu != math.inf else [math.inf, math.inf, math.inf]
        self.profit = float(0)
        self.service_cpu = []
        self.service_disk = []
        self.service_memory = []
        self.service_arrival_time = [] # TODO: this'll be a data-frame
                                       # with arrival times, and resource reqs
        self.service_length = []
        self.federated_service_cpu = []
        self.federated_service_disk = []
        self.federated_service_memory = []
        self.federated_service_arrival = []
        self.federated_service_length = []
        self.total_num_services = int(0)

    def get_state(self):
        # returns the curr state [cpu, mem, disk, f_cpu, f_mem, f_disk]
        pass

    def take_action(self, a):
        # a={1,2,3} local deployment, federate, reject
        # returns the reward and next state
        pass

    def reset(self):
        # go back to the initial state
        pass





def create_q_network(k):
    # The NN recives as input
    #  k-[loc_cpu,loc_mem,loc_disk, fed_cpu,fed_mem,fed_disk]
    # and that gives the state representation.
    # Then it should give as output x3 Q-values, one per action:
    #
    #                               --->  Q(state, accept)
    #                              /
    # (state) --->  NN   ------> Q(state, reject) 
    #                              \
    #                               ----> Q(state, federate)
    #
    # return: the tf.model for the NN

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(k*6)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(3)
    ])

    print('Just have created the NN, check below the TF summary')
    model.summary()

    return model



def train_q_network(epsilon, gamma, M, env):
    # Implement the training specified in Algorithm 1 of
    # "Playing Atari with Deep Reinforcement Learning"
    #
    # It uses an epsilon-greedy behaviour-policy, and
    # gamma as discounted reward factor. The process uses
    # the reported environment and repeats each episode M
    # times.

    # TODO: it is necessary to have an environment slightly
    #       different than the one in environment.py



if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Plot EC2 spot prices')
    parser.add_argument('prices_csvs', type=str,
                        help='ec2_aws_csv_1|ec2_aws_csv2|... list of CSVs' +\
                             'for testing is just one CSV')
    parser.add_argument('instance_types', type=str,
                        help='|-separated list of instances: ' +\
                            't3a.nano|t3a.small|...\n' +\
                            'or * wildcard to plot all')
    parser.add_argument('k', type=int, help='size of history to represent' +\
                                            'the state')
    parser.add_argument('epsilon', type=float,
                        help='epsilon-greedy for the off-policy')
    parser.add_argument('gamma', type=float,
                        help='discounted factor for the reward')
    parser.add_argument('train', type=bool, help='True|False')
    parser.add_argument('out_weights', type=str, default='/tmp/weights',
                        help='Path where the DQN weights are stored')
    args = parser.parse_args()



    # Check arguments
    if args.epsilon > 1 or args.epsilon < 0:
        print(f'epsion={args.epsilon}, but it must belong to [0,1]')
        sys.exit(1)
    if args.k < 0:
        print(f'k={k}, but it must be >0')
        sys.exit(1)
    if args.gamma > 1 or args.gamma < 0:
        print(f'epsion={args.gamma}, but it must belong to [0,1]')
        sys.exit(1)


    # Get the instances
    instances = list(prices_df['InstanceType'].unique())\
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
    




