import argparse
import json
import numpy as np
import datetime
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import random


# ACTIONs macros
A_LOCAL = 0
A_FEDERATE = 1
A_REJECT = 2
ACTIONS = [A_LOCAL, A_FEDERATE, A_REJECT]

# FUTURE
FUTURE = pd.Timestamp('2200-01-01 00:00:00+00:00')



class AWS_env():
    def __init__(self, cpu, memory, disk,
                 f_cpu=math.inf, f_memory=math.inf, f_disk=math.inf,
                 arrivals, spot_prices):
        # arrivals: pandas DataFrame with columns
        #           ['time', 'instance', 'spotprice', 'cpu', 'memory',
        #            'os', 'disk', 'reward', 'lifetime']
        #           the reward is expressed as $/hour
        #           lifetime in days
        #           time is the UNIX epoch in seconds since 1970
        # spot_prices: pandas DataFrame with columns
        #              ['AvailabilityZone', 'InstanceType',
        #               'ProductDescription', 'SpotPrice', 'Timestamp']

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
        self.arrivals = arrivals
        self.service_length = []
        self.total_num_services = int(0)
        # INDEXING
        self.curr_idx = 0 # current index in the arrivals DataFrame
        self.in_local = [] # list of arrival indexes in local domain
        self.in_fed = [] # list of arrival indexes in federated domain
        # SPOT PRICE
        self.spot_prices = spot_prices
        self.spot_prices['Timestamp'] =\
            pd.to_datetime(self.spot_prices['Timestamp'])


    def get_state(self):
        if self.curr_idx == len(self.arrivals) - 1:
            return None

        return [self.cpu, self.disk, self.memory,
                self.f_cpu, self.f_disk, self.f_memory]


    def take_action(self, a):
        # a={1,2,3} local deployment, federate, reject
        # returns the reward and next state
        # in case it is the last state, it returns None as state

        # t = t + 1
        prev_time = self.arrivals.time.iloc[self.curr_idx]
        self.curr_idx += 1
        curr_time = self.arrivals.time.iloc[self.curr_idx]

        # calculate the reward from [t, t+1]
        reward = self.__calc_reward(prev_time, curr_time)
        # services leave
        self.__free_resources(curr_time)

        
        # Assign the resources based on the action
        asked_cpu = self.arrivals.iloc[curr_idx]['cpu']
        asked_memory = self.arrivals.iloc[curr_idx]['memory']
        asked_disk = self.arrivals.iloc[curr_idx]['disk']
        if action == A_REJECT:
            pass
        elif action == A_LOCAL:
            if self.cpu < asked_cpu or self.memory < asked_memory or\
                    self.disk < asked_disk:
                reward -= self.arrivals.iloc[curr_idx]['reward']
            else:
                self.cpu -= asked_cpu
                self.memory -= asked_memory
                self.disk -= asked_disk
        elif action == A_FEDERATE:
            if self.f_cpu < asked_cpu or self.f_memory < asked_memory or\
                    self.f_disk < asked_disk:
                reward -= self.arrivals.iloc[curr_idx]['reward']
            else:
                self.f_cpu -= asked_cpu
                self.f_memory -= asked_memory
                self.f_disk -= asked_disk


        return reward, self.get_state() # it'll handle the episode END


    def __calc_arrival_reward(self, prev_time, curr_time, arrival_idx,
                              federated):
        # Compute the reward
        arrival = self.arrivals.iloc[arrival_idx]
        t_end = arrival['lifetime']*24*60*60 + arrival['Timestamp']
        until = min(t_end, curr_time)
        reward = self.arrivals.iloc[arrival_idx]['reward'] *\
                 (until - prev_time) / (60*60) 

        if not federated:
            return reward

        #######################################################################
        # From here down we substract the spot price for the federated arrival
        #######################################################################
        
        # Get the spot_prices of the arrival instance and OS
        spot_history = self.spot_prices[self.spot_prices['InstanceType'] ==\
                                        arrival['instance'] &\
                                    self.spot_prices['ProductDescription'] ==\
                                    arrival['os']]

        # Find first spot price <= prev_time
        before_prev = spot_history[spot_history['Timestamp'] <=\
                pd.Timestamp(prev_time, unit='s')]
        before_prev.sort_values(by=['Timestamp'], ascending=False, inplace=True)
        
        # Derive spot prices in [prev_time, curr_time]
        spot_history = spot_history[spot_history['Timestamp'] >=\
                before_prev.iloc[0]['Timestamp'] &\
                spot_history['Timestamp'] <= pd.Timestamp(until, unit='s')]
        spot_history.sort_values(by=['Timestamp'], ascending=False,
                                  inplace=True)
        spot_history = spot_history[['Timestamp', 'SpotPrice']].values
        spot_history[0] = [spot_history[0][0], until]

        # Now compute the pricing cost
        # end=[Timestamp,spot_price]
        for end, begin in zip(spot_history[:-1], spot_history[1:]):
            delta = (end[0].timestamp() - begin[0].timestamp()) / (60*60) # h
            reward -= delta * begin[1]

        return reward


    def __calc_reward(self, prev_time, curr_time):
        # compute the reward in between [prev_time,curr_time]
        # for local and federated instances
        reward = 0

        for local_idx in self.in_local:
            reward += self.__calc_arrival_reward(prev_time, curr_time,
                    local_idx, federated=False)
        for fed_idx in self.in_fed:
            reward += self.__calc_arrival_reward(prev_time, curr_time,
                    fed_idx, federated=True)

        return reward


    def __free_resources(self, curr_time):
        remove_local, remove_fed  = [], []

        # Check local arrivals that have expired
        for local_idx in self.in_local:
            expires = self.arrivals.iloc[local_idx]['time'] +\
                self.arrivals.iloc[local_idx]['lifetime']*24*60*60
            if expires <= curr_time:
                remove_local.append(self.in_local.index(local_idx))

        # Check federated arrivals that have expired
        for fed_idx in self.in_fed:
            expires = self.arrivals.iloc[fed_idx]['time'] +\
                self.arrivals.iloc[fed_idx]['lifetime']*24*60*60
            if expires <= curr_time:
                remove_fed.append(self.in_fed.index(fed_idx))

        # Remove the arrivals from the lists, and free resources
        for rem_loc_idx in remove_local:
            del self.in_local[rem_loc_idx]
            self.cpu    += self.arrivals.iloc[rem_loc_idx]['cpu']
            self.disk   += self.arrivals.iloc[rem_loc_idx]['disk']
            self.memory += self.arrivals.iloc[rem_loc_idx]['memory']
        for rem_fed_idx in remove_fed:
            del self.in_fed[rem_fed_idx]
            self.f_cpu    += self.arrivals.iloc[rem_fed_idx]['cpu']
            self.f_disk   += self.arrivals.iloc[rem_fed_idx]['disk']
            self.f_memory += self.arrivals.iloc[rem_fed_idx]['memory']


    def reset(self):
        self.__free_resources(curr_time=FUTURE)
        self.curr_idx = 0







class ReplayMemory():
    self.__init__(self, N):
        self.N = N
        experience = [] # tuples [st , at, rt , st+1]

    def add_experience(self, state, action, reward, next_state):
        experience += [state, action, reward, next_state]
        if len(experience) > self.N:
            experience = experience[1:]

    def sample(self, num_experiences):
        samples = experience
        while len(samples) != num_experiences:
            del samples[random.randint(0, len(samples)-1)]
        return samples

    def __len__(self):
        return len(self.experience)


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
        keras.layers.Flatten(input_shape=(k,6)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(3)
    ])

    print('Just have created the NN, check below the TF summary')
    model.summary()

    return model



def atari_loss(model, next_phi, phi, reward, action, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    # loss function of Algorithm 1 in
    # "Playing Atari with Deep Reinforcement Learning"
    #
    # next_phi: phi_j+1
    # phi: phi_j
    # reward: r_j
    # action: action_j
    y_ = model(x, training=training)

    rewards = model(next_phi) # the NN yields accept,reject,federate rewards
    y = reward + gamma * max(rewards)

    return tf.math.square(y, model(phi)[action])



def train_q_network(model, k, epsilon, gamma, M, batch_size, env):
    # Implement the training specified in Algorithm 1 of
    # "Playing Atari with Deep Reinforcement Learning"
    #
    # It uses an epsilon-greedy behaviour-policy, and
    # gamma as discounted reward factor. The process uses
    # the reported environment and repeats each episode M
    # times.
    #
    # batch_size: batch size to compute gradient using replay
    #             memory

    # TODO: it is necessary to have an environment slightly
    #       different than the one in environment.py

    # Note: you have to define a custom training loop and define as
    #       loss function the one in Algorithm 1

    phi = []
    curr_state, next_state = env.get_state(), env.get_state()
    D = ReplayMemory()


    for episode in range(M):
        phi = [curr_state for _ in range(k)]

        while next_state not None:
            # epsilon-greedy action selection
            action = 0
            if random.random() < epsilon:
                action = ACTIONS[random.randint(0, len(ACTIONS))]
            Q = model(phi)
            action = Q.index(max(Q))

            # execute in the environment
            reward, next_state = env.take_action(action)
            next_phi = phi[1:] + [next_state]

            D.add_experience(curr_state, action, reward, next_state)
            # TODO: from here on you have to D.sample() and perform
            #       atari_loss over all samples to do gradient descend



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
    




