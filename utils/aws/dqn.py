import argparse
import json
import numpy as np
import datetime
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import random
import math



# print("TensorFlow version: {}".format(tf.__version__))
# print("Eager execution: {}".format(tf.executing_eagerly()))



# ACTIONs macros
A_LOCAL = 0
A_FEDERATE = 1
A_REJECT = 2
ACTIONS = [A_LOCAL, A_FEDERATE, A_REJECT]

# FUTURE
FUTURE = pd.Timestamp('2200-01-01 00:00:00+00:00')



class AWS_env():
    def __init__(self, cpu, memory, disk, f_cpu, f_memory, f_disk,
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


    def take_action(self, action):
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
        asked_cpu = self.arrivals.iloc[self.curr_idx]['cpu']
        asked_memory = self.arrivals.iloc[self.curr_idx]['memory']
        asked_disk = self.arrivals.iloc[self.curr_idx]['disk']
        if action == A_REJECT:
            pass
        elif action == A_LOCAL:
            if self.cpu < asked_cpu or self.memory < asked_memory or\
                    self.disk < asked_disk:
                reward -= self.arrivals.iloc[self.curr_idx]['reward']
            else:
                self.cpu -= asked_cpu
                self.memory -= asked_memory
                self.disk -= asked_disk
        elif action == A_FEDERATE:
            if self.f_cpu < asked_cpu or self.f_memory < asked_memory or\
                    self.f_disk < asked_disk:
                reward -= self.arrivals.iloc[self.curr_idx]['reward']
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
    def __init__(self, N):
        self.N = N
        self.experience = [] # tuples [phi_t , at, rt , phi_t+1]

    def add_experience(self, phi, action, reward, next_phi):
        self.experience += [[phi, action, reward, next_phi]]
        if len(self.experience) > self.N:
            self.experience = self.experience[1:]

    def sample(self, num_experiences):
        samples = list(self.experience)
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
    # (state,action) --->  NN  ---> Q(state, action) 
    #
    # return: the tf.model for the NN

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(k*6 + 1,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1)
    ])

    print('Just have created the NN, check below the TF summary')
    model.summary()

    return model



def atari_loss(model, transitions, training, k, gamma):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    # loss function of Algorithm 1 in
    # "Playing Atari with Deep Reinforcement Learning"
    #
    # transitions: [phi, action, reward, next_phi]
    #   next_phi: phi_j+1
    #   phi: phi_j
    #   reward: r_j
    #   action: action_j
    #
    # return: loss for the transition (phi , action , reward , next_phi)
    #

    ins = []
    labels = None

    # Get the maximum reward for next_phi
    for phi, action, reward, next_phi in transitions:
        rewards = [model(tf.reshape(tf.concat([next_phi, [a]], 0),
                                    shape=(1,6*k+1)))\
                   for a in ACTIONS]
        max_reward = rewards[0]
        for reward in rewards:
            if reward < max_reward:
                max_reward = reward

        ins.append(phi + [action])

        if labels == None:
            labels = reward + gamma * max_reward
        else:
            labels = tf.concat([labels, reward + gamma * max_reward], 0)

    # reshape the transitions to feed them into the Q-network model
    ins = tf.constant(ins, shape=(len(transitions), k*6+1))
    labels = tf.constant(labels, shape=(len(transitions), 1))
    pred = model(ins)

    return tf.keras.losses.MSE(y_true=labels, y_pred=pred)



def phi_(sequence, k):
    # it flattens the sequence of last k states
    # sequence: list of lists [[cpu,mem,disk,f_cpu,f_mem,f_disk],
    #                          [cpu2,mem2,disk2,f_cpu2,f_mem2,f_disk2], ...]
    flat = []
    for state in sequence[-k:]:
        flat += state
    
    return flat


def cast(phi, action, k):
    # casts the phi, action lists to TF matrix
    # phi: list
    # action: integer
    #
    # return tf tensor of shape (1, k*6+1)
    return tf.reshape(tf.concat([phi, [action]], axis=0), shape=(k*6+1))



def train_q_network(model, k, epsilon, gamma, M, batch_size, N, env):
    # Implement the training specified in Algorithm 1 of
    # "Playing Atari with Deep Reinforcement Learning"
    #
    # It uses an epsilon-greedy behaviour-policy, and
    # gamma as discounted reward factor. The process uses
    # the reported environment and repeats each episode M
    # times.
    #
    # N: replay memory size
    # batch_size: batch size to compute gradient using replay
    #             memory


    sequence = []
    curr_state, next_state = env.get_state(), env.get_state()
    D = ReplayMemory(N=N)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    for episode in range(M):
        sequence = [curr_state for _ in range(k)]
        now_phi = phi_(sequence, k=k)

        while next_state != None:
            # epsilon-greedy action selection
            action = 0
            if random.random() < epsilon:
                action = ACTIONS[random.randint(0, len(ACTIONS) - 1)]
            else:
                in__ = tf.reshape(tf.concat([now_phi, [action]],0), shape=(1, 6*k+1))
                Qs = [model(in__) for action in ACTIONS]
                #action = ACTIONS[tf.math.argmax(Qs)]
                action = ACTIONS[Qs.index(max(Qs))]

            # execute selected action in the environment
            reward, next_state = env.take_action(action)
            sequence += [next_state]
            next_phi = phi_(sequence, k=k)
            D.add_experience(now_phi, action, reward, next_phi)

            # If replay memory is small, don't do gradient
            if len(D) < batch_size:
                now_phi = next_phi
                continue

            # Sample from experience replay, compute loss,
            # and apply the gradient
            with tf.GradientTape() as tape:
                loss = atari_loss(model=model,
                        transitions=D.sample(num_experiences=batch_size),
                        training=True, k=k, gamma=gamma)
                print(f'loss shape={loss.shape}')
                print(f'loss type={type(loss)}')
                print(f'loss={loss}')
                print(f'mod-train-vars={model.trainable_variables}')
                grads = tape.gradient(loss, model.trainable_variables)
            print(f'grads={grads}')
            optimizer.apply_gradients(zip(grads,
                                          model.trainable_variables))

            now_phi = next_phi
            # TODO: one can record the progress




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
    parser.add_argument('arrivals', type=str,
                        help='path to CSV with arrivals dataframe')
    parser.add_argument('domains', type=str,
                        help='path to JSON with local|federated resources')
    parser.add_argument('k', type=int,
                        help='size of history to represent the state')
    parser.add_argument('train', type=bool, help='True|False')
    # Training arguments
    parser.add_argument('--epsilon', type=float,
                        help='epsilon-greedy for the off-policy')
    parser.add_argument('--gamma', type=float,
                        help='discounted factor for the reward')
    parser.add_argument('--M', type=int, help='number of episodes')
    parser.add_argument('--N', type=int, help='replay memory size')
    parser.add_argument('--batch', type=int, help='batch size')
    parser.add_argument('--out_weights', type=str, default='/tmp/weights',
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


    # Load domains JSON with their resources
    with open(args.domains) as fp:
        domain = json.load(fp)
    

    print(f'k={args.k}')


    #############
    # START DQN #
    #############
    env = AWS_env(cpu=domain['local']['cpu'], memory=domain['local']['memory'],
            disk=domain['local']['disk'], f_cpu=domain ['federated']['cpu'],
            f_disk=domain['federated']['disk'],
            f_memory=domain['federated']['memory'],
            arrivals=pd.read_csv(args.arrivals),
            spot_prices=prices_df)
    model = create_q_network(k=args.k)
    train_q_network(model=model, k=args.k, epsilon=args.epsilon,
            gamma=args.gamma, M=args.M, batch_size=args.batch,
            N=args.N, env=env)




