import time
import argparse
import json
import numpy as np
import datetime
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import random
import math
from aws_env import AWS_env
import logging



# print("TensorFlow version: {}".format(tf.__version__))
# print("Eager execution: {}".format(tf.executing_eagerly()))




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
    #                     --> Q(state, local)
    #                    /
    # (states) --->  NN  ---> Q(state, federate) 
    #                    \
    #                     --> Q(state, reject)
    #
    # return: the tf.model for the NN

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(k*11,)),
        #keras.layers.Dense(k*6, activation='relu'),
        keras.layers.Dense(k*11, activation='tanh'),
        #keras.layers.Dense(k*11, activation='sigmoid'),
        #keras.layers.Dense(k*11),
        #keras.layers.Dense(len(AWS_env.ACTIONS), activation='sigmoid')
        #keras.layers.Dense(len(AWS_env.ACTIONS), activation='tanh')
        #keras.layers.Dense(len(AWS_env.ACTIONS), activation='relu')
        keras.layers.Dense(len(AWS_env.ACTIONS))
    ])

    logging.info('Just have created the NN, check below the TF summary')
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

    ins = None
    labels = []

    # Get the maximum reward for next_phi
    for phi, action, reward, next_phi in transitions:
        # TODO: a doubt is wether we must still select max action
        #       i.e., y=[reward + gamma*model(next_phi).max() for _ in\
        #                                                     range(3)]
        # TODO-answer: yes it should be, after doing the action, the
        #              agent will select the maximum benefit by best next
        #              action
        Q_next = model(next_phi)
        Q_max = tf.math.multiply( tf.ones(shape=Q_next.shape),
                                  tf.math.reduce_max(Q_next) )
        y = tf.math.add(reward, tf.math.multiply(gamma, Q_max))
        y = y.numpy()
        phi = tf.cast(phi, dtype=tf.float64)
        ins = tf.concat([ins, phi], 0) if ins != None else phi
        pred = model(phi)

        # Set y=pred for a!=action
        for a in AWS_env.ACTIONS:
            if a != action:
                y[0][a] = pred.numpy()[0][a]
        # y = tf.constant(y) # TODO: check if changing this to constant

        if len(labels):
            labels = tf.concat([labels, y], 0)
        else:
            labels = y

    # reshape the transitions to feed them into the Q-network model
    ins = tf.constant(ins, shape=(len(transitions), k*11))
    #labels = tf.constant(labels, shape=(len(transitions), 1))
    preds = model(ins)

    return tf.keras.losses.MSE(y_true=labels, y_pred=preds)



def phi_(sequence, k):
    # it flattens the sequence of last k states
    # sequence: list of lists [[cpu,mem,disk,f_cpu,f_mem,f_disk],
    #                          [cpu2,mem2,disk2,f_cpu2,f_mem2,f_disk2],
    #                           ...
    #                         ]
    flat = []
    for state in sequence[-k:]:
        flat += state

    return tf.reshape(flat, shape=(1, k*11))


# TODO: deprecated
def cast(phi, action, k):
    # casts the phi, action lists to TF matrix
    # phi: list
    # action: integer
    #
    # return tf tensor of shape (1, k*6)
    return tf.reshape(tf.concat([phi, [action]], axis=0), shape=(k*6+1))



def train_q_network(model, k, epsilon_start, epsilon_end, gamma, alpha, M,
                    batch_size, N, env, out=None):
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
    # out: path where the trained model is stored
    #
    # returns: list of episode rewards


    sequence = []
    D = ReplayMemory(N=N)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=alpha) #alpha=0.01
    optimizer = tf.keras.optimizers.RMSprop()
    episodes_rewards = []

    for episode in range(M):
        env.reset()
        curr_state, next_state = env.get_state(), env.get_state()
        logging.info(f'EPISODE={episode}')
        sequence = [curr_state for _ in range(k)]
        now_phi = phi_(sequence, k=k)
        expisode_reward = 0
        epsilon = (episode+1) / M * (epsilon_end - epsilon_start)\
                  + epsilon_start
	# TODO - hardcoded for TID experiment, decrease until M=50
        epsilon = (episode+1) / 50 * (epsilon_end - epsilon_start)\
                  + epsilon_start
	# TODO - hardcoded to force TID experiment epsilon=0.1 after M=50
        if episode >= 50:
            epsilon = epsilon_end


        t = 0
        while next_state != None:
            start_interval = time.time()
            t = t + 1
            logging.debug(f'\nt={t}\t')
            logging.debug(f'Q-network at t={t}: {model(now_phi)}')
            # epsilon-greedy action selection
            action = 0
            if random.random() < epsilon:
                action = AWS_env.ACTIONS[random.randint(0,
                                         len(AWS_env.ACTIONS) - 1)]
                logging.debug(f'ϵ-greedy: random action={action}')
            else:
                Q = model(now_phi)
                action = AWS_env.ACTIONS[Q[0].numpy().argmax()]
                logging.debug(f'ϵ-greedy: max action={action}')
                #print(f'\tmax all action values={Q}')

            # execute selected action in the environment
            start_action = time.time()
            reward, next_state = env.take_action(action)
            expisode_reward += reward
            logging.debug(f'time action = {time.time() - start_action}')
            if next_state == None:
                break
            logging.debug(f'action={action},reward={reward},next_state={next_state}')
            sequence += [next_state]
            next_phi = phi_(sequence, k=k)
            D.add_experience(now_phi, action, reward, next_phi)

            logging.debug(f'time interval = {time.time() - start_interval}')
            # If replay memory is small, don't do gradient
            if len(D) < batch_size:
                now_phi = next_phi
                continue

            # Sample from experience replay, compute loss,
            # and apply the gradient
            start_action = time.time()
            with tf.GradientTape() as tape:
                loss = atari_loss(model=model,
                        transitions=D.sample(num_experiences=batch_size),
                        training=True, k=k, gamma=gamma)
                logging.debug(f'loss shape={loss.shape}')
                logging.debug(f'loss type={type(loss)}')
                logging.debug(f'loss={loss}')
                logging.debug(f'mod-train-vars={model.trainable_variables}')
                grads = tape.gradient(loss, model.trainable_variables)
            logging.debug(f'grads={grads}')
            optimizer.apply_gradients(zip(grads,
                                          model.trainable_variables))
            logging.debug(f'time gradient descend = {time.time() - start_interval}')

            now_phi = next_phi
            # TODO: one can record the progress
        
        episodes_rewards.append(expisode_reward)

    if out != None:
        model.save(out)


    return episodes_rewards



def test_q_network(model, k, env):
    # model: q_network trained with parameter k
    # k: history length
    # env: environmnet to take actions
    #
    # returns: the SAR, i.e, sequences, actions, rewards

    env.reset()
    rewards, actions = [], []
    state = env.get_state()
    state_sequence = [state for _ in range(k)]

    t = 0
    while state != None:
        logging.info(f't={t} test')
        # Select the maximum action for the Q(·)
        logging.info('state sequence')
        logging.info(state_sequence)
        phi = phi_(state_sequence, k=k)
        logging.info('phi')
        logging.info(phi)
        st_q = time.time()
        Q = model(phi)
        logging.info(f'it takes {time.time() - st_q} seconds to feed forward')
        logging.info(f'Q={Q}')
        actions += [AWS_env.ACTIONS[Q[0].numpy().argmax()]]

        # execute selected action in the environment
        st_a = time.time()
        reward, state = env.take_action(actions[-1])
        logging.info(f'it takes {time.time() - st_a} seconds to act')
        rewards += [reward]
        state_sequence += [state]
        t += 1

    return state_sequence[-(len(actions)+1):-1], actions, rewards


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
    parser.add_argument('--train', action='store_true', default=False)
    # Training arguments
    parser.add_argument('--epsilon_start', type=float,
                        help='epsilon-greedy 1st value for the off-policy')
    parser.add_argument('--epsilon_end', type=float,
                        help='epsilon-greedy last value for the off-policy')
    parser.add_argument('--gamma', type=float,
                        help='discounted factor for the reward')
    parser.add_argument('--alpha', type=float,
                        help='learning rate of optimizer')
    parser.add_argument('--M', type=int, help='number of episodes')
    parser.add_argument('--N', type=int, help='replay memory size')
    parser.add_argument('--batch', type=int, help='batch size')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--in_model', type=str, default='/tmp/model',
                        help='Path where the trained DQN model is stored')
    parser.add_argument('--out_model', type=str, default='/tmp/model',
                        help='Path where the trained DQN model is stored')
    args = parser.parse_args()


    # Set the logger
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)


    # Check arguments
    if args.k < 0:
        logging.info(f'k={k}, but it must be >0')
        sys.exit(1)
    logging.info(f'traIN={args.train}')
    if args.train == True:
        if args.epsilon_start > 1 or args.epsilon_start < 0:
            logging.info(f'epsilon_start={args.epsilon_start}, but it must belong to [0,1]')
            sys.exit(1)
        if args.epsilon_end > 1 or args.epsilon_end < 0:
            logging.info(f'epsilon_end={args.epsilon_end}, but it must belong to [0,1]')
            sys.exit(1)
        if args.gamma > 1 or args.gamma < 0:
            logging.info(f'gamma={args.gamma}, but it must belong to [0,1]')
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

    logging.info(f'k={args.k}')


    # Create the environment
    env = AWS_env(cpu=domain['local']['cpu'], memory=domain['local']['memory'],
            disk=domain['local']['disk'], f_cpu=domain ['federated']['cpu'],
            f_disk=domain['federated']['disk'],
            f_memory=domain['federated']['memory'],
            arrivals=arrivals,
            spot_prices=prices_df)

    # Create the Q-network #### TRAIN
    if args.train == True:
        model = create_q_network(k=args.k)
        epi_rewards = train_q_network(model=model, k=args.k,
                epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
                gamma=args.gamma, alpha=args.alpha, M=args.M,
                batch_size=args.batch, N=args.N, env=env, out=args.out_model)
        logging.info('== EPISODE REWARDS ==')
        logging.info(epi_rewards)
    # Load the Q-network   #### TEST
    else:
        model = tf.keras.models.load_model(args.in_model)
        states, actions, rewards = test_q_network(model, args.k, env)
        logging.info(f'actions={actions}')
        logging.info('State|action|reward')
        for t in range(len(states)):
            logging.info(f'{states[t]}|{actions[t]}|{rewards[t]}')




