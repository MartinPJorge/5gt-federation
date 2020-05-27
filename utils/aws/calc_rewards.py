import argparse
import json
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.units as munits
import sys
from scipy.stats import truncnorm
from numpy.random import exponential as rexp

hows = ['x2avgPrice', 'x2maxPrice']


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate EC2 arrivals')
    parser.add_argument('instaces_info', type=str,
                        help='Path to JSON with instance information')
    parser.add_argument('prices_csv', type=str,
                        help='Path to CSV with ec2 pricing history')
    parser.add_argument('how', type=str, help='how to calculate each ' +\
                        f'instance reward: {hows}')
    parser.add_argument('out', type=str,
            default='/tmp/instances-info-and-rewards.json',
                        help='Path of the CSV where EC2 arrivals are stored')
    args = parser.parse_args()


    if args.how not in hows:
        print(f'error: how={args.how} is not listed in {hows}')
        sys.exit(1)


    # Read the JSON with instance characteristics
    instances_info = None
    with open(args.instaces_info, 'r') as f:
        instances_info = json.load(f)

    # Load prices data
    prices_df = pd.read_csv(args.prices_csv)
    prices_df['Timestamp'] = pd.to_datetime(prices_df['Timestamp'])

    # Compute the rewards using the specified method
    rewards = {}
    for instance in prices_df['InstanceType'].unique():
        if args.how == 'x2maxPrice':
            max_price = prices_df[prices_df['InstanceType'] ==\
                                  instance]['SpotPrice'].max()
            rewards[instance] = 2 * max_price
        else:
            mean_price = prices_df[prices_df['InstanceType'] ==\
                                  instance]['SpotPrice'].mean()
            rewards[instance] = 2 * mean_price
        print(f'reward for {instance} = {rewards[instance]}')


    # Save the JSON with the specified reward
    for instance in rewards.keys():
        instances_info[instance]['reward'] = rewards[instance]
    with open(args.out, 'w') as f:
        json.dump(instances_info, fp=f, indent=4)



    
