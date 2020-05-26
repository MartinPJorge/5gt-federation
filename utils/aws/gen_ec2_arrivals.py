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



# Arrival/departure functions below written by Josep Xavier Salvat
# and adapted by Jorge Martín Pérez


def fArrival(p, k=2, a=2, b=0.5):
    #arrival rate function depending on the price
    #k is the maximum arrival rate
    #a, b are parameters of the arrival rate function
    #p is the price [0, 1]
    return float(k*((1-p**a)**b))


def gDeparture(p, k=2, a=2, b=0.5):
    # departure rate function depending on the price
    # k is the maximum arrival rate
    # a, b are parameters of the arrival rate function
    # p is the price [0, 1]
    return float(k - k*((1-p**a)**b))






if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate EC2 arrivals')
    parser.add_argument('prices_csv', type=str,
                        help='path to CSV with EC2 spot prices')
    parser.add_argument('instaces_info', type=str,
                        help='Path to JSON with instance information')
    parser.add_argument('instance_types', type=str,
                        help='|-separated list of instances: ' +\
                            't3a.nano|t3a.small|...\n' +\
                            'or * wildcard to plot all')
    parser.add_argument('out', type=str, default='/tmp/ec2-arrivals.csv',
                        help='Path of the CSV where EC2 arrivals are stored')
    parser.add_argument('--fk', default=2, help='k value for arrival rate f()')
    parser.add_argument('--fa', default=2, help='a value for arrival rate f()')
    parser.add_argument('--fb', default=0.5, help='b value for arrival rate f()')
    parser.add_argument('--gk', default=2, help='k value for departure rate g()')
    parser.add_argument('--ga', default=2, help='a value for departure rate g()')
    parser.add_argument('--gb', default=0.5, help='b value for departure rate g()')
    args = parser.parse_args()


    # Read the JSON with instance characteristics
    instances_info = None
    with open(args.instaces_info, 'r') as f:
        instances_info = json.load(f)

    # Load data
    prices_df = pd.read_csv(args.prices_csv)
    prices_df['Timestamp'] = pd.to_datetime(prices_df['Timestamp'])
    # Parse the instances to be plotted
    instances = list(prices_df['InstanceType'].unique())\
            if args.instance_types == '*' else args.instance_types.split('|')
    # Filter the asked instances
    prices_df = prices_df[prices_df['InstanceType'].isin(instances)]


    
    #############################
    # OBTAIN AVG. PRICE PER DAY #
    #############################
    prices_df['Timestamp'] = pd.to_datetime(prices_df['Timestamp'])\
                               .dt.floor('d')
    avg_prices_df = prices_df.groupby(['Timestamp',
                                       'InstanceType'])['SpotPrice'].mean()\
                                     .reset_index()

    print(avg_prices_df[avg_prices_df['InstanceType']=='c5.metal'].head())
    # TODO: now avg_prices_df have [Time, instance-type, avg spot price]
    #       next is to generate the arrival rate



