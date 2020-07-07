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
    parser.add_argument('lf_std', type=float,
                        help='lifetime in lifetime+-lf_std/100')
    parser.add_argument('fee_margin', type=float,
                        help='0.n fee price above the spot price percentage')
    parser.add_argument('out', type=str, default='/tmp/ec2-arrivals.csv',
                        help='Path of the CSV where EC2 arrivals are stored')
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

    # Retain the original dataframe to latter query spot prices
    orig_prices_df = pd.DataFrame(prices_df)
    orig_prices_df['Timestamp'] = pd.to_datetime(orig_prices_df['Timestamp'])
    orig_prices_df.sort_values(by='Timestamp', ascending=True, inplace=True)

    
    #############################
    # OBTAIN AVG. PRICE PER DAY #
    #############################
    prices_df['Timestamp'] = pd.to_datetime(prices_df['Timestamp'])\
                               .dt.floor('d')
    avg_prices_df = prices_df.groupby(['Timestamp', 'InstanceType',
        'ProductDescription'])['SpotPrice'].mean().reset_index()
    avg_prices_df = avg_prices_df.rename(columns={"SpotPrice": "AvgSpotPrice"})


    ###################################
    # GENERATE THE INSTANCES ARRIVALS # 
    ###################################
    arrivals = {
        'time': [],
        'instance': [],
        'spotprice': [],
        'lifetime': [], # expressed in days
        'os': [],
        'cpu': [],
        'memory': [],
        'disk': [],
        'reward': []
    }

    # Get maximum spot prices
    max_spot_price = {
        instance: avg_prices_df[avg_prices_df['InstanceType'] ==\
                       instance]['AvgSpotPrice'].max()
        for instance in avg_prices_df['InstanceType'].unique()
    }

    print('Generating the time arrivals')
    for idx, row in avg_prices_df.iterrows():
        instance_info = instances_info[row['InstanceType']]
        print('Generating arrivals for ' + row['InstanceType'])
        arrival_rate = fArrival(p=row['AvgSpotPrice'] /\
                                    max_spot_price[row['InstanceType']],
                                k=instance_info['fk'], a=instance_info['fa'],
                                b=instance_info['fb'])
        instance_prices = orig_prices_df[orig_prices_df['InstanceType'] ==\
                                            row['InstanceType']]
        i, epoch = 0, row['Timestamp'].timestamp()
        while i < int(arrival_rate): # e.g. i < 3 instances/that-day
            epoch += rexp(scale=1/arrival_rate)*24*60*60
            lifetime = truncnorm.rvs(size=1,
                    a=instance_info['lifetime']*(1-args.lf_std/100),
                    b=instance_info['lifetime']*(1+args.lf_std/100))[0]
            i += 1

            # Find associated spot price
            past_prices = instance_prices[instance_prices['Timestamp'] <=\
                                pd.Timestamp(epoch, unit='s', tz='UTC')]
            if len(past_prices) == 0: # avg spot price starts at 00:00
                                      # whilst first real price might be at
                                      # 23:00
                past_prices = instance_prices[instance_prices['Timestamp'] <=\
                             pd.Timestamp(epoch+24*60*60, unit='s', tz='UTC')]
            spot_price = past_prices['SpotPrice'].iloc[-1]
            print('\t', spot_price)

            arrivals['time'].append(epoch)
            arrivals['instance'].append(row['InstanceType'])
            arrivals['spotprice'].append(spot_price)
            arrivals['os'].append(row['ProductDescription'])
            arrivals['lifetime'].append(lifetime)
            arrivals['cpu'].append(instance_info['cpu'])
            arrivals['memory'].append(instance_info['memory'])
            arrivals['disk'].append(instance_info['disk'])
            arrivals['reward'].append(spot_price * (1+args.fee_margin))


    ############################################################
    # Sort the arrivals by their arrival time and store in CSV #
    ############################################################
    arrivals_df = pd.DataFrame(data=arrivals)
    arrivals_df.sort_values(by='time', inplace=True)
    arrivals_df.to_csv(args.out, index=False)



