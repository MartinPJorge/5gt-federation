import argparse
import json
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.units as munits





if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Plot EC2 spot prices')
    parser.add_argument('prices_csv', type=str,
                        help='path to CSV with EC2 spot prices')
    parser.add_argument('instance_types', type=str,
                        help='|-separated list of instances: ' +\
                            't3a.nano|t3a.small|...\n' +\
                            'or * wildcard to plot all')
    parser.add_argument('out', type=str, default='/tmp/ec2-prices.eps',
                        help='Path of the file where the graph is plotted')
    args = parser.parse_args()


    # Load data
    prices_df = pd.read_csv(args.prices_csv)
    prices_df['Timestamp'] = pd.to_datetime(prices_df['Timestamp'])
    # Parse the instances to be plotted
    instances = list(prices_df['InstanceType'].unique())\
            if args.instance_types == '*' else args.instance_types.split('|')

    # Filter the asked instances
    prices_df = prices_df[prices_df['InstanceType'].isin(instances)]
    
    fig, ax = plt.subplots()
    for key, grp in prices_df.groupby(['InstanceType']):
        #grp.set_index('Timestamp')
        grp.sort_values(by='Timestamp', ascending=True)
        ax = grp.plot(ax=ax, kind='line', x='Timestamp', y='SpotPrice',
                      label=key, rot=0)
    # plot_df = prices_df.pivot(index='Timestamp', columns='InstanceType',
    #                           values='SpotPrice')
    plt.legend(loc='best')
    plt.ylabel('Spot price ($)')
    plt.grid(axis='y', color='gray', alpha=0.2)
    plt.tight_layout()
    plt.xlabel(None)
    

    date_form = DateFormatter("%d-%b")
    ax.xaxis.set_major_formatter(date_form)
    # Ensure a major tick for each week using (interval=1) 
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))


    plt.savefig(args.out)


