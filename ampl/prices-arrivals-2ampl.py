#!/usr/bin/python3

from amplpy import AMPL, DataFrame
import argparse
import json
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Given arrivals+prices CSVs, ' + \
                                                 'it creates an AMPL .dat file')

    parser.add_argument('model', metavar='model', type=str,
                        help='Path to the AMPL model')
    parser.add_argument('arrivals', metavar='arrivals', type=str,
                        help='Path to the arrivals CSV file')
    parser.add_argument('prices', metavar='prices', type=str,
                        help='Path to the spot prices CSV file')
    parser.add_argument('margin', metavar='margin', type=float,
                        help='specify local margin of arrivals, e.g. 0.2')
    parser.add_argument('instances', metavar='instances', type=str,
                        help='list of instances to consider '+\
                                '"t3a.small|c5d.2xlarge|c5d.4xlarge"')
    parser.add_argument('out', metavar='out', type=str,
                        help='Path to the output where .dat is created')
    args = parser.parse_args()


    # Create the AMPL object
    ampl = AMPL()
    ampl.read(args.model)

    # Set the marginal benefit
    margin = ampl.getParameter('margin')
    margin.set(args.margin)


    # Open the spot prices
    spot_prices_df = pd.read_csv(args.prices)
    spot_prices_df = spot_prices_df[spot_prices_df['InstanceType'].isin(
                    args.instances.split('|'))]
    spot_prices_df['Timestamp'] = pd.to_datetime(spot_prices_df['Timestamp'])
    spot_prices_df.sort_values(by='Timestamp', inplace=True, ascending=True)

    # Open the arrivals of specified instances
    arrivals_df = pd.read_csv(args.arrivals)
    arrivals_df = arrivals_df[arrivals_df['instance'].isin(
                    args.instances.split('|'))]
    arrivals_df.sort_values(by='time', inplace=True, ascending=True)

    # Define the parameters
    federate_fee = {}
    instances, instance_arrival, instance_departure = [], [], []
    available = {}
    asked_cpu, asked_mem, asked_disk = [], [], []
    frees_cpu, frees_mem, frees_disk, frees_arrival  = [], [], [], []

    # Set the federate fee - note we consider arrivals' timestamps
    for idx, row in arrivals_df.iterrows():
        t, instance = row['time'], row['instance']
        historic = prices_df[(prices_df['InstanceType']==instance) &
                (prices_df['time'] <= pd.Timestamp(t, unit='s', tz='utc'))]

        # Double check if there are historic spot price values
        if len(historic) == 0:
            federate_fee[(t, instance)] = prices_df[
                    prices_df['InstanceType'] == instance]['SpotPrice'].iloc[0]
        else:
            federate_fee[(t, instance)] = historic['SpotPrice'].iloc[0]

        timestamps += [t]
        instances += [f'instance-{t}']
        instance_arrival += [t]
        asked_cpu += [row['cpu']]
        asked_memory += [row['memory']]
        asked_disk += [row['disk']]


    # TODO - introduce the null instances for each instance departure
    #        that way the model holds
    # Fill availability
    # TODO
    # Fill frees_{cpu,mem,disk,arrival}
    # note, aggregate freed resources 
    # fill instance_departure - round it to the next arrival
    #instance_departure += [t + 60*60 * row['lifetime']] # lifetime->days

    with open(args.arrivals, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        for row in reader:
            times += [float(row[2])]
            asked_cpu += [float(row[3])]
            asked_mem += [float(row[4])]
            asked_disk += [float(row[5])]
            asked_lifes += [float(row[6])]
            profit_federate += [1]
            profit_local += [float(row[7])]
            profit_reject += [-float(row[7])]

            # Include the leaving
            leaves_time += [times[-1] + asked_lifes[-1]]
            leaves_arrival += [times[-1]]
            leaves_cpu += [asked_cpu[-1]]
            leaves_mem += [asked_mem[-1]]
            leaves_disk += [asked_disk[-1]]


    # create the events dictionary
    events = {}
    for i in range(len(times)):
        events[times[i]] = {
            'profit_federate': profit_federate[i],
            'profit_local': profit_local[i],
            'profit_reject': profit_reject[i],
            'asked_cpu': asked_cpu[i],
            'asked_mem': asked_mem[i],
            'asked_disk': asked_disk[i],
            'frees_mem': 0,
            'frees_cpu': 0,
            'frees_disk': 0,
            'frees_arrival': leaves_arrival[0]
        }
    for i in range(len(leaves_time)):
        events[leaves_time[i]] = {
            'profit_federate': 0,
            'profit_local': 0,
            'profit_reject': 0,
            'asked_cpu': 0,
            'asked_mem': 0,
            'asked_disk': 0,
            'frees_mem': leaves_mem[i],
            'frees_cpu': leaves_cpu[i],
            'frees_disk': leaves_disk[i],
            'frees_arrival': leaves_arrival[i]
        }

    # Set the ordered timestamps
    timestamps = times + leaves_time
    timestamps.sort()
    ampl.set['timestamps'] = timestamps

    # Set profits
    df = DataFrame(('timestamps'), 'profit_federate')
    df.setValues({t: events[t]['profit_federate'] for t in events.keys()})
    ampl.setData(df)
    df = DataFrame(('timestamps'), 'profit_local')
    df.setValues({t: events[t]['profit_local'] for t in events.keys()})
    ampl.setData(df)
    df = DataFrame(('timestamps'), 'profit_reject')
    df.setValues({t: events[t]['profit_reject'] for t in events.keys()})
    ampl.setData(df)

    # Set asked resources
    df = DataFrame(('timestamps'), 'asked_cpu')
    df.setValues({t: events[t]['asked_cpu'] for t in events.keys()})
    ampl.setData(df)
    df = DataFrame(('timestamps'), 'asked_mem')
    df.setValues({t: events[t]['asked_mem'] for t in events.keys()})
    ampl.setData(df)
    df = DataFrame(('timestamps'), 'asked_disk')
    df.setValues({t: events[t]['asked_disk'] for t in events.keys()})
    ampl.setData(df)

    # Set leavings
    df = DataFrame(('timestamps'), 'frees_cpu')
    df.setValues({t: events[t]['frees_cpu'] for t in events.keys()})
    ampl.setData(df)
    df = DataFrame(('timestamps'), 'frees_mem')
    df.setValues({t: events[t]['frees_mem'] for t in events.keys()})
    ampl.setData(df)
    df = DataFrame(('timestamps'), 'frees_disk')
    df.setValues({t: events[t]['frees_disk'] for t in events.keys()})
    ampl.setData(df)
    df = DataFrame(('timestamps'), 'frees_arrival')
    df.setValues({t: events[t]['frees_arrival'] for t in events.keys()})
    ampl.setData(df)
    ampl.exportData(datfile=args.out)

