#!/usr/bin/python3

from amplpy import AMPL, DataFrame
import argparse
import sys
import json
import pandas as pd
import time


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
    spot_ = pd.read_csv(args.prices)
    spot_ = spot_[spot_['InstanceType'].isin(args.instances.split('|'))]
    spot_['Timestamp'] = pd.to_datetime(spot_['Timestamp'])
    spot_.sort_values(by='Timestamp', inplace=True, ascending=True)

    # Open the arrivals of specified instances
    arrivals_df = pd.read_csv(args.arrivals)
    arrivals_df = arrivals_df[arrivals_df['instance'].isin(
                    args.instances.split('|'))]
    arrivals_df.sort_values(by='time', inplace=True, ascending=True)

    # Define the parameters
    instance_types = args.instances.split('|') + ['null']
    federate_fee = {}
    itype = {}
    instances, instance_arrival, instance_departure = [], {}, {}
    instantiations = {it : [] for it in instance_types}
    available = {}
    timestamps = []
    asked_cpu, asked_mem, asked_disk = [], [], []
    frees_cpu, frees_mem, frees_disk, frees_arrival  = [], [], [], []

    # Set the federate fee - note we consider arrivals' timestamps
    print('Filling arrivals resources parameters')
    for idx, row in arrivals_df.iterrows():
        t, instance = row['time'], row['instance']

        # Arrival event
        timestamps += [t]
        instances += [f'{instance}-{t}']
        itype[instances[-1]] = instance
        instantiations[instance] += [instances[-1]]
        instance_arrival[instances[-1]] = t
        instance_departure[instances[-1]] = t + 24*60*60 * row['lifetime']
        asked_cpu += [row['cpu']]
        asked_mem += [row['memory']]
        asked_disk += [row['disk']]
        frees_cpu += [0]
        frees_mem += [0]
        frees_disk += [0]
        frees_arrival += [timestamps[0]] # whatever here - freed resources is 0

        # Departure event - null instance associated
        timestamps += [t + 24*60*60 * row['lifetime']]
        instances += [f'null-{t}']
        itype[instances[-1]] = 'null'
        instantiations['null'] += [instances[-1]]
        instance_arrival[instances[-1]] = timestamps[-1]
        instance_departure[instances[-1]] = sys.maxsize # leaves at infinity
        asked_cpu += [0]
        asked_mem += [0]
        asked_disk += [0]
        frees_cpu += [row['cpu']]
        frees_mem += [row['memory']]
        frees_disk += [row['disk']]
        frees_arrival += [t]

    # Create arrivals-departure dataframe
    arrivals_departures = pd.DataFrame({
        'instance': instances,
        'time': timestamps,
        'asked_cpu': asked_cpu,
        'asked_mem': asked_mem,
        'asked_disk': asked_disk,
        'frees_cpu': frees_cpu,
        'frees_mem': frees_mem,
        'frees_disk': frees_disk,
        'frees_arrival': frees_arrival
    })
    arrivals_departures.sort_values(by='time', inplace=True, ascending=True)


    # Derive the spot price at t for every instance
    print('Filling federation fees parameters')
    j = 0
    for t in timestamps:
        h = spot_[spot_['Timestamp'] <= pd.Timestamp(t, unit='s', tz='utc')]
        print(f'{j}/{len(timestamps)}', end='\r')
        j += 1

        for i_type in instance_types:
            if len(h) > 0:
                hi = h[h['InstanceType'] == i_type]
            hi = None if len(h) == 0 or len(hi) == 0 else hi

            if type(hi) == type(None):
                federate_fee[(i_type, t)] = spot_[spot_['InstanceType'] ==\
                        i_type]['SpotPrice'].iloc[0]\
                            if i_type != 'null' else 0
            else:
                federate_fee[(i_type, t)] = hi['SpotPrice'].iloc[-1]\
                        if i_type != 'null' else 0


    # Fill each instance availability
    ## print('Filling time availability')
    ## available = {
    ##     (i, t): 0
    ##     for i in instances
    ##     for t in timestamps
    ## }
    ## avk = available.keys()
    ## j = 0
    ## for idx, row in arrivals_departures.iterrows():
    ##     print(f'{j}/{len(arrivals_departures)}', end='\r')
    ##     j += 1
    ##     departure = instance_departure[row['instance']]
    ##     for t in timestamps:
    ##         if t >= row['time'] and t <= departure:
    ##             available[(i,t)] = 1


    # Specify the precision to avoid timestamps rounding
    ampl.setOption('display_eps', 0);
    ampl.setOption('display_precision', 0);

    # Fill the sets
    print('Dumping instance_types using amplpy')
    start = time.time()
    ampl.set['instance_types'] = instance_types
    print(f'It took {time.time() - start} seconds')
    print('Dumping timestamps using amplpy')
    start = time.time()
    ampl.set['timestamps'] = list(arrivals_departures['time'])
    print(f'It took {time.time() - start} seconds')
    print('Dumping instances using amplpy')
    start = time.time()
    ampl.set['instances'] = instances
    print(f'It took {time.time() - start} seconds')


    # Fill the parameters
    print('Dumping itypes using amplpy')
    start = time.time()
    df = DataFrame(('instances'), 'itype')
    df.setValues({i: itype[i] for i in instances})
    ampl.setData(df)
    print(f'It took {time.time() - start} seconds')

    print('Dumping federate_fee using amplpy')
    start = time.time()
    df = DataFrame(('instances', 'timestamps'), 'federate_fee')
    df.setValues({(it,t): federate_fee[(it,t)]
                          for (it,t) in federate_fee.keys()})
    ampl.setData(df)
    print(f'It took {time.time() - start} seconds')

    print('Dumping instance_arrival using amplpy')
    start = time.time()
    df = DataFrame(('instances'), 'instance_arrival')
    df.setValues({i: instance_arrival[i] for i in instance_arrival.keys()})
    ampl.setData(df)
    print(f'It took {time.time() - start} seconds')

    print('Dumping instance_departure using amplpy')
    start = time.time()
    df = DataFrame(('instances'), 'instance_departure')
    df.setValues({i: instance_departure[i] for i in instance_departure.keys()})
    ampl.setData(df)
    print(f'It took {time.time() - start} seconds')

    ## print('Dumping available using amplpy')
    ## start = time.time()
    ## df = DataFrame(('instances', 'timestamps'), 'available')
    ## df.setValues({(i,t): available[(i,t)] for (i,t) in available.keys()})
    ## ampl.setData(df)
    ## print(f'It took {time.time() - start} seconds')

    print('Dumping asked_cpu using amplpy')
    start = time.time()
    df = DataFrame(('timestamps'), 'asked_cpu')
    df.setValues({r['time']: r['asked_cpu']
                  for _, r in arrivals_departures.iterrows()})
    ampl.setData(df)
    print(f'It took {time.time() - start} seconds')

    print('Dumping asked_mem using amplpy')
    start = time.time()
    df = DataFrame(('timestamps'), 'asked_mem')
    df.setValues({r['time']: r['asked_mem']
                  for _, r in arrivals_departures.iterrows()})
    ampl.setData(df)
    print(f'It took {time.time() - start} seconds')

    print('Dumping asked_disk using amplpy')
    start = time.time()
    df = DataFrame(('timestamps'), 'asked_disk')
    df.setValues({r['time']: r['asked_disk']
                  for _, r in arrivals_departures.iterrows()})
    ampl.setData(df)
    print(f'It took {time.time() - start} seconds')

    print('Dumping frees_cpu sing amplpy')
    start = time.time()
    df = DataFrame(('timestamps'), 'frees_cpu')
    df.setValues({r['time']: r['frees_cpu']
                  for _, r in arrivals_departures.iterrows()})
    ampl.setData(df)
    print(f'It took {time.time() - start} seconds')

    print('Dumping frees_mem sing amplpy')
    start = time.time()
    df = DataFrame(('timestamps'), 'frees_mem')
    df.setValues({r['time']: r['frees_mem']
                  for _, r in arrivals_departures.iterrows()})
    ampl.setData(df)
    print(f'It took {time.time() - start} seconds')

    print('Dumping frees_disk sing amplpy')
    start = time.time()
    df = DataFrame(('timestamps'), 'frees_disk')
    df.setValues({r['time']: r['frees_disk']
                  for _, r in arrivals_departures.iterrows()})
    ampl.setData(df)
    print(f'It took {time.time() - start} seconds')

    print('Dumping frees_arrival sing amplpy')
    start = time.time()
    df = DataFrame(('timestamps'), 'frees_arrival')
    df.setValues({r['time']: r['frees_arrival']
                  for _, r in arrivals_departures.iterrows()})
    ampl.setData(df)
    print(f'It took {time.time() - start} seconds')
    
    # Dump to .dat file
    print(f'Storing data to {args.out}')
    ampl.exportData(datfile=args.out)

