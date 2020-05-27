import argparse
import json
import pandas as pd


PRICES='aws-ec2-prices-march-to-may.json'
OUT='/tmp/aws-ec2-prices-march-to-may.json'
ZONE=''


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Transform EC2 JSON ' +\
            'prices to CSV')
    parser.add_argument('in_json', type=str,
                        help='input JSON file')
    parser.add_argument('zone', type=str,
                        help='eu-west-3b|us-east-1|...')
    parser.add_argument('instance_types', type=str,
                        help='t3a.nano|t3a.small|...')
    parser.add_argument('out_csv', type=str, default='/tmp/ec2-prices.csv',
                        help='path to store the csv dataframe')
    args = parser.parse_args()



    with open(args.in_json, 'r') as f:
        aws_prices = json.load(f)

    # Parse zones and instance types
    zones = args.zone.split('|')
    instace_types = args.instance_types.split('|')


    data = {
        'AvailabilityZone': [],
        'InstanceType': [],
        'ProductDescription': [],
        'SpotPrice': [],
        'Timestamp': []
    }
    for spot_price in aws_prices['SpotPriceHistory']:
        if spot_price['AvailabilityZone'] in zones and\
                spot_price['InstanceType'] in instace_types:
            data['AvailabilityZone'].append(spot_price['AvailabilityZone'])
            data['InstanceType'].append(spot_price['InstanceType'])
            data['ProductDescription'].append(spot_price['ProductDescription'])
            data['SpotPrice'].append(spot_price['SpotPrice'])
            data['Timestamp'].append(spot_price['Timestamp'])

    df = pd.DataFrame(data)
    df.to_csv(args.out_csv, index=False)



