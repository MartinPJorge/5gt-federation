# based on @sdksb post
# https://medium.com/cloud-uprising/the-data-science-of-aws-spot-pricing-8bed655caed2

import sys
import boto as boto
import boto.ec2 as ec2
import datetime, time
import pandas as pd
import matplotlib.pyplot as plt

##########
# Inputs #
##########
instance_types  = ['c3.xlarge', 'c3.2xlarge', 'c3.4xlarge', 'c3.8xlarge']
# https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Concepts.RegionsAndAvailabilityZones.html
region = 'us-east-1' # eu-west-3 (Paris)
number_of_days = 90
end = os.popen('date -u "+%Y-%m-%dT%H:%M:%S"').read().split('\n')[0]
end = pd.to_datetime(end)
start = pd.to_datetime(end.timestamp() - number_of_days*24*60*60, unit='s')
start = start[0]
print "will process from " + start + " to " + end

# TODO: continue the script adaptation from here on
# - receive as argument number_of_days


pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier

ec2 = boto.ec2.connect_to_region(region)



#
# access the ec2 price history api and download the data for the 
# instance types of interest
#
l = []
for instance in instance_types:
    sys.stdout.write("*** processing " + instance + " ***\n")
    sys.stdout.flush()
    prices = ec2.get_spot_price_history(start_time=start, end_time=end, instance_type=instance)
    for price in prices:
        d = {'InstanceType': price.instance_type, 
             'AvailabilityZone': price.availability_zone, 
             'SpotPrice': price.price, 
             'Timestamp': price.timestamp}
        l.append(d)
    next = prices.next_token
    while (next != ''):
        sys.stdout.write(".")
        sys.stdout.flush()
        prices = ec2.get_spot_price_history(start_time=start, end_time=end, instance_type=instance, next_token=next)
        for price in prices:
            d = {'InstanceType': price.instance_type, 
                 'AvailabilityZone': price.availability_zone, 
                 'SpotPrice': price.price, 
                 'Timestamp': price.timestamp}
            l.append(d)
        next = prices.next_token
    sys.stdout.write("\n")

df = pd.DataFrame(l)
df = df.set_index(pd.to_datetime(df['Timestamp']))


