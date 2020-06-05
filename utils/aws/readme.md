# AWS
This directory contains some scripts used to generate service arrivals using
AWS data, the truncated lifetime [1], and the arrival/departure rates of [2].

Image below illustrates the pipeline of how to use the scripts to generate the
arrivals.
The AWS JSON is obtained using a line like:
```bash
aws ec2 describe-spot-price-history  --start-time 2020-03-01 --end-time 2020-05-25
```
then that file is fed to the 1st script of the pipeline.

<img src="./pipeline.png"/>

## Genrate the spot prices CSV
To generate covnert the AWS spot prices data from JSON to CSV execute:
```bash
python3 ec2_prices_to_csv.py\
    aws-ec2-prices-march-to-may.json\   # AWS JSON with spot prices over time
    eu-west-3a\                         # region to filter
    "m5a.2xlarge|m5a.12xlarge|m5a.16xlarge|m5ad.2xlarge|m5ad.16xlarge|m5d.2xlarge|m5d.large|m5d.8xlarge|m5d.8xlarge|m5d.16xlarge|m5d.24xlarge|t2.large|t3.large|t3a.nano|t3a.medium|t3a.small|c5x.large|c5.2xlarge|c5.18xlarge|c5.metal|c5d.xlarge|c5d.4xlarge|c5d.18xlarge"\  # instances to filter
    /tmp/ec2-prices-march-to-may.csv    # CSV file where prices are stored
```

The resulting CSV `/tmp/ec2-march-to-may.csv` contains this information:

| Column   |  Definition   | Example |
|----------|:-------------:|---------|
| **AvailabilityZone**   | AWS zone where pricing applies | eu-west-3a |
| **InstanceType**       | AWS instance type | t3a.nano |
| **ProductDescription** | Operating system | Linux/UNIX |
| **SpotPrice**          | $/hour               | 0.061700 |
| **Timestamp**          | UTC record timestamp | 020-05-24T23:00:03+00:00 |


## Calculate the rewards
The pipeline assumes that each service arrival has an associated reward
expressed as $/h, i.e., if the service is locally deployed, the infrastructure
owner wins $ each hour it hosts the service.

Given the spot prices CSV, the following script derives the service arrivals
rewards running:
```bash
python3 calc_rewards.py\
        instance-info.json\              # JSON with the EC2 instances info.
        /tmp/ec2-march-to-may.csv\       # CSV with the spot prices history
        x2avgPrice\                      # method used to derive the reward
        /tmp/instances-info-rewards.json # path where JSON with rewards goes
```

There are two methods to derive the the reward

| method | description |
|--------|-------------|
| **x2avgPrice** |  the instance reward is twice the average spot price |
| **x2maxPrice** | the instance reward is twice the maximum spot price |



## Generate the arrivals
After executing all scripts above, one derives the arrivals of services
running:
```bash
python3 gen_ec2_arrivals.py\
    /tmp/ec2-prices-march-to-may.csv\  # path to CSV with EC2 spot prices
    /tmp/instances-info-rewards.json\  # Path to JSON with instance information
    "*"\                               # instances to include
    40\                                # lifetime std %
    /tmp/aws-arrivals.csv              # out file
```

the resulting CSV `/tmp/aws-arrivals.csv` contains a list of service arrivals 
with following information:

| Column   | description | example |
|----------|-------------|---------|
| **time** | UNIX epoch sec. since 1970 | 1582934578.9602706 |
| **instance** | AWS instance type | c5.2xlarge | 
| **spotprice** | $/h when service arrived | 0.2521 |
| **lifetime** | days the service lasts | 6.007251392402977 |
| **os** | OS required by the service | Linux/UNIX |
| **cpu** | #cpus required by the service | 8
| **memory** | GB of memory required by service | 16.0 |
| **disk** | GB of disk space required by the service | 400 |
| **reward** | $/h reward obtained by hosting the service | 0.472 |


## References
[1] Xu, Hong, and Baochun Li. "Dynamic cloud pricing for revenue maximization." IEEE Transactions on Cloud Computing 1.2 (2013): 158-171.

[2] Information Exchange to Support Multi-Domain Slice Service Provision for 5G/NFV

