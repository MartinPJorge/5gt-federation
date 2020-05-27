The AWS eu-west-3 prices were obtained using:
```aws
aws ec2 describe-spot-price-history --start-time 2020-03-01 --end-time 2020-05-25
```
having the region set to `eu-west-3` in the aws-cli configuration


`ec2-march-to-may.csv` was generated running:

```bash
python3 ec2_prices_to_csv.py aws-ec2-prices-march-to-may.json eu-west-3a "m5a.2xlarge|m5a.12xlarge|m5a.16xlarge|m5ad.2xlarge|m5ad.16xlarge|m5d.2xlarge|m5d.large|m5d.8xlarge|m5d.8xlarge|m5d.16xlarge|m5d.24xlarge|t2.large|t3.large|t3a.nano|t3a.medium|t3a.small|c5x.large|c5.2xlarge|c5.18xlarge|c5.metal|c5d.xlarge|c5d.4xlarge|c5d.18xlarge" /tmp/ec2-march-to-may.csv
```
