# 5GT federation simulations
This repository is oriented to perform simulations for federation
interactions in 5G networks.

### Clone with submodule
This repository uses the `vnfs-mapping` submodule to generate the network
service graphs. To clone with the submodule run:
```bash
git clone --recursive https://github.com/MartinPJorge/5gt-federation.git
```

## NS jukebox
The NS jukebox tool `utils/ns_jukebox.py` randomly generates network service
graphs based on a set of characteristics specified within a configuration file
as the one in `utils/config/nsGenConf.json`.

These are the parameters available in the configuration file:
 * **splits**: number of splits within the NS graph. A split is the point in
   the network service graph where a VNF1 receives traffic only from VNF0, but
   then forwards traffic to VNF2...VNFn;
 * **splitWidth**: number of VNFs after the VNF split, if VNF1 forwards traffic
   towards VNF2, VNF3, VNF4; the split width is equal to 3;
 * **branches**: maximum number of parallel flows within the NS graph;
 * **vnfs**: maximum number of VNFs within the NS graph;
 * **linkTh**: (min,max) values to generate the virtual links characteristics
   below. If `linkTh["traffic"]["min"]=2` and `linkTh["traffic"]["min"]=5`,
   then the algorithm can generate a bandwidth demand between 2 and 5.
   * **linkTh["traffic"]**: (min,max) interval for virtual link bandwidth;
   * **linkTh["delay"]**: (min,max) interval for virtual link bandwidth;
 * **vnfTh**: (min,max) values to generate the VNF characteristics below.
   * **vnfTh["processing-time"]**: VNF processing time required by a VNF;
   * **vnfTh["processing-time"]["cpu"]**: VNF required CPU;
   * **vnfTh["processing-time"]["memory"]**: VNF required memory;
   * **vnfTh["processing-time"]["storage"]**: VNF required disk;

### Execution example
Following example generates 2 NS graphs based on the example configuration
file, and stores the virtual links and VNFs CSVs under `/tmp`:
```bash
$ python ns_jukebox.py config/nsGenConf.json 2 /tmp
Generating 0-th NS for config config/nsGenConf.json
  VNF CSV at: /tmp/vls-0.csv
  VL CSV at: /tmp/vnfs-0.csv
Generating 1-th NS for config config/nsGenConf.json
  VNF CSV at: /tmp/vls-1.csv
  VL CSV at: /tmp/vnfs-1.csv
```

### Output CSV examples
For each generated network service, two files are created. One for the virtual
links:
```csv
delay,traffic,prob,idVNFa,idVNFb
2,110,1,1,2
1,822,0.2576827604988282,2,3
5,537,0.7354732267205919,2,4
4,268,0.006844012780579889,2,5
5,861,1,3,6
5,552,1,4,7
1,770,1,5,7
4,948,1,6,7
4,852,1,7,8
2,531,1,8,9
2,362,1,9,10
```
and another for the VNFs:
```csv
requirements,processing_time,id,memory,vnf_name,disk,cpu,idVNF
{},6,,400,v_gen_1_9_1553095419.88,5418,21,1
{},16,,921,v_gen_2_3_1553095419.88,2458,7,2
{},11,,597,v_gen_3_2_1553095419.89,8699,14,3
{},2,,579,v_gen_4_16_1553095419.89,5207,6,4
{},19,,369,v_gen_5_10_1553095419.89,4888,21,5
{},7,,868,v_gen_6_18_1553095419.89,7408,16,6
{},1,,573,v_gen_7_18_1553095419.89,4895,2,7
{},3,,356,v_gen_8_18_1553095419.89,9714,13,8
{},1,,476,v_gen_9_16_1553095419.89,4828,5,9
{},8,,226,v_gen_10_15_1553095419.9,8732,24,10
```


#### Acknowledgements
5G-TRANSFORMER Project under Grant 761536. Parts of this paper have
been published in the Proceedings of the IEEE BMSB 2018, Valencia, Spain.
