set instance_types;
set instances;
set timestamps ordered; # UNIX epoch-1970

# Old parameters 
#param profit_federate {timestamps};
#param profit_local {timestamps};
#param profit_reject {timestamps};

param itype {instances} symbolic;
param margin; # 0.2 above the federate_fee
param federate_fee {instance_types, timestamps};
param instance_arrival {instances} in timestamps;
param instance_departure {instances} >= 0;
#param available {instances, timestamps} binary;
param asked_cpu {timestamps};
param asked_mem {timestamps};
param asked_disk {timestamps};
param frees_cpu {timestamps};
param frees_mem {timestamps};
param frees_disk {timestamps};
param frees_arrival {timestamps}; # Arrival timestamp of leaving
                                  # service

var federate {timestamps} binary;
var local {timestamps} binary;
var reject {timestamps} binary;
var cpu {timestamps};
var mem {timestamps};
var disk {timestamps};
var federation_cpu {timestamps};
var federation_mem {timestamps};
var federation_disk {timestamps};

# Local resource conservation
subject to match_cpu {t in timestamps: first(timestamps) <> t}:
        cpu[t] = cpu[prev(t)] - local[prev(t)]*asked_cpu[prev(t)]
            + local[frees_arrival[prev(t)]]*frees_cpu[prev(t)];
subject to match_mem {t in timestamps: first(timestamps) <> t}:
        mem[t] = mem[prev(t)] - local[prev(t)]*asked_mem[prev(t)]
            + local[frees_arrival[prev(t)]]*frees_mem[prev(t)];
subject to match_disk {t in timestamps: first(timestamps) <> t}:
        disk[t] = disk[prev(t)] - local[prev(t)]*asked_disk[prev(t)]
            + local[frees_arrival[prev(t)]]*frees_disk[prev(t)];

# Federation resource conservation
subject to match_federation_cpu {t in timestamps: first(timestamps) <> t}:
        federation_cpu[t] = federation_cpu[prev(t)]
            - federate[prev(t)]*asked_cpu[prev(t)]
            + federate[frees_arrival[prev(t)]]*frees_cpu[prev(t)];
subject to match_federation_mem {t in timestamps: first(timestamps) <> t}:
        federation_mem[t] = federation_mem[prev(t)] - federate[prev(t)]*asked_mem[prev(t)]
            + federate[frees_arrival[prev(t)]]*frees_mem[prev(t)];
subject to match_federation_disk {t in timestamps: first(timestamps) <> t}:
        federation_disk[t] = federation_disk[prev(t)] - federate[prev(t)]*asked_disk[prev(t)]
            + federate[frees_arrival[prev(t)]]*frees_disk[prev(t)];

subject to choose_one_option {t in timestamps}:
    federate[t] + local[t] + reject[t] = 1;

# Locally don't run out of resources
subject to no_cpu_runout {t in timestamps}:
    cpu[t] >= 0;
subject to no_mem_runout {t in timestamps}:
    mem[t] >= 0;
subject to no_disk_runout {t in timestamps}:
    disk[t] >= 0;

# Federated domain don't run out of resources
subject to no_federation_cpu_runout {t in timestamps}:
    federation_cpu[t] >= 0;
subject to no_federation_mem_runout {t in timestamps}:
    federation_mem[t] >= 0;
subject to no_federation_disk_runout {t in timestamps}:
    federation_disk[t] >= 0;


# Note: as an instance might leave in between [t-1,t]
#       we compute the profit in between [t-1, min(t,departure)]
maximize dynamic_profit:
    sum {i in instances, t in timestamps: t <> first(timestamps) and
            instance_arrival[i] <= prev(t) and prev(t) <= instance_departure[i]}
        ( min(t, instance_departure[i]) - prev(t) ) / (60*60) *
        (local[instance_arrival[i]] 
            * (1+margin) * federate_fee[itype[i],instance_arrival[i]]
        + federate[instance_arrival[i]] 
            * ( (1+margin) * federate_fee[itype[i],instance_arrival[i]] 
                - federate_fee[itype[i],prev(t)] ) );
       
         

# Old profit function
## maximize total_profit:
##     sum {t in timestamps} (profit_federate[t] * federate[t] +
##         profit_local[t] * local[t] +
##         profit_reject[t] * reject[t]);

