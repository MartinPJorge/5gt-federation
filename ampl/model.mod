set timestamps;

param profit_federate;
param profit_local;
param profit_reject;
param frees_cpu {timestamps};
param frees_mem {timestamps};
param frees_disk {timestamps};

var federate {t in timestamps} binary;
var local {t in timestamps} binary;
var reject {t in timestamps} binary;
var cpu {t in timestamps: card(t) > 1} :=
        cpu[t] - local[prev(t)]*asked_cpu[prev(t)]
            + local[prev(t)]*frees_cpu[prev(t)];
var mem {t in timestamps: card(t) > 1} :=
        mem[t] - local[prev(t)]*asked_mem[prev(t)]
            + local[prev(t)]*frees_mem[prev(t)];
var disk {t in timestamps: card(t) > 1} :=
        disk[t] - local[prev(t)]*asked_disk[prev(t)]
            + local[prev(t)]*frees_disk[prev(t)];

subject to choose_one_option {t in timestamps}:
    federate[t] + local[t] + reject[t] = 1;

subject to no_cpu_runout {t in timestamps}:
    cpu[t] >= 0;
subject to no_mem_runout {t in timestamps}:
    mem[t] >= 0;
subject to no_disk_runout {t in timestamps}:
    disk[t] >= 0;

maximize total_profit:
    sum {t in timestamps} profit_federate * federate[t] +
        profit_local * local[t] +
        profit_reject * reject[t];

