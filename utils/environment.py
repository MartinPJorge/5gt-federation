#import numpy as np

class Env:
    def __init__(self, cpu, disk, memory):
        self.cpu = cpu
        self.disk = disk
        self.memory = memory
        self.time = float(0)
        self.capacity = [float(cpu), float(memory), float(disk)]
        self.profit = float(0)
        self.service_cpu = []
        self.service_disk = []
        self.service_memory = []
        self.service_arrival_time = []
        self.service_length = []


    def reset(self):
        self.cpu = float(self.capacity[0])
        self.disk = float(self.capacity[1])
        self.memory = float(self.capacity[2])
        self.time = float(0)
        self.profit = float(0)
        self.service_cpu = []
        self.service_disk = []
        self.service_memory = []
        self.service_arrival_time = []
        self.service_length = []

    def current_capacity(self):
        return [int(self.cpu), int(self.memory), int(self.disk)]
    def print_status(self):
        print("Environment status:\n")
        print("\tCPU: " + str(self.cpu))
        print("\tDisk: " + str(self.disk))
        print("\tMemory: " + str(self.memory))
        print("\tTime: " + str(self.time) + " seconds")
        print("\tProfit: " + str(self.profit))
        print("\tNumber of Services: "+ str(len(self.service_length)))


    def totalCapacity(self):
        print("Environment capacity:\n ")
        print("\tCPU: " + str(self.capacity[0]))
        print("\tDisk: " + str(self.capacity[1]))
        print("\tMemory: " + str(self.capacity[2]))

    def update_domain(self, current_time):
        self.time = current_time
        if len(self.service_arrival_time) > 0:
            for i in range((len(self.service_arrival_time)-1), -1, -1):

                if float(current_time) > (float(self.service_arrival_time[i]) + float(self.service_length[i])):
                    self.cpu += float(self.service_cpu[i])
                    self.disk += float(self.service_disk[i])
                    self.memory += float(self.service_memory[i])
                    del self.service_cpu[i]
                    del self.service_memory[i]
                    del self.service_disk[i]
                    del self.service_arrival_time[i]
                    del self.service_length[i]


            return int(len(self.service_length)), float(self.profit)
        else:
            return int(0), float(self.profit)

    def instantiate_service(self, service_cpu, service_memory, service_disk, service_profit, arrival_time, service_length):
        if self.cpu >= service_cpu and self.memory >= service_memory and self.disk >= service_disk \
                and float(self.time) <= float(arrival_time):
            self.cpu -= service_cpu
            self.memory -= service_memory
            self.disk -= service_disk
            self.profit += service_profit
            self.service_cpu.append(service_cpu)
            self.service_memory.append(service_memory)
            self.service_disk.append(service_disk)
            self.service_arrival_time.append(arrival_time)
            self.service_length.append(service_length)
            # print("New service added")
            return int(len(self.service_length)), float(self.profit)
        else:
            # print("Error!")
            return int(len(self.service_length)), float(self.profit)

    def get_num_services(self):
        return int(len(self.service_length))

    def get_cpu(self):
        return self.cpu

    def get_memory(self):
        return self.memory

    def get_disk(self):
        return self.disk

    def get_profit(self):
        return float(self.profit)
