import numpy as np

class Env:
    def __init__(self, cpu, memory, disk):
        self.cpu = cpu
        self.disk = disk
        self.memory = memory
        self.time = float(0)
        self.capacity = [int(cpu), int(memory), int(disk)]
        self.profit = float(0)
        self.service_cpu = []
        self.service_disk = []
        self.service_memory = []
        self.service_arrival_time = []
        self.service_length = []
        self.federated_service_arrival = []
        self.federated_service_length = []
        self.total_num_services = int(0)


    def reset(self):
        self.cpu = float(self.capacity[0])
        self.memory = float(self.capacity[1])
        self.disk = float(self.capacity[2])
        self.time = float(0)
        self.profit = float(0)
        self.service_cpu = []
        self.service_disk = []
        self.service_memory = []
        self.service_arrival_time = []
        self.service_length = []
        self.federated_service_arrival = []
        self.federated_service_length = []
        self.total_num_services = int(0)

    def current_capacity(self):
        return [int(self.cpu), int(self.memory), int(self.disk)]

    def calculate_state(self):
        return int((self.cpu*self.capacity[1]*self.capacity[2]) + (self.memory*self.capacity[2]) + (self.disk-1))

    def capacity_to_state(self,cpu,memory,disk):
        return int((cpu*self.capacity[1]*self.capacity[2]) + (memory*self.capacity[2]) + (disk-1))

    def state_to_capacity(self, state):
        state = int(state+1)
        cpu = divmod(state, int((self.capacity[1]*self.capacity[2])))
        if cpu[0]<=self.capacity[0]:
            value_cpu = cpu[0]
        else:
            value_cpu = self.capacity[0]
        state = int(state-(value_cpu*self.capacity[1]*self.capacity[2]))
        if state == 0:
            value_memory = 0
            value_disk = 0
        else:
            memory = divmod(int(state), int(self.capacity[2]))
            if memory[0]<=self.capacity[1]:
                value_memory = memory[0]
            else:
                value_memory = self.capacity[1]
            state = int(state - (value_memory*self.capacity[2]))
            if state == 0:
                value_disk = 0
            else:
                value_disk = int(state)
        return [value_cpu, value_memory, value_disk]

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
        if self.total_num_services > 0:
            for i in range((len(self.service_arrival_time)-1), -1, -1):

                if float(current_time) > (float(self.service_arrival_time[i]) + float(self.service_length[i])):
                    self.cpu += float(self.service_cpu[i])
                    self.disk += float(self.service_disk[i])
                    self.memory += float(self.service_memory[i])
                    self.total_num_services -= 1
                    del self.service_cpu[i]
                    del self.service_memory[i]
                    del self.service_disk[i]
                    del self.service_arrival_time[i]
                    del self.service_length[i]

            for j in range((len(self.federated_service_arrival)-1), -1, -1):
                if float(current_time) > (float(self.federated_service_arrival[j]) + float(self.federated_service_length[j])):
                    self.total_num_services -= 1
                    del self.federated_service_length[j]
                    del self.federated_service_arrival[j]

            return int(self.calculate_state()), float(self.profit), self.total_num_services
        else:
            return int(0), float(self.profit), self.total_num_services

    def instantiate_service(self, action, service_cpu, service_memory, service_disk, service_profit, arrival_time, service_length):
        if action == 2:
            # rejection
            self.profit -= service_profit
            return [self.cpu, self.memory, self.disk, -(service_profit)]
        elif action == 1:
            # FEDERATION
            if float(self.time) <= float(arrival_time):
                self.federated_service_arrival.append(arrival_time)
                self.federated_service_length.append(service_length)
                k = float(self.capacity[0]-self.cpu)/self.capacity[0]
                l = float(self.capacity[1]-self.memory)/self.capacity[1]
                m = float(self.capacity[2]-self.disk)/self.capacity[2]

                # k = 1 - float(self.capacity[0]-self.cpu)/self.capacity[0]
                # l = 1 - float(self.capacity[1]-self.memory)/self.capacity[1]
                # m = 1 - float(self.capacity[2]-self.disk)/self.capacity[2]
                coeff = k+l+m
                print("Coeff:", coeff)
                # coeff = np.random.uniform(0,service_profit)
                now_profit = service_profit*coeff if coeff > 0 else (-5*service_profit)
                self.profit += now_profit
                self.total_num_services += 1
                return [self.cpu, self.memory, self.disk, now_profit]
        else:
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
                self.total_num_services += 1
                # print("New service added")
                return [self.cpu, self.memory, self.disk, service_profit]
            else:
                print("penalty!")
                self.profit -= (1.2*service_profit)
                return [self.cpu, self.memory, self.disk, (-3.2*service_profit)]

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
