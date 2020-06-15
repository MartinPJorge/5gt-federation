import pandas as pd
import math
import time

class AWS_env():
    # ACTIONs macros
    A_LOCAL = 0
    A_FEDERATE = 1
    A_REJECT = 2
    ACTIONS = [A_LOCAL, A_FEDERATE, A_REJECT]
    # FUTURE
    FUTURE = pd.Timestamp('2200-01-01 00:00:00+00:00')

    def __init__(self, cpu, memory, disk, f_cpu, f_memory, f_disk,
                 arrivals, spot_prices):
        # arrivals: pandas DataFrame with columns
        #           ['time', 'instance', 'spotprice', 'cpu', 'memory',
        #            'os', 'disk', 'reward', 'lifetime']
        #           the reward is expressed as $/hour
        #           lifetime in days
        #           time is the UNIX epoch in seconds since 1970
        # spot_prices: pandas DataFrame with columns
        #              ['AvailabilityZone', 'InstanceType',
        #               'ProductDescription', 'SpotPrice', 'Timestamp']

        self.cpu, self.max_cpu = cpu, cpu
        self.disk, self.max_disk = disk, disk
        self.memory, self.max_memory = memory, memory
        self.f_cpu, self.max_f_cpu = f_cpu, f_cpu
        self.f_disk, self.max_f_disk = f_disk, f_disk
        self.f_memory, self.max_f_memory = f_memory, f_memory
        self.time = float(0)
        self.capacity = [int(cpu), int(memory), int(disk)]
        self.f_capacity = [int(f_cpu), int(f_memory), int(f_disk)] if\
            f_cpu != math.inf else [math.inf, math.inf, math.inf]
        self.profit = float(0)
        self.arrivals = arrivals
        self.service_length = []
        self.total_num_services = int(0)
        # INDEXING
        self.curr_idx = 0 # current index in the arrivals DataFrame
        self.in_local = [] # list of arrival indexes in local domain
        self.in_fed = [] # list of arrival indexes in federated domain
        # ARRIVAL reward
        self.max_reward = 0
        # SPOT PRICE
        self.spot_prices = spot_prices
        self.spot_prices.sort_values(by='Timestamp', ascending=True,
                                     inplace=True)
        self.spot_prices['Timestamp'] =\
            pd.to_datetime(self.spot_prices['Timestamp'])
        self.max_spot_prices = {
            i: 0
            for i in spot_prices['InstanceType'].unique()
        }

        # Create the dataframes for each spot instance
        unique_instances = self.spot_prices[['InstanceType',
            'ProductDescription']].drop_duplicates()
        self.instance_prices = {
            (instance, os): self.spot_prices[
                    (self.spot_prices['InstanceType'] == instance) &
                    (self.spot_prices['ProductDescription'] == os)
                ][['Timestamp', 'SpotPrice']]
            for instance, os in unique_instances[['InstanceType',
                'ProductDescription']].values
        }
        # Cache to store the computed pricing intervals
        # Note: the cache is per-time-step basis
        self.pricing_cache = {}



    def get_state(self):
        if self.curr_idx == len(self.arrivals) - 1:
            return None

        arrival = self.arrivals.iloc[self.curr_idx]
        instance = arrival['instance']
        arrival_t = pd.Timestamp(arrival['time'], unit='s', tz='UTC')

        self.max_reward = max(self.max_reward, arrival['reward'])

        # Find what is the latest spot price for that instance
        spot_price_df = self.spot_prices[
                    self.spot_prices['InstanceType'] == instance]
        # spot_price_df.sort_values(by='Timestamp', ascending=True, inplace=True)
        past_prices = spot_price_df[spot_price_df['Timestamp'] <= arrival_t]
        if len(past_prices) == 0:
            spot_price = spot_price_df.iloc[0]['SpotPrice']
        else:
            spot_price = past_prices.iloc[-1]['SpotPrice']
        
        # Update the max spot price for that instance
        if spot_price > self.max_spot_prices[instance]:
            self.max_spot_prices[instance] = spot_price


        return [
            # Local resources
            self.cpu    / self.max_cpu,
            self.disk   / self.max_disk,
            self.memory / self.max_memory,
            # Federated resources
            self.f_cpu    / self.max_f_cpu,
            self.f_disk   / self.max_f_disk,
            self.f_memory / self.max_f_memory,
            # Arrival local resources consumption
            arrival['cpu']    / self.cpu    if self.cpu    > 0 else 1.1,
            arrival['disk']   / self.disk   if self.disk   > 0 else 1.1,
            arrival['memory'] / self.memory if self.memory > 0 else 1.1,
            # Arrival reward per hour
            arrival['reward'] / self.max_reward,
            # Arrival spot price
            spot_price / max(p for p in self.max_spot_prices.values())
        ]

        # return [self.cpu, self.disk, self.memory,
        #         self.f_cpu, self.f_disk, self.f_memory]


    def take_action(self, action):
        # a={1,2,3} local deployment, federate, reject
        # returns the reward and next state
        # in case it is the last state, it returns None as state

        # t = t + 1
        reward = 0
        curr_arrival = self.arrivals.iloc[self.curr_idx]
        curr_time = curr_arrival['time']
        self.curr_idx += 1
        next_time = self.arrivals.time.iloc[self.curr_idx]

        # Assign the resources based on the action
        asked_cpu = curr_arrival['cpu']
        asked_memory = curr_arrival['memory']
        asked_disk = curr_arrival['disk']
        print(f'asked resources = (CPU={asked_cpu}, mem={asked_memory},disk={asked_disk})')
        if action == AWS_env.A_LOCAL:
            if self.cpu < asked_cpu or self.memory < asked_memory or\
                    self.disk < asked_disk:
                print('local action but NO RESSSSSSSSS')
                reward -= curr_arrival['reward']
            else:
                self.cpu -= asked_cpu
                self.memory -= asked_memory
                self.disk -= asked_disk
                self.in_local.append(self.curr_idx)
        elif action == AWS_env.A_FEDERATE:
            if self.f_cpu < asked_cpu or self.f_memory < asked_memory or\
                    self.f_disk < asked_disk:
                print('federate action but NO RESSSSSSSSS')
                reward -= curr_arrival['reward']
            else:
                self.f_cpu -= asked_cpu
                self.f_memory -= asked_memory
                self.f_disk -= asked_disk
                self.in_fed.append(self.curr_idx)
        elif action == AWS_env.A_REJECT:
            pass

        # calculate the reward from [t, t+1]
        reward += self.__calc_reward(curr_time, next_time)
        # services leave
        self.__free_resources(next_time)
        # Reset the time-step cache
        self.pricing_cache = {}

        return reward, self.get_state() # it'll handle the episode END


    def __cache_get_interval(self, instance, os, prev, until):
        if (instance, os, prev, until) not in self.pricing_cache:
            return []
        else:
            return self.pricing_cache[(instance, os, prev, until)]


    def __get_pricing_intervals(self, prev_time, until, arrival_idx):
        # creates a list of spot pricing betweeen [prev, until]
        # the returned is a list of lists like
        #   [ [time0, spot_price0], ... [timeN, spot_priceN] ]
        # with time0>=prev_time and timeN<=until
        #
        #
        # NOTE: this functions handles with the below corner cases in
        #       which the prev and until values do not lie within the
        #       SpotPrices history
        #
        #
        #                 --- prev     
        #                  |          --- prev
        #                 --- until    | 
        #    t1,$1 ---                 |
        #           |                  |
        #           |                 --- until
        #           |     --- prev                  
        #           |      |               
        #           |      |                       
        #    t2,$2 ---    ---
        #           |      |           --- prev
        #           |     --- until     |
        #           |                   |
        #           |                  --- until
        #    t3,$3 ---
        #     
        #       SpotPrices
        #
        #def __get_pricing_intervals(self, prev_time, until, arrival_idx):

        # Create timestamp versions for prev and until
        prev_ts = pd.Timestamp(prev_time, unit='s', tz='UTC')
        until_ts = pd.Timestamp(until, unit='s', tz='UTC')

        # Get the spot prices history of the arrival instance
        arrival = self.arrivals.iloc[arrival_idx]

        # If the interval has been already computed, return it
        cached_interval = self.__cache_get_interval(arrival['instance'],
                arrival['os'], prev_ts, until_ts)
        if len(cached_interval) > 0:
            print('CACHE HIT')
            return cached_interval

        ### tic = time.time()
        ### spot_history = self.spot_prices[(self.spot_prices['InstanceType'] ==\
        ###                                 arrival['instance']) &\
        ###                             (self.spot_prices['ProductDescription'] ==\
        ###                             arrival['os'])]
        ### print(f'\n\t\tfilter instances take {time.time() - tic}')
        ### spot_history = spot_history[['Timestamp', 'SpotPrice']]
        ### tic = time.time()
        ### # spot_history.sort_values(by='Timestamp', ascending=True, inplace=True)
        ### print(f'\t\tsorting spot prices take {time.time() - tic}')
        spot_history = self.instance_prices[arrival['instance'], arrival['os']]
        spot_historyFst = spot_history.iloc[0]

        # DEBUG the head
        # print('HHEAD')
        # print(spot_history.head())
        # print(f'PREV={pd.Timestamp(prev_time, unit="s")}, UNTIL={pd.Timestamp(until, unit="s")}')

        # until < t1
        if until < spot_historyFst['Timestamp'].timestamp():
            interval = [[prev_ts, spot_historyFst['SpotPrice']],
                    [until_ts, spot_history.iloc[1]['SpotPrice']]]
            self.pricing_cache[(arrival['instance'], arrival['os'],
                                prev_ts, until_ts)] = interval
            return interval 

        tic = time.time()
        spot_history_until = spot_history[
                                spot_history['Timestamp'] <= until_ts]
        print(f'\t\tfilter until takes {time.time() - tic}')

        # prev_time < t1  &  until >= t1
        if prev_ts < spot_historyFst['Timestamp']:
            ret_prices = spot_history_until.values
            ret_prices = [list(sp_t) for sp_t in ret_prices]
            print('prev_time < t1  &  until >= t1')
            print(f'type(ret_prices[0][0])={type(ret_prices[0][0])}')
            print('ret_prices')
            print(ret_prices)
            if len(spot_history_until) == 1:
                ret_prices = [[prev_ts, spot_historyFst['SpotPrice']]] +\
                             ret_prices
            ret_prices[-1][0] = until_ts

            self.pricing_cache[(arrival['instance'], arrival['os'],
                                prev_ts, until_ts)] = ret_prices
            return ret_prices

        # prev_time >= t1  &  until >= t1
        tic = time.time()
        spot_history_between = spot_history_until[
                spot_history_until['Timestamp'] >= prev_ts]
        print(f'\t\tfilter after prev takes {time.time() - tic}')

        # no tn:  prev_time <= tn <= until
        if len(spot_history_between) == 0:
            spot_history_untilLst = spot_history_until.iloc[-1]
            interval = [
                [prev_ts, spot_history_untilLst['SpotPrice']],
                [until_ts, spot_history_untilLst['SpotPrice']]
            ]
            self.pricing_cache[(arrival['instance'], arrival['os'],
                                prev_ts, until_ts)] = interval
            return interval

        # !E tn:  prev_time <= tn <= until
        if len(spot_history_between) == 1:
            spot_history_betweenLst = spot_history_between.iloc[-1]
            interval = [
                [prev_ts, spot_history_betweenLst['SpotPrice']],
                [until_ts, spot_history_betweenLst['SpotPrice']]
            ]
            self.pricing_cache[(arrival['instance'], arrival['os'],
                                prev_ts, until_ts)] = interval
            return interval

        # E {tn, tn+1, ...}: prev_time <= tn <= until
        ret_prices = spot_history_between.values
        #print(f'ret_prices={ret_prices}')
        ret_prices[0][0] = prev_ts
        ret_prices[-1][0] = until_ts

        self.pricing_cache[(arrival['instance'], arrival['os'],
                            prev_ts, until_ts)] = ret_prices
        return ret_prices



    def __calc_arrival_reward(self, prev_time, curr_time, arrival_idx,
                              federated):
        # Compute the reward
        arrival = self.arrivals.iloc[arrival_idx]
        t_end = arrival['lifetime']*24*60*60 + arrival['time']
        until = min(t_end, curr_time)
        reward = self.arrivals.iloc[arrival_idx]['reward'] *\
                 (until - prev_time) / (60*60) 

        if not federated:
            return reward

        #######################################################################
        # From here down we substract the spot price for the federated arrival
        #######################################################################


        pricing = self.__get_pricing_intervals(prev_time, until, arrival_idx)
        print(f'FEDERATED arrival={arrival_idx}')
        print(f'reward={reward}')
        print(f'pricing={pricing}')
        penalty = 0
        for end, begin in zip(pricing[:-1], pricing[1:]):
            delta = (end[0].timestamp() - begin[0].timestamp()) / (60*60) # h
            print('\t\tpenaly iter')
            penalty += delta * begin[1]
            reward -= delta * begin[1]
        print(f'\tpenalty={penalty}')


        return reward


    def __calc_reward(self, prev_time, curr_time):
        # compute the reward in between [prev_time,curr_time]
        # for local and federated instances
        reward = 0

        st_loc = time.time()
        for local_idx in self.in_local:
            reward += self.__calc_arrival_reward(prev_time, curr_time,
                    local_idx, federated=False)
        print(f'\tit takes {time.time() - st_loc} to obtain local rewards')
        st_fed = time.time()
        for fed_idx in self.in_fed:
            reward += self.__calc_arrival_reward(prev_time, curr_time,
                    fed_idx, federated=True)
        print(f'\tit takes {time.time() - st_fed} to obtain fed rewards')

        return reward


    def __free_resources(self, curr_time):
        remove_local, remove_fed  = [], []

        print(f' ENV:: local deployed={self.in_local}')
        print(f' ENV:: fed deployed={self.in_fed}')

        # Check local arrivals that have expired
        for local_idx in self.in_local:
            expires = self.arrivals.iloc[local_idx]['time'] +\
                self.arrivals.iloc[local_idx]['lifetime']*24*60*60
            if expires <= curr_time:
                remove_local.append(local_idx)

        # Check federated arrivals that have expired
        for fed_idx in self.in_fed:
            expires = self.arrivals.iloc[fed_idx]['time'] +\
                self.arrivals.iloc[fed_idx]['lifetime']*24*60*60
            if expires <= curr_time:
                remove_fed.append(fed_idx)

        # Remove the arrivals from the lists, and free resources
        for rem_loc_idx in remove_local:
            print(f'remove local idx = {rem_loc_idx}')
            print(f'loc list= {self.in_local}')
            del self.in_local[self.in_local.index(rem_loc_idx)]
            self.cpu    += self.arrivals.iloc[rem_loc_idx]['cpu']
            self.disk   += self.arrivals.iloc[rem_loc_idx]['disk']
            self.memory += self.arrivals.iloc[rem_loc_idx]['memory']
        for rem_fed_idx in remove_fed:
            print(f'remove federeated idx = {rem_fed_idx}')
            print(f'fed list= {self.in_fed}')
            del self.in_fed[self.in_fed.index(rem_fed_idx)]
            self.f_cpu    += self.arrivals.iloc[rem_fed_idx]['cpu']
            self.f_disk   += self.arrivals.iloc[rem_fed_idx]['disk']
            self.f_memory += self.arrivals.iloc[rem_fed_idx]['memory']


    def reset(self):
        self.__free_resources(curr_time=AWS_env.FUTURE.timestamp())
        self.curr_idx = 0
