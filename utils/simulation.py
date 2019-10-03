import csv
import numpy as np
import argparse
import json
from environment import Env
from matplotlib import pyplot as plt


if __name__ == '__main__':
    small_arrivals = []
    big_arrivals = []
    big_cpus, big_mems, big_disks, big_time = [], [], [], []
    small_cpus, small_mems, small_disks, small_time = [], [], [], []
    # Parse args
    parser = argparse.ArgumentParser(description='Generate arrivals of' + \
                                                 'Services')
    parser.add_argument('configFile', type=str,
                        help='path to resources config file')
    parser.add_argument('--inCSV', type=str, default=None,
                        help='CSV path where results are stored')
    args = parser.parse_args()

    #config file for Administrative domain capacity
    domain_config = None
    with open(args.configFile) as f:
        domain_config = json.load(f)


    print(domain_config["cpu"])
    # Write CSV only if asked
    if args.inCSV:
        with open(args.inCSV, mode='r') as f:
            arrival_reader = csv.DictReader(f)
            for row in arrival_reader:
                if float(row["big"]) > float(0):
                    big_cpus += [float(row["cpu"])]
                    big_disks += [float(row["disk"])]
                    big_mems += [float(row["mem"])]
                    big_time += [float(row["arrival_time"])]
                else:
                    small_cpus += [float(row["cpu"])]
                    small_disks += [float(row["disk"])]
                    small_mems += [float(row["mem"])]
                    small_time += [float(row["arrival_time"])]

    print("SMALL (last 5)")
    print("\tarrivals: " + str(small_time[-5:]))
    print("\tcpus: " + str(small_cpus[-5:]))
    print("\tdisk: " + str(small_disks[-5:]))

    # Environment initialization
    simTime = 30  # seconds
    # startSim = True
    # stepTime = 0.005  # seconds

    domain_cpu = int(domain_config["cpu"])
    domain_disk = int(domain_config["disk"])
    domain_memory = int(domain_config["mem"])

    print("Domain:")
    print("\tcpu: " + str(domain_cpu))
    print("\tmemory: " + str(domain_memory))
    print("\tdisk: " + str(domain_disk))


    env = Env(domain_cpu,domain_memory,domain_disk)
    env.print_status()
    current_time = 0
    alpha = 0.75
    discount = 0.95
    sim_active = True
    i = -1
    j = -1
    print(i,j)
    arrival_cpu = 0
    arrival_mem = 0
    arrival_disk = 0
    arrival_length = 0
    env_capacity = []
    rewards = []
    print(env.get_profit())

    tot_states = int((domain_cpu*domain_memory*domain_disk) + (domain_memory*domain_disk) + domain_disk)
    tot_actions = 3
    Q = np.zeros(shape=(tot_states,tot_actions), dtype=np.int)
    state = env.calculate_state()
    print("total_length:",tot_states)
    print("Q length: ",len(Q))
    print("State:", str(env.calculate_state()))
    print("Reverse calculate (state):", str(env.state_to_capacity(state)))
    print("TRIAL CAPACITY TO STATE [4,15,9] =", str(env.capacity_to_state(4,15,9)))
    print("Reverse calculate (state):", str(env.state_to_capacity(26568)))

    episodes= 100
    episode_reward = []
    actions = []
    num_services = 0

    for episode in range(episodes):
        sim_active = True

        while True:
            if((i+1) >= len(small_time)):
                episode_reward.append(tot_profit)
                i=-1
                j=-1
                env.reset()
                sim_active = False
                break

    #   GET ARRIVAL INFORMATION
            if((j+1) < len(big_time) and big_time[j+1]<small_time[i+1]):
                j += 1
                current_time = float(big_time[j])
                arrival_cpu = big_cpus[j]
                arrival_mem = big_mems[j]
                arrival_disk = big_disks[j]
                arrival_profit = np.random.uniform(5.0, 10.0)
                arrival_length = np.random.uniform(5.0, 10.0)
            else:
                i += 1
                current_time = float(small_time[i])
                arrival_cpu = small_cpus[i]
                arrival_mem = small_mems[i]
                arrival_disk = small_disks[i]
                arrival_profit = np.random.uniform(1.0, 3.0)
                arrival_length = np.random.uniform(1.0, 4.0)


            # UPDATE ENVIRONMENT WITH LATEST TIME
            state, tot_profit, num_services = env.update_domain(current_time)

            action = np.argmax(Q[state, :] + np.random.randn(1, tot_actions) * (1 / float(episode + 1)))
            print("time:", current_time)
            print("action:", action)
            actions.append(action)
            now_cpu, now_memory, now_disk, now_profit = env.instantiate_service(action, arrival_cpu, arrival_mem, arrival_disk, arrival_profit, current_time, arrival_length)
            updated_state = env.capacity_to_state(now_cpu, now_memory, now_disk)
            print("Profit of step:", now_profit)
            print("Profit total:", tot_profit+now_profit)
            print("Capacity:", str(now_cpu), str(now_memory), str(now_disk), "\n")

            # print("State", str(updated_state))
            # print("Reverse calculate (state):", str(env.state_to_capacity(updated_state)), "\n")

            Q[state, action] += alpha * (now_profit + discount * np.max(Q[updated_state, :]) - Q[state, action])
            unique, counts = np.unique(actions, return_counts=True)
            total_actions = dict(zip(unique, counts))
            rewards.append(tot_profit)

        print("Episode:", episode)
        print("Rewards:", episode_reward[episode-1])
        print("Actions:", total_actions)
        print("Num.Services:", num_services)



    x = np.arange(0, len(rewards), 1)
    fig, ax = plt.subplots()
    ax.plot(x, rewards)
    plt.show()

    x = np.arange(0, len(episode_reward), 1)
    fig, ax = plt.subplots()
    ax.plot(x, episode_reward)
    plt.show()

    # env_capacity = env.current_capacity()
    # print("before", num_services, tot_profit)
    # print("Domain capacity:", str(env_capacity))
    # num_services, tot_profit = env.instantiate_service(arrival_cpu, arrival_mem, arrival_disk, arrival_profit,
    #                                                    current_time, arrival_length)
    # env_capacity = env.current_capacity()
    #
    # print("after", num_services, tot_profit)
    # print("Domain capacity:", str(env_capacity))
    # print("\n")
    # # print("new service added",num_services)

    # env.status()
    # env.totalCapacity()
    #
    # env.update(float(100), float(100), float(100), float(2), float(50), 100)
    # env.status()
    # env.check(float(60))
    # env.status()
    # env.update(float(100), float(100), float(100), float(2), float(70), 100)
    # env.status()
    #
    # env.check(float(90))
    # env.status()
    #
    # env.check(float(100))
    # env.check(float(169))
    # env.status()
    # env.check(float(171))
    #
    # env.status()
    # # Q and rewards
    # Q = np.zeros(shape=(4, 11, 5), dtype=np.float)
    # action = np.zeros(shape=(4), dtype=np.uint)
    #
    # rewards = []
    # iterations = []
    #
    # # Parameters
    # alpha = 0.75
    # discount = 0.95
    # episodes = 100
    #
    # # Episodes
    # for episode in range(episodes):
    #     print("###########  Episode: ", episode)
    #     # Refresh state
    #     state = env.reset()
    #     # print('state:', state)
    #     state = np.uint(np.array(state, dtype=np.uint32))
    #     done = False
    #     t_reward = 0
    #
    #     i = 0
    #     # Run episode
    #     while True:
    #         if done:
    #             break
    #
    #         i += 1
    #         current = state
    #
    #         for n in range(4):
    #             action[n] = np.argmax(Q[n, current[n], :] + np.random.randn(1, 5) * (1 / float(episode + 1)))
    #
    #             #        saction = np.uint(action * 100) + 1
    #         saction = action
    #         print("step:", i)
    #         print("action: ", saction * 2 + 1)
    #         state, reward, done, info = env.step(saction)
    #         print('state: ', state)
    #         print("reward: ", reward)
    #         state = np.uint(np.array(state))
    #
    #         t_reward += reward
    #         for n in range(4):
    #             Q[n, current[n], action[n]] += alpha * (
    #                         reward + discount * np.max(Q[n, state[n], :]) - Q[n, current[n], action[n]])
    #
    #     print("TotalReward:", t_reward)
    #
    #     rewards.append(t_reward)
    #     iterations.append(i)
    #
    # # Close environment
    # env.close()