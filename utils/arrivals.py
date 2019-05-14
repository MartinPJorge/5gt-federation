import numpy as np

if __name__ == '__main__':
    small_arrivals = []
    big_arrivals = []

    # fill small and big arrivals
    # two independent random variables
    n = 10
    frequency = 2
    small_arrivals = np.random.poisson(n, frequency) # 2 secs frequency, 10 random
    #poiss
    big_arrivals = np.random.poisson(n,frequency)

    
    #
    small_lifetime = []
    big_lifetime = []

    # fill lifetimes for small and big
    mean = 3
    small_lifetime = np.random.exponential(n,mean) # a mean of 3h lifetime
    big_lifetime = np.random.exponential(n, mean*3)

    small_network_service_resources = []
    big_network_service_resources = []
    for i in range(10):
        small_network_service_resources.append({
            'mem': np.random.normal(4, 1), # 4gb memoru
            'cpu': 30,
            'disk':5
        })
        big_network_service_resources.append({
            'mem': np.random.normal(10, 1), # 4gb memoru
            'cpu': 30,
            'disk': 1
        })

    print(small_arrivals)
    print(small_lifetime)
    print(small_network_service_resources)

    print(big_arrivals)
    print(big_lifetime)
    print(big_network_service_resources)


