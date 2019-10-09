import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import seaborn as sns



if __name__ == '__main__':
    q_combs = None
    with open('alpha-discount-combinations.json') as json_file:
        q_combs = json.load(json_file)


    # Parse alpha, discount, and profit values
    alphas = []
    discounts =  []
    profits = []
    for a_d in q_combs.keys():
        a_d_ = a_d[1:-1].split(',') # remove parenthesis
        alphas += [float(a_d_[0])]
        discounts += [float(a_d_[1])]
        profits += [q_combs[a_d]]
        
    alphas = np.array(alphas)
    discounts = np.array(discounts)
    profits = np.array(profits)
    df = pd.DataFrame.from_dict(np.array([alphas,discounts,profits]).T)
    df.columns = ['$\\alpha$','discount','profit']
    pivotted = df.pivot('discount','$\\alpha$','profit')

    sns.heatmap(pivotted, cmap='RdBu', linewidth=0.5)

    plt.show()

