import networkx as nx
import sys
import os
import json
import argparse
from hac import GreedyAgglomerativeClusterer

# Import NsGenerator based on relative path
absPath = os.path.abspath(os.path.dirname(__file__))
srcGen= '/'.join(absPath.split('/')[:-1]) + '/ns_generator/src'
sys.path.append(srcGen)
from vnfsMapping.NsGenerator import NSgenerator
from vnfsMapping.NS import NS


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Performs the clustering ' +\
            'of a specified network service graph.')
    parser.add_argument('vnfCSV', type=str,
                        help='path to the CSV with VNFs of te NS graph')
    parser.add_argument('vlCSV', type=str,
                        help='number of graphs to generate')
    parser.add_argument('out', type=str,
                        help='path to store the generated clusters')
    args = parser.parse_args()
    
    # Create the NS and set weights to be traffic
    readNS = NS.readCSV(vnfCSV = args.vnfCSV, vlCSV = args.vlCSV)
    nsG = readNS.getChain()
    weights = {}
    for (vnfA, vnfB, data) in nsG.edges(data=True):
        weights[vnfA, vnfB] = float(data['traffic'])
    nx.set_edge_attributes(nsG, 'weight', weights)

    # Perform the clustering
    clusterer = GreedyAgglomerativeClusterer()
    dendoVnf = clusterer.cluster(nsG)

    clustering = {}
    for n in range(2, len(nsG) + 1):
        clustering[n] = {}
        for vnf in nsG.nodes():
            cId = [i for i in range(n) if vnf in dendoVnf.clusters(n)[i]][0]
            if cId not in clustering[n]:
                clustering[n][cId] = []
            clustering[n][cId].append(vnf)


    print clustering



