#!/usr/bin/python3

import networkx as nx
from amplpy import AMPL, DataFrame
import argparse

import graphs.generate_service as gs

AMPLInfraTypes = ['APs', 'servers', 'mobiles']

# TODO: refactor the loading to a class to avoid parameter multiplication and/or global variables
infra_type_sets = {itype: [] for itype in AMPLInfraTypes}

def fill_service(ampl: AMPL, service: gs.ServiceGMLGraph) -> None:
    ampl.param['vertices'][service.name] = service.vnfs
    ampl.set['edges'][service.name] = [
        (service.nodes[c1][service.node_name_str],
            service[c2][service.node_name_str]) for c1,c2 in service.edges()]

    # set the CPU demands
    ampl.getParameter('demands').setValues({
        vnf: props[service.node_name_str]
        for vnf,props in service.nodes(data=True)
    })


def fill_infra(ampl: AMPL, infra: gs.InfrastructureGMLGraph) -> None:
    # Get the infrastructure nodes' names
    endpoint_names = [infra.nodes[id_][infra.node_name_str]
        for id_ in infra.endpoint_ids]
    access_point_names = [infra.nodes[id_][infra.node_name_str]
        for id_ in infra.access_point_ids]
    server_names = [infra.nodes[id_][infra.node_name_str]
        for id_ in infra.server_ids]
    mobile_names = [infra.nodes[id_][infra.node_name_str]
        for id_ in infra.mobile_ids]
    infra_names = endpoint_names + access_point_names + server_names +\
            mobile_names

    # All the infra graph vertices
    ampl.set['vertices'][infra.name] = infra_names
    ampl.set['APs'] = access_point_names
    ampl.set['servers'] = server_names
    ampl.set['mobiles'] = mobile_names

    # Infrastructure edges
    ampl.set['edges'][infra.name] = [(infra.nodes[c1][infra.node_name_str],
        infra.nodes[c2][infra.node_name_str]) for c1,c2 in infra.edges()]
    
    # set the CPU capabilities
    ampl.getParameter('resources').setValues({
        props[infra.node_name_str]: props[infra.infra_node_capacity_str]
        for node,props in infra.nodes(data=True)
    })

    # TODO: put the cost unit demand
    ampl.getParameter('cost_unit_demand').setValues({
        props[infra.node_name_str]: props[infra.infra_unit_cost_str]
        for node,props in infra.nodes(data=True)
    })

    # fill the battery probability constraints
    ampl.getParameter('max_used_battery').setValues({
        mobile_node: infra.full_loaded_battery_alive_prob
        for mobile_node in mobile_names
    })
    ampl.getParameter('min_used_battery').setValues({
        mobile_node: infra.unloaded_battery_alive_prob
        for mobile_node in mobile_names
    })


def fill_AP_coverage_probabilities(ampl: AMPL, interval_length: int) -> None:
    subintervals = [i for i in range(1, interval_length + 1)]
    APs = infra_type_sets[AMPLInfraTypes['APs']]
    df = DataFrame(('AP_name', 'subinterval'), 'prob')
    df.setValues({(AP, subint): 0.9 for AP in APs for subint in subintervals})
    ampl.param['prob_AP'].setValues(df)
    # TODO: refine this method to actually fill with the coverage prob


def get_complete_ampl_model_data(ampl_model_path, service : gs.ServiceGMLGraph, infra : gs.InfrastructureGMLGraph) -> AMPL:
    """
    Reads all service and infrastructure information to AMPL python data structure

    :param service:
    :param infra:
    :return:
    """
    ampl = AMPL()
    ampl.read(ampl_model_path)
    ampl.set['graph'] = [infra.name, service.name]
    ampl.param['infraGraph'] = infra.name
    ampl.param['serviceGraph'] = service.name

    # interval_length 
    ampl.param['interval_length'] = infra.time_interval_count

    fill_AP_coverage_probabilities(ampl, interval_length)

    # TODO @Jorge: continue from here on
        # fill the affinity variable

    fill_service(ampl, service)
    fill_infra(ampl, infra)
    return ampl


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Given arrivals CSV, ' + \
                                                 'it creates an AMPL .dat file')

    parser.add_argument('model', metavar='model', type=str,
                        help='Path to the AMPL model')
    parser.add_argument('arrivals', metavar='arrivals', type=str,
                        help='Path to the arrivals CSV file')
    parser.add_argument('out', metavar='out', type=str,
                        help='Path to the output where .dat is created')

    args = parser.parse_args()

    # Fill our model data
    ampl = AMPL()
    ampl.read(args.model)

    fill_service(ampl, service)
    fill_infra(ampl, infra)
    fill_AP_coverage_probabilities(ampl, args.interval_length)
    ampl.exportData(datfile=args.out)

