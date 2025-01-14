"""
Utility and help functions
Note that all function until geodesic distance are from or adapted from
https://github.com/schmitt-hub/preferential_access_and_fairness_in_waste_management
"""

import os
from math import exp
import numpy as np
import pandas as pd
from geopy import distance
import json
import bz2
import _pickle as cPickle
import random
from typing import List
from pathlib import Path

# functions for computing travel probabilities
def f_urban(d):
    if d < 5:
        return exp(-0.2550891696011455 * d ** 0.8674531576586394)
    else:
        return 4.639450774188538 * exp(-1.4989521421856289 * d ** 0.3288777336829004)


def f_rural(d):
    if d < 5:
        return exp(-0.24990116894290326 * d ** 0.8201058149904008)
    else:
        return 1.6114912595353221 * exp(-0.6887217475464711 * d ** 0.43652329253292316)


def create_travel_dict(users_and_facs_df, users, facs, output_filename = 'travel_dict'):
    """
    Create the travel dictionary that specifies the probabilities for each travel combination
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :return: travel dictionary that specifies the probabilities for each travel combination
    """
    travel_dict = {i: {} for i in users}
    for i in users:
        print('user', i)
        regiotype = users_and_facs_df.at[i, 'regional spatial type']
        lat_1 = users_and_facs_df.at[i, 'centroid_lat']
        lon_1 = users_and_facs_df.at[i, 'centroid_lon']
        for j in facs:
            lat_2 = users_and_facs_df.at[j, 'rc_centroid_lat']
            lon_2 = users_and_facs_df.at[j, 'rc_centroid_lon']
            dist = distance.distance((lat_1, lon_1), (lat_2, lon_2)).km
            if regiotype == "urban":
                travel_dict[i][j] = f_urban(dist)
            else:
                travel_dict[i][j] = f_rural(dist)
    #with open(output_filename + '.json', 'w') as outfile:
    #    json.dump(travel_dict, outfile)
    save_travel_dict(travel_dict, travel_dict_filename=output_filename+ ".json.pbz2")


# functions for loading and saving data files

def save_travel_dict(travel_dict, travel_dict_filename, abs_path=None):
    if not abs_path:
        abs_path = os.getcwd() + "/data"
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)
    with bz2.BZ2File(Path(abs_path + "/" + travel_dict_filename), "w") as f:
        cPickle.dump(travel_dict, f)


def load_users_and_facs(users_and_facs_filename="users_and_facilities.csv", abs_path=None):
    if not abs_path:
        abs_path = os.getcwd() + "/data"
    users_and_facs_df = pd.read_csv(Path(abs_path + "/" + users_and_facs_filename))
    return users_and_facs_df


def load_travel_dict(travel_dict_filename="travel_dict.json.pbz2", abs_path=None):
    if not abs_path:
        abs_path = os.getcwd() + "/data"
    data = bz2.BZ2File(Path(abs_path + "/" + travel_dict_filename), 'rb')
    travel_dict = cPickle.load(data)
    travel_dict = {int(i): {int(j): travel_dict[i][j] for j in travel_dict[i]} for i in travel_dict}
    return travel_dict

def load_facility_distance_dict(facility_distance_dict_filename = "facility_distance_dict.json.pbz2", abs_path=None):
    if not abs_path:
        abs_path = os.getcwd() + "/data"
    data = bz2.BZ2File(Path(abs_path + "/" + facility_distance_dict_filename), 'rb')
    travel_dict = cPickle.load(data)
    travel_dict = {int(i): {int(j): travel_dict[i][j] for j in travel_dict[i]} for i in travel_dict}
    return travel_dict

def load_input_data(
    users_and_facilities_filename="users_and_facilities.csv",
    travel_dict_filename="travel_dict.json.pbz2",
    abs_path=None
):
    users_and_facs_df = load_users_and_facs(users_and_facilities_filename, abs_path)
    travel_dict = load_travel_dict(travel_dict_filename, abs_path)
    return users_and_facs_df, travel_dict

def write_results_list(results, output_filename, output_abs_path = None):
    # save the table
    if not output_abs_path:
        output_abs_path = os.getcwd() + "/own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    with open(Path(output_abs_path + "/" + output_filename), 'w') as f:
        json.dump(results, f)

def read_results_list(filename, abs_path = None):
    """
    Read a results list and fix keys being string instead of int
    :param filename: filename, if no abs_path given in own_results folder
    :return: results list with fixed keys being int
    """
    if not abs_path:
        abs_path = os.getcwd() + "/own_results"
    with open(Path(abs_path+"/"+filename), 'r') as infile:
        results_list = json.load(infile)
    for results in results_list:
        results['solution_details']['assignment'] = {int(i): j for (i, j) in
                                                     results['solution_details']['assignment'].items()}
    return results_list

def geodesic_distance(users_and_facs_df, i, j):
    """
    Compute the geodesic distance from user i to facility j
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param i: index of the user
    :param j: index of the facility
    :return: distance in kilometers from user i to facility j
    """
    return distance.distance(
        (users_and_facs_df.at[i, 'centroid_lat'], users_and_facs_df.at[i, 'centroid_lon']),
        (users_and_facs_df.at[j, 'rc_centroid_lat'],
         users_and_facs_df.at[j, 'rc_centroid_lon'])).km


def capacity_constraints_slack(users_and_facs_df, travel_dict,cap_dict, facs, assignment):
    """
    Compute how much capacity is left for each facility given an assignment.
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param cap_dict: dictionary of capacities of the facilities
    :param facs: list of the facilities used in the instance
    :param assignment: dictionary of key user, value facility
    :return: return a dictionary of slack for each facility, 
        array of facilities who do not have their capacity constraint satisfied,
        dictionary with key facilities and value list of users assigned to that facility in assignment
    """
    slacks = cap_dict.copy()
    assigned_users = {j: [] for j in facs}
    for i, j in assignment.items():
        assigned_users[j].append(i)
        slacks[j] = slacks[j] - users_and_facs_df.at[i, 'population'] * travel_dict[i][j]
    not_satisfed = [j for j in facs if slacks[j] < 0]
    return slacks, not_satisfed, assigned_users


def check_valid_solution(users_and_facs_df, travel_dict, facs, open_facs, users, assignment,
                          budget_factor, cap_factor = 1.5):
    """"
    Check if assignment is valid solution to a BFLP / BUAP instance
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param facs: list of the open facilities used in the instance
    :param users: list of users
    :param assignment: dictionary of key user, value facility
    :param budget_factor: proportion of facilities that should be open at a maximum
    :param cap_factor: factor by which all capacities are scaled
    :return: true if assignment is valid, false otherwise.
    """
    # check each user assigned to a facility
    if(not all(i in assignment.keys() for i in users)):
        print("Some user is not assigned a facility")
        return False
    # check only open facilities used 
    if(not all(j in open_facs for j in assignment.values())):
        print("Some user is assigned to a closed facility")
        return False
    # check capacity constraints satisfied
    slacks = {j: cap_factor * users_and_facs_df.at[j, 'capacity'] for j in facs}
    for i, j in assignment.items():
        slacks[j] = slacks[j] - users_and_facs_df.at[i, 'population'] * travel_dict[i][j]
    not_satisfed = [j for j in facs if slacks[j] < 0]
    if len(not_satisfed) > 0:
        print("Capacity constraints not satisfied")
        print(not_satisfed)
        return False
    # check correct number of open facilities
    if len(open_facs) > round(budget_factor*len(facs)):
        print("Too many facilities open ")
        return False
    objective = sum(cap_factor*users_and_facs_df.at[j, "capacity"]*(slacks[j]/(cap_factor*users_and_facs_df.at[j, "capacity"]))**2 for j in facs)
    print("Objective: " + str(objective))
    return True


def check_valid_solution_with_unassigned(users_and_facs_df, travel_dict, facs, open_facs, users,
                                          assignment, budget_factor, cap_factor = 1.5):
    """"
    Check if assignment is valid solution, apart from allowing some users being unassigned.
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param facs: list of the open facilities used in the instance
    :param users: list of users
    :param assignment: dictionary of key user, value facility
    :param budget_factor: proportion of open facilities
    :param cap_factor: factor by which all capacities are scaled
    :return: true if assignment is valid, false otherwise
    """
    # check each user assigned to a facility
    if(not all(i in assignment.keys() for i in users)):
        print("Some user is not assigned a facility")
        return False
    # check only open facilities used 
    if(not all(j in open_facs or j=="unassigned" for j in assignment.values())):
        print("Some user is assigned to a closed facility")
        return False
    # check capacity constraints satisfied
    slacks = {j: cap_factor * users_and_facs_df.at[j, 'capacity'] for j in facs}
    for i, j in assignment.items():
        if j != "unassigned":
            slacks[j] = slacks[j] - users_and_facs_df.at[i, 'population'] * travel_dict[i][j]
    not_satisfed = [j for j in facs if slacks[j] < 0]
    if len(not_satisfed) > 0:
        print("Capacity constraints not satisfied")
        print(not_satisfed)
        return False
    # check correct number of open facilities
    if len(open_facs) > round(budget_factor*len(facs)):
        print("Too many facilities open ")
        return False
    objective = sum(cap_factor*users_and_facs_df.at[j, "capacity"]*(slacks[j]/(cap_factor*users_and_facs_df.at[j, "capacity"]))**2 for j in facs)
    print("Objective: " + str(objective))
    return True

def generate_data(n_users: int, p_fac: float, output_file_name_prefix = "large_random_instance"):
    '''
    Generates random BFLP instance and saves a users_and_facs_df and a dictionary of P_ij (travel_dict) for this instance.
    :param: n_users: number of users of generated instance
    :param: p_fac: probability each ZIP code has a facility
    :param: output_file_name_prefix: First part of the name for the files in which the instance is saved
    '''
    # Longitude and latitude range for Germany
    #lon_min, lon_max = 5.8663153, 15.0419319
    #lat_min, lat_max = 47.2701114, 55.099161
    # Longitude and latitude range for Bavaria
    #lon_min, lon_max = 9.01111334936577, 13.778347475506
    #lat_min, lat_max = 47.3625560497034, 50.5248689558092

    # Quarter of Bavaria size
    lon_min, lon_max = 9.0, 11.5
    lat_min, lat_max = 47.3, 49.0

    data = []
    users = list(range(n_users))
    facs = []
    for i in range(n_users):
        zip = i
        lon = random.uniform(lon_min, lon_max)
        lat = random.uniform(lat_min, lat_max)
        population = random.randint(0, 20000)
        capacity = random.randint(0, 80000) if random.random() < p_fac else 0
        if capacity > 0:
            facs.append(i)
        spatial_type = "rural" if random.random() < 0.5 else "urban"
        data.append({'zipcode': zip,'centroid_lat': lat,'centroid_lon': lon, 
                     'regional spatial type': spatial_type,'population': population, 
                     'capacity': capacity,'rc_centroid_lat': lat + random.gauss(0.0, 0.01),
                     'rc_centroid_lon': lon + random.gauss(0.0, 0.01)})
    users_and_facs_df = pd.DataFrame(data)
    users_and_facs_df.to_csv("data/"+ output_file_name_prefix+'_users_and_facs.csv', index=False)
    create_travel_dict(users_and_facs_df, users, facs, output_filename=output_file_name_prefix+'_travel_dict')

def create_users_facs_by_zip_code_division(users_and_facs_df, users,facs):
    """
    Splits up instance 1 into small instances
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :return: list of dictionaries of users and facs split by the first two digits of their ZIP codes
    """
    zip_codes_starts = []
    for i in users:
        if str(users_and_facs_df.at[i, "zipcode"])[0:2] not in zip_codes_starts:
            zip_codes_starts.append(str(users_and_facs_df.at[i, "zipcode"])[0:2])

    split_users = {start: [] for start in zip_codes_starts}
    split_facs = {start: [] for start in zip_codes_starts}
    for i in users:
        split_users[str(users_and_facs_df.at[i, "zipcode"])[0:2]].append(i)

    for j in facs:
        split_facs[str(users_and_facs_df.at[j, "zipcode"])[0:2]].append(j)
    return split_users, split_facs

def create_larger_sets_by_combining_zip_code_starts(users_and_facs_df, users, facs):
    """
    Splits up instance 1 into medium sized instances
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :return: list of dictionaries of users and facs split by the first two digits of their ZIP codes
        but combined to larger sets
    """
    split_users, split_facs = create_users_facs_by_zip_code_division(users_and_facs_df, users,facs)
    combined_zip_codes = {"63, 97, 96, 95" : [63, 97, 96, 95], "90, 91, 92": [90, 91, 92],
                            "93, 94": [93, 94], "80, 81, 82, 83, 84, 85" : [80, 81, 82, 83, 84, 85],
                            "86, 87,88, 89": [86, 87, 88, 89]}
    split_users_combined = {start: [] for start in combined_zip_codes.keys()}
    split_facs_combined = {start: [] for start in combined_zip_codes.keys()}
    for key, value in split_users.items():
        for key_combined,value_combined  in combined_zip_codes.items():
            if int(key) in value_combined:
                split_users_combined[key_combined].extend(value)
    for key, value in split_facs.items():
        for key_combined,value_combined  in combined_zip_codes.items():
            if int(key) in value_combined:
                split_facs_combined[key_combined].extend(value)
    return split_users_combined, split_facs_combined

def get_instance_2_users_facs(users_and_facs_df, users, facs):
    '''
    Creates and returns the user and facility sets for Instance 2 from Instance 1 input
    :param users_and_facs_df: dataframe of user and facility related input data
    :param users: set of users
    :param facs: set of facilities
    :return: the users and facilities of instance 2
    '''
    users_list, facs_list = create_larger_sets_by_combining_zip_code_starts(users_and_facs_df, users, facs)
    users_2 = users_list["90, 91, 92"]
    facs_2 =  facs_list["90, 91, 92"]
    return users_2, facs_2

def load_an_instance(instance_number, sufficient_cap = False):
    '''
    Helper function for getting all the data needed for a specific instance
    :param instance_number: the number of the instance; possible inputs are 1,2,3,4
    :param sufficient_cap: boolean indicating whether the version of the instance where
        sum_{i in I} U_i P_ij <= C_j should be loaded
    :return: users and facilities dataframe, dictionary of P_ij, list of users, list of facilities
    '''
    if instance_number != 1 and instance_number != 2 and instance_number != 3 and instance_number != 4:
        print("The instance number is not in the allowed range of 1,2,3 or 4.")
        print("This function may fail if you have not created an instance with your input instance number.")
    actual_instance_number = str(instance_number)
    if instance_number == 2:
        actual_instance_number = "1"
    
    suff_postfix = "suff_" if sufficient_cap else ""
    uaf_filename = "instance_" + actual_instance_number + "_" + suff_postfix + "users_and_facs.csv"
    td_filename = "instance_" + actual_instance_number + "_travel_dict.json.pbz2"

    users_and_facs_df, travel_dict = load_input_data(users_and_facilities_filename=uaf_filename,
                                                travel_dict_filename=td_filename)
    
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    if instance_number == 2:
        users, facs = get_instance_2_users_facs(users_and_facs_df, users, facs)

    return users_and_facs_df, travel_dict, users, facs
    
