'''
Functions for running the heuristics with multiple inputs (get_...) resulting in a json
and functions for converting these results into more easily readable tables (write_...)
'''
from BUAP_heuristics import *
from BFLP_heuristics import *
import math
from utils import *
from random import sample
from BUAP_MIP import *
import os
import time
from pathlib import Path

def get_results_close_greedy_final(users_and_facs_df, travel_dict, users, facs, 
                            budget_factors = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], 
                            cap_factor=1.5, threads=1, tolerance=0.0,
                            cutoff_localsearch = 0.2, num_consider_in_relaxation = 50, 
                            num_consider_per_iterations = [5,50], time_limit_relaxation=1000,
                            assignment_methods =["greedy_assign"],local_search_methods = ["None"], 
                            time_limit_while_localsearch = 1, num_fix_assignment_iterations = [100],
                            final_assignment_methods = ["relaxation_rounding"],output_filename = "test.json"):
    '''
    Use the final version of close greedy to get results for the input instances at different input parameters
        whenever a input to this function is a list, we consider all possible combinations of input paramters
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param budget_factors: list of proportions of facilities we want to remain open
    :param cap_factor: factor by which all capacities are scaled
    :param threads: number of threads used in the relaxation
    :param tolerance: tolerance on the optimality of the solution used in the relaxation
    :param cutoff_localsearch: parameter for BUAP localsearch for minimum preference required to reassign to the facility
    :param num_consider_in_relaxation: number of most preferred facilities to consider in relaxation,
        if -1 all facilities are considered in relaxation
    :param num_consider_per_iterations: list of number of facilities to consider closing in each iteration
    :param time_limit_relaxation: time limit in seconds for the optimization used in the relaxation
    :param assignment_methods: list of methods which is used to create the first assignment,
        options are "relaxation_rounding" or "greedy_assign"
    :param local_search_methods: list of BUAP local search method to run on the assignment resulting from assignment_method
        and final_assignment_method; options are "None", "local_random_reassign", "local_random_swap"
    :param: time_limit_while_localsearch: the time limit in seconds used in the while loop in the localsearch
    :param num_fix_assignment_iterations: list of number of iterations after which to recompute the assignment from scratch
        using assignment_method; no saving of model for relaxation rounding implemented so this is only for 
        when assignment_method is "greedy_assign"
    :param final_assignment_method: list of assignment method to run after open facilities decided,
       currently only support relaxation rounding
    :param output_filename: name of the output file
    :return: list of results
    '''
    results_list = []
    for n_c in num_consider_per_iterations:
        for assignment_method in assignment_methods:
            for local_search_method in local_search_methods:
                for final_assignment_method in final_assignment_methods:
                    for n_f in num_fix_assignment_iterations:
                        for budget_factor in budget_factors:
                            is_feasible, result = close_greedy_final(users_and_facs_df, travel_dict, users, facs,
                                                    budget_factor = budget_factor, starting_open_facs = None,cap_factor=cap_factor,
                                                    threads = threads, tolerance = tolerance,cutoff_localsearch = cutoff_localsearch,
                                                    num_consider_in_relaxation = num_consider_in_relaxation, 
                                                    num_consider_per_iteration = n_c,time_limit_relaxation = time_limit_relaxation,
                                                    assignment_method = assignment_method,local_search_method = local_search_method, 
                                                    time_limit_while_localsearch = time_limit_while_localsearch,
                                                    num_fix_assignment_iteration = n_f, final_assignment_method =final_assignment_method,
                                                    output_filename = "test.json", write_to_file = False)
                            results_list.append(result)

    write_results_list(results_list, output_filename)
    return results_list


def get_results_open_greedy(users_and_facs_df, travel_dict, users, facs, 
                            budget_factors = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], 
                            cap_factor=1.5, threads=1, tolerance=0.0,
                            cutoff_localsearch = 0.2, num_consider_in_relaxation = 50, 
                            num_consider_per_iterations = [5,50], time_limit_relaxation=1000,
                            assignment_methods =["greedy_assign"],local_search_methods = ["None"], 
                            time_limit_while_localsearch = 1, num_fix_assignment_iterations = [100],
                            depths_greedy_reassign = [1,2],
                            final_assignment_methods = ["relaxation_rounding"],output_filename = "test.json"):
    '''
    Use the open greedy to get results for the input instances at different input parameters
        whenever a input to this function is a list, we consider all possible combinations of input paramters
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param budget_factors: list of proportions of facilities we want to remain open
    :param cap_factor: factor by which all capacities are scaled
    :param threads: number of threads used in the relaxation
    :param tolerance: tolerance on the optimality of the solution used in the relaxation
    :param cutoff_localsearch: parameter for BUAP localsearch for minimum preference required to reassign to the facility
    :param num_consider_in_relaxation: number of most preferred facilities to consider in relaxation,
        if -1 all facilities are considered in relaxation
    :param num_consider_per_iterations: list of number of facilities to consider opening in each iteration
    :param time_limit_relaxation: time limit in seconds for the optimization used in the relaxation
    :param assignment_methods: list of methods which is used to create the first assignment,
        options are "relaxation_rounding" or "greedy_assign"
    :param local_search_methods: list of BUAP local search method to run on the assignment resulting from assignment_method
        and final_assignment_method; options are "None", "local_random_reassign", "local_random_swap"
    :param: time_limit_while_localsearch: the time limit in seconds used in the while loop in the localsearch
    :param num_fix_assignment_iterations: list of number of iterations after which to recompute the assignment from scratch
        using assignment_method; no saving of model for relaxation rounding implemented so this is only for 
        when assignment_method is "greedy_assign"
    :param depths_greedy_reassign: list of depths of the greedy reassignment within the greedy reassign open algorithm
    :param final_assignment_method: list of assignment method to run after open facilities decided,
       currently only support relaxation rounding
    :param output_filename: name of the output file
    :return: list of results
    '''
    results_list = []
    for n_c in num_consider_per_iterations:
        for n_f in num_fix_assignment_iterations:
            for d in depths_greedy_reassign:
                for assignment_method in assignment_methods:
                    for local_search_method in local_search_methods:
                        for final_assignment_method in final_assignment_methods:
                            for budget_factor in budget_factors:
                                is_feasible, result = open_greedy(users_and_facs_df, travel_dict, users, facs,
                                                        budget_factor = budget_factor, cap_factor=cap_factor, threads=threads,
                                                        tolerance=tolerance,cutoff_localsearch = cutoff_localsearch,
                                                        num_consider_in_relaxation = num_consider_in_relaxation,
                                                        num_consider_per_iteration = n_c, depth_greedy_reassignment = d,
                                                        time_limit_relaxation=time_limit_relaxation, assignment_method =assignment_method,
                                                        local_search_method = local_search_method, 
                                                        time_limit_while_localsearch = time_limit_while_localsearch,
                                                        num_fix_assignment_iteration = n_f,
                                                        final_assignment_method = final_assignment_method, 
                                                        output_filename = "test.json", write_to_file = False)
                                results_list.append(result)

    write_results_list(results_list, output_filename)
    return results_list

def get_results_Schmitt_Singh(users_and_facs_df, travel_dict, users, facs, lb = 0.0,
                                budget_factors = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], cap_factor = 1.5,
                                turnover_factor = 0.02, tolerance = 5e-3, time_limit = 20000, iteration_limit = 100,
                                output_filename = "test.json"):
    '''
    Runs the Schmitt Singh heuristics for all the input budget factors
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param lb: a lower bound on the optimal objective value
    :param budget_factor: ratio of facilities that are allowed to be opened
    :param cap_factor: factor by which all capacities are scaled
    :param turnover_factor: maximal fraction of facilities that are swapped in each iteration
    :param tolerance: tolerance on the optimality of the solution
    :param time_limit: time limit in seconds
    :param iteration_limit: limit f
    :return: list of results of the Schmitt Singh heuristic
    '''
    results_list = []
    for budget_factor in budget_factors:
        is_feasible, results = Schmitt_Singh_localsearch(users_and_facs_df, travel_dict, users, facs, lb = lb,
                                                            budget_factor = budget_factor, cap_factor = cap_factor,
                                                            turnover_factor = turnover_factor, tolerance = tolerance,
                                                            time_limit = time_limit, iteration_limit = iteration_limit)
        results_list.append(results)
    write_results_list(results_list, output_filename)
    return results_list

    


def write_results_close_greedy_table(results_list_filename, output_filename, BLFP_MIP_filename, output_abs_path = None):
    '''
    Given a json format of close greedy results, saves the relevant results as an excel table.
    :param results_list_filename: file name of the close greedy results list, expected to be in own_results folder
    :param output_filename: file name of the output excel file
    :param BFLP_MIP_filename: file name of the corresponding results for the BFLP MIP in order to compare the results
        of the heuristics to what the MIP achieves, expected to be in own_results folder
    :param output_abs_path: file path for the output file, if not given the resulting excel file will be put
        into the own_results folder
    '''
    results_list = read_results_list(results_list_filename)
    BFLP_MIP_list = read_results_list(BLFP_MIP_filename)
    dictionary_BFLP_MIP = {}
    for result in BFLP_MIP_list:
        dictionary_BFLP_MIP[result["model_details"]["budget_factor"]] = result["solution_details"]["objective_value"]
    assignment_method = []
    local_search_method = []
    final_assignment_method = []
    num_consider_per_iterations = []
    num_fix_assignments_iterations = []
    budget_factors = []
    running_time = []
    objective_value =  []
    MIP_objectives = []
    Delta_MIPs = []
    for result in results_list:
        budget_factor = result["model_details"]["budget_factor"]
        assignment_method.append(result["algorithm_details"]["assignment_method"])
        local_search_method.append(result["algorithm_details"]["local_search_assignment_method"])
        final_assignment_method.append(result["algorithm_details"]["final_assignment_method"])
        budget_factors.append(budget_factor)
        running_time.append(result["solution_details"]["solving_time"])
        objective_value.append(result["solution_details"]["objective_value"])
        num_consider_per_iterations.append(result["algorithm_details"]["num_consider_per_iteration"])
        num_fix_assignments_iterations.append(result["algorithm_details"]["num_fix_assignment_iteration"])
        closest_budget_factor = 0.0
        closest_budget_factor_distance = 1.0
        for b in dictionary_BFLP_MIP.keys():
            if abs(budget_factor - b) < closest_budget_factor_distance:
                closest_budget_factor_distance = abs(budget_factor - b)
                closest_budget_factor = b
        if closest_budget_factor_distance > 1e-4:
            print("Failed to find MIP results. Creating table failed.")
            return
        MIP_objective = dictionary_BFLP_MIP[closest_budget_factor]
        MIP_objectives.append(MIP_objective)
        Delta_MIPs.append(100*(MIP_objective-result["solution_details"]["objective_value"])/ MIP_objective)

    data = {"Budget factor": budget_factors, "Number of facilities considered per iteration (n_c)": num_consider_per_iterations, 
            "Number of iterations after which to fix assignment (n_f)":num_fix_assignments_iterations,
            "Assignment method": assignment_method, 
            "Final assignment method": final_assignment_method, "Local search assignment method": local_search_method,
            "Run time [s]": running_time,"Objective value close greedy": objective_value, "Objective value MIP": MIP_objectives,
            "Delta_MIP [%]": Delta_MIPs}
    df = pd.DataFrame(data=data)

    # save the table
    if not output_abs_path:
        output_abs_path = os.getcwd() + "/own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    with pd.ExcelWriter(Path(output_abs_path + "/" + output_filename)) as writer:
        df.to_excel(writer, index=False)#


def write_results_open_greedy_table(results_list_filename, output_filename, BLFP_MIP_filename, output_abs_path = None):
    '''
    Given a json format of open greedy results, saves the relevant results as an excel table.
    :param results_list_filename: file name of the open greedy results list, expected to be in own_results folder
    :param output_filename: file name of the output excel file
    :param BFLP_MIP_filename: file name of the corresponding results for the BFLP MIP in order to compare the results
        of the heuristics to what the MIP achieves, expected to be in own_results folder
    :param output_abs_path: file path for the output file, if not given the resulting excel file will be put
        into the own_results folder
    '''
    results_list = read_results_list(results_list_filename)
    BFLP_MIP_list = read_results_list(BLFP_MIP_filename)
    dictionary_BFLP_MIP = {}
    for result in BFLP_MIP_list:
        dictionary_BFLP_MIP[result["model_details"]["budget_factor"]] = result["solution_details"]["objective_value"]
    assignment_method = []
    local_search_method = []
    final_assignment_method = []
    num_consider_per_iterations = []
    num_fix_assignments_iterations = []
    depths = []
    budget_factors = []
    running_time = []
    objective_value =  []
    MIP_objectives = []
    Delta_MIPs = []
    for result in results_list:
        budget_factor = result["model_details"]["budget_factor"]
        assignment_method.append(result["algorithm_details"]["assignment_method"])
        local_search_method.append(result["algorithm_details"]["local_search_assignment_method"])
        final_assignment_method.append(result["algorithm_details"]["final_assignment_method"])
        depths.append(result["algorithm_details"]["depth_greedy_reassignment"])
        budget_factors.append(budget_factor)
        running_time.append(result["solution_details"]["solving_time"])
        objective_value.append(result["solution_details"]["objective_value"])
        num_consider_per_iterations.append(result["algorithm_details"]["num_consider_per_iteration"])
        num_fix_assignments_iterations.append(result["algorithm_details"]["num_fix_assignment_iteration"])
        closest_budget_factor = 0.0
        closest_budget_factor_distance = 1.0
        for b in dictionary_BFLP_MIP.keys():
            if abs(budget_factor - b) < closest_budget_factor_distance:
                closest_budget_factor_distance = abs(budget_factor - b)
                closest_budget_factor = b
        if closest_budget_factor_distance > 1e-4:
            print("Failed to find MIP results. Creating table failed.")
            return
        MIP_objective = dictionary_BFLP_MIP[closest_budget_factor]
        MIP_objectives.append(MIP_objective)
        Delta_MIPs.append(100*(MIP_objective-result["solution_details"]["objective_value"])/ MIP_objective)

    data = {"Budget factor": budget_factors, "Number of facilities considered per iteration (n_c)": num_consider_per_iterations,
            "Depth greedy reassignment (d)":depths,"Number of iterations after which to fix assignment (n_f)":num_fix_assignments_iterations,
            "Assignment method": assignment_method, 
            "Final assignment method": final_assignment_method, "Local search assignment method": local_search_method,
            "Run time [s]": running_time,"Objective value open greedy": objective_value, "Objective value MIP": MIP_objectives,
            "Delta_MIP [%]": Delta_MIPs}
    df = pd.DataFrame(data=data)

    # save the table
    if not output_abs_path:
        output_abs_path = os.getcwd() + "/own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    with pd.ExcelWriter(Path(output_abs_path + "/" + output_filename)) as writer:
        df.to_excel(writer, index=False)

def write_results_Schmitt_Singh_table(results_list_filename, output_filename, BLFP_MIP_filename, output_abs_path = None):
    '''
    Given a json format of the Schmitt Singh heuristics results, saves the relevant results as an excel table.
    :param results_list_filename: file name of the Schmitt Singh heuristic results list, expected to be in own_results folder
    :param output_filename: file name of the output excel file
    :param BFLP_MIP_filename: file name of the corresponding results for the BFLP MIP in order to compare the results
        of the heuristics to what the MIP achieves, expected to be in own_results folder
    :param output_abs_path: file path for the output file, if not given the resulting excel file will be put
        into the own_results folder
    '''
    results_list = read_results_list(results_list_filename)
    BFLP_MIP_list = read_results_list(BLFP_MIP_filename)
    dictionary_BFLP_MIP = {}
    for result in BFLP_MIP_list:
        dictionary_BFLP_MIP[result["model_details"]["budget_factor"]] = result["solution_details"]["objective_value"]
    assignment_method = []
    budget_factors = []
    running_time = []
    objective_value =  []
    iteration_limits = []
    turnover_factors = []
    MIP_objectives = []
    Delta_MIPs = []
    for result in results_list:
        budget_factor = result["model_details"]["budget_factor"]
        assignment_method.append("greedy_assign")
        budget_factors.append(budget_factor)
        running_time.append(result["solution_details"]["solving_time"])
        objective_value.append(result["solution_details"]["objective_value"])
        iteration_limits.append(result["model_details"]["iteration_limit"])
        turnover_factors.append(result["model_details"]["turnover_factor"])
        closest_budget_factor = 0.0
        closest_budget_factor_distance = 1.0
        for b in dictionary_BFLP_MIP.keys():
            if abs(budget_factor - b) < closest_budget_factor_distance:
                closest_budget_factor_distance = abs(budget_factor - b)
                closest_budget_factor = b
        if closest_budget_factor_distance > 1e-4:
            print("Failed to find MIP results. Creating table failed.")
            return
        MIP_objective = dictionary_BFLP_MIP[closest_budget_factor]
        MIP_objectives.append(MIP_objective)
        Delta_MIPs.append(100*(MIP_objective-result["solution_details"]["objective_value"])/ MIP_objective)

    data = {"Budget factor": budget_factors,"Assignment method": assignment_method,
            "Run time [s]": running_time,"Objective value Schmitt Singh heuristic": objective_value, "Objective value MIP": MIP_objectives,
            "Delta_MIP [%]": Delta_MIPs}
    df = pd.DataFrame(data=data)

    # save the table
    if not output_abs_path:
        output_abs_path = os.getcwd() + "/own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    with pd.ExcelWriter(output_abs_path + "/" + output_filename) as writer:
        df.to_excel(writer, index=False)

def get_results_all_BUAP_methods(users_and_facs_df, travel_dict, users, facs, open_facs,
                        cap_factor=1.5,define_u = True,cutoff_relaxation = 0.0,
                         num_consider_relaxation = 20,
                        cutoff_localsearch = 0.2, time_limit_while_localsearch = 1,
                        output_filename = "BUAP_heuristics.json"):
    '''
    Functions for running all the BUAP methods on a single instance.
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param open_facs: list of the open facilities for the BUAP instance
    :param cap_factor: factor by which all capacities are scaled
    :param define_u: boolean indicating whether to explicitely define u in the model
    :param cutoff_relaxation: preference cutoff used when craeting the relaxation model
        only used if num_consider_relaxation is -1
    :param num_consider_relaxation: number of closest facilities to consider for a user (alternative to cutoff value)
        same as n_r in paper; setting this to -1 means we use a cutoff instead
    :param cutoff_localsearch: minimum preference to consider reassinging a user in local search approaches
    :param time_limit_while_localsearch: time limit of the while loop of the local search approaches
    :param output_filename: name of the file where all the resuls of the heuristics are saved.
    :return: list of results of all the heuristics on the input instance
    '''
    cap_dict = {j: cap_factor * users_and_facs_df.at[j, 'capacity'] for j in facs}
    nr_of_users = len(users)
    nr_of_facs = len(open_facs)
    budget_factor = len(open_facs) / len(facs)
    # create dictionary of how many users are expected to travel to each facility
    # using numpy here since it is 10 times faster
    travel_matrix = np.zeros((nr_of_users, nr_of_facs))
    for i in range(nr_of_users):
        for j in range(nr_of_facs):
            travel_matrix[i][j] = travel_dict[users[i]][open_facs[j]]
    exp_travelers_matrix = np.array(users_and_facs_df.loc[users]['population']).reshape(-1, 1) * travel_matrix
    exp_travelers_dict = {users[i]: {open_facs[j]: exp_travelers_matrix[i][j] for j in range(nr_of_facs)}
                            for i in range(nr_of_users)}
    results = []
    # relaxation rounding
    is_feasible, result_relaxation_rounding = relaxation_rounding(users_and_facs_df, travel_dict, users, facs,
                        cap_factor=cap_factor, threads=1, tolerance=0.0,cutoff = cutoff_relaxation,
                        time_limit=1000, open_facs = open_facs, budget_factor = budget_factor, 
                        define_u = define_u, num_consider = num_consider_relaxation)
    result_relaxation_rounding["model_details"]["local_search"] = "None"
    results.append(result_relaxation_rounding)
    # reassign users local search based on relaxation rounding
    start_time = time.time()
    is_feasible, result_relaxation_localsearch_short = local_search_reassign(users_and_facs_df, travel_dict, users, facs,
                                                            open_facs, result_relaxation_rounding["solution_details"]["assignment"].copy(),
                                                            exp_travelers_dict, cap_dict, cutoff = cutoff_localsearch,
                                                            time_limit_while = time_limit_while_localsearch) 
    end_time = time.time()
    if not is_feasible:
        print("local search failed")
        return results
    result_relaxation_localsearch = {"solution_details":
                    {"assignment": result_relaxation_localsearch_short["assignment"],  "objective_value": result_relaxation_localsearch_short["obj"],
                     "solving_time": end_time - start_time},
                "model_details":
                    {"users": users,"open_facs": open_facs, "facs": facs, "cap_factor": cap_factor, "budget_factor": budget_factor, "cutoff": cutoff_localsearch,
                     "heuristic": "Localsearch reassign with relaxation start","time_limit_while": time_limit_while_localsearch
                }
            }
    results.append(result_relaxation_localsearch)
    # swap users local search based on relaxation rounding
    start_time = time.time()
    is_feasible, result_relaxation_localsearch_swap_short = local_search_swap(users_and_facs_df, travel_dict, users, facs,
                                                                open_facs, 
                                                                result_relaxation_rounding["solution_details"]["assignment"].copy(),
                                                                exp_travelers_dict, cap_dict, cutoff = cutoff_localsearch,
                                                                time_limit_while = time_limit_while_localsearch) 
    end_time = time.time()
    if not is_feasible:
        print("local search failed")
        return results
    result_relaxation_localsearch = {"solution_details":
                    {"assignment": result_relaxation_localsearch_swap_short["assignment"], 
                     "objective_value": result_relaxation_localsearch_swap_short["obj"],
                     "solving_time": end_time - start_time},
                "model_details":
                    {"users": users,"open_facs": open_facs, "facs": facs, "cap_factor": cap_factor,
                     "budget_factor": budget_factor, "cutoff": cutoff_localsearch,
                    "heuristic": "Localsearch swap with relaxation start",
                    "time_limit_while": time_limit_while_localsearch
                }
            }
    results.append(result_relaxation_localsearch)

    # greedy assignment
    start_time = time.time()
    is_feasible, result_greedy_short = greedily_assign_users(users_and_facs_df, travel_dict, users, facs, open_facs, exp_travelers_dict, cap_dict, assignment = None, fac_to_close = None)
    end_time = time.time()
    result_greedy = {"solution_details":
                    {"assignment": result_greedy_short["assignment"],  "objective_value": result_greedy_short["obj"],
                     "solving_time": end_time - start_time},
                "model_details":
                    {"users": users,"open_facs": open_facs, "facs": facs, "cap_factor": cap_factor, "budget_factor": budget_factor,
                    "heuristic": "Greedy assignment"
                }
            }
    results.append(result_greedy)
    # reassign users local search based on greedy
    start_time = time.time()
    is_feasible, result_greedy_localsearch_short = local_search_reassign(users_and_facs_df, travel_dict, users, facs,
                                                                                        open_facs, result_greedy_short["assignment"].copy(),
                                                                                          exp_travelers_dict, cap_dict, cutoff = cutoff_localsearch,
                                                                                            time_limit_while = time_limit_while_localsearch) 
    end_time = time.time()
    if not is_feasible:
        print("local search failed")
        return results
    result_greedy_localsearch = {"solution_details":
                    {"assignment": result_greedy_localsearch_short["assignment"],  "objective_value": result_greedy_localsearch_short["obj"],
                     "solving_time": end_time - start_time},
                "model_details":
                    {"users": users,"open_facs": open_facs, "facs": facs, "cap_factor": cap_factor, "budget_factor": budget_factor, "cutoff": cutoff_localsearch,
                    "heuristic": "Localsearch reassign with greedy start","time_limit_while": time_limit_while_localsearch
                }
            }
    results.append(result_greedy_localsearch)
    # swap users local search based on greedy
    start_time = time.time()
    is_feasible, result_greedy_localsearch_swap_short = local_search_swap(users_and_facs_df, travel_dict, users, facs,
                                                                                        open_facs, result_greedy_short["assignment"].copy(),
                                                                                          exp_travelers_dict, cap_dict, cutoff = cutoff_localsearch,
                                                                                            time_limit_while = time_limit_while_localsearch) 
    end_time = time.time()
    if not is_feasible:
        print("local search failed")
        return results
    result_greedy_localsearch = {"solution_details":
                    {"assignment": result_greedy_localsearch_swap_short["assignment"],  "objective_value": result_greedy_localsearch_swap_short["obj"],
                     "solving_time": end_time - start_time},
                "model_details":
                    {"users": users,"open_facs": open_facs, "facs": facs, "cap_factor": cap_factor, "budget_factor": budget_factor, "cutoff": cutoff_localsearch,
                    "heuristic": "Localsearch swap with greedy start","time_limit_while": time_limit_while_localsearch
                }
            }
    results.append(result_greedy_localsearch)
    write_results_list(results, output_filename)
    return results

def write_BUAP_results_table(results_list_filename, MIP_objective, output_filename, output_abs_path = None):
    '''
    Given a json format of the BUAP heuristics results, saves the relevant results as an excel table.
    :param results_list_filename: file name of the BUAP heuristic results list, expected to be in own_results folder
    :param output_filename: file name of the output excel file
    :param MIP_objective: objective value of the MIP in order to compare the heuristic performance with this
    :param output_abs_path: file path for the output file, if not given the resulting excel file will be put
        into the own_results folder
    '''
    results_list = read_results_list(results_list_filename)
    method = []
    objective = []
    runtime = []
    MIP_objectives = []
    Delta_MIPs = []
    for result in results_list:
        method.append(result["model_details"]["heuristic"])
        objective.append(result["solution_details"]["objective_value"])
        runtime.append(result["solution_details"]["solving_time"])
        MIP_objectives.append(MIP_objective)
        Delta_MIPs.append(100*(MIP_objective-result["solution_details"]["objective_value"])/MIP_objective)
    data = {"BUAP method": method,"Run time [s]": runtime, "Objective value BUAP heuristic": objective,
             "MIP objective value": MIP_objectives, "Delta_MIP [%]": Delta_MIPs}
    df = pd.DataFrame(data=data)

    # save the table
    if not output_abs_path:
        output_abs_path = os.getcwd() + "/own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    with pd.ExcelWriter(Path(output_abs_path + "/" + output_filename)) as writer:
        df.to_excel(writer, index=False)   


def get_results_localsearch(users_and_facs_df, travel_dict, input_filename, output_filename,
                            iteration_limits = [50, 100, 200], num_consider_per_iterations = [10,50,100],
                            depths_greedy_reassign = [1,2], time_limit_while_loop = 3600, assignment_method="greedy_assign",
                            final_assignment_method = "relaxation_rounding", 
                            localsearch_method = "local_random_reassign", num_consider_in_relaxation = 50, 
                            cutoff_BUAP_localsearch = 0.2, time_limit_while_BUAP_localsearch = 1, 
                            fix_assignment = False):
    '''
    Runs the local search (with delta) with the given input parameters on the starting instances given in the input
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param input_filename: name of the input file (i.e. results of open or close greedy) on which the local search
        should be run
    :param output_filename: name of the output file
    :param iteration_limits: list of iteration limits to be used within the local search
    :param num_consider_per_iterations: list of number of facilities to consider opening / closing in each iteration
    :param depths_greedy_reassign: list of depths of the greedy reassignment within the greedy reassign open algorithm
    :param: time_limit_while_loop: the time limit in seconds used in the while loop in the BFLP local search
    :param assignment_method: method which is used to recompute the assignment if a change is made
    param final_assignment_method: assignment method to run after open facilities decided,
       currently only support relaxation rounding
    :param local_search_methods: BUAP local search method to run on the assignment resulting from assignment_method
        and final_assignment_method; options are "None", "local_random_reassign", "local_random_swap"
    :param num_consider_in_relaxation: number of most preferred facilities to consider in relaxation,
        if -1 all facilities are considered in relaxation
    :param cutoff_BUAP_localsearch: parameter for BUAP local search for minimum preference required to reassign to the facility
    :param time_limit_while_BUAP_localsearch: time limit in the BUAP local search
    :param fix_assignment: boolean indicating whether the assignment should be recomputed from scratch
        using assignment_method every time the open facilities are changed
    :return: list of results of the local search
    '''
    input_results_list = read_results_list(input_filename)
    output_results_list = []
    for result in input_results_list: 
        for num_consider in num_consider_per_iterations:
            for depth in depths_greedy_reassign:
                for iteration_limit in iteration_limits:
                    starting_assignment = result["solution_details"]["assignment"]
                    starting_open_facs = result["solution_details"]["open_facs"]
                    starting_obj = result["solution_details"]["objective_value"]
                    budget_factor = result["model_details"]["budget_factor"]
                    users = result["model_details"]["users"]
                    facs = result["model_details"]["facs"]
                    cap_factor = result["model_details"]["cap_factor"]
                    print("Budget factor: " + str(budget_factor))
                    print("Num consider per iteration: " + str(num_consider))
                    print("Depth: " + str(depth))
                    print("Iteration limit: " + str(iteration_limit))
                    is_feasible, localsearch_result = localsearch_with_change(users_and_facs_df, travel_dict, users, 
                                                                              facs, starting_assignment,
                                starting_open_facs, starting_obj, budget_factor, cap_factor = cap_factor,
                                num_consider_per_iteration = num_consider, time_limit_while_loop = time_limit_while_loop, 
                                iteration_limit_while_loop = iteration_limit, assignment_method=assignment_method,
                                final_assignment_method = final_assignment_method, 
                                localsearch_method = localsearch_method, num_consider_in_relaxation = num_consider_in_relaxation, 
                                tolerance_relaxation = 2e-3, time_limit_relaxation = 1000,
                                threads_relaxation= 1,
                                cutoff_BUAP_localsearch = cutoff_BUAP_localsearch, 
                                time_limit_while_BUAP_localsearch = time_limit_while_BUAP_localsearch,
                                depth_greedy_reassignment = depth, fix_assignment = fix_assignment,
                                output_filename = "test.json", write_to_file = False)
                    overall_result = localsearch_result.copy()
                    overall_result["starting_results"] = result.copy()
                    output_results_list.append(overall_result)
                    write_results_list(output_results_list, output_filename)
    return output_results_list

def get_results_localsearch_without_change(users_and_facs_df, travel_dict, input_filename, output_filename,
                            iteration_limits = [50, 100, 200], num_consider_per_iterations = [10,50,100],
                            depths_greedy_reassign = [1,2], time_limit_while_loop = 3600, assignment_method="greedy_assign",
                            final_assignment_method = "relaxation_rounding", 
                            localsearch_method = "local_random_reassign", num_consider_in_relaxation = 50, 
                            cutoff_BUAP_localsearch = 0.2, time_limit_while_BUAP_localsearch = 1, 
                            fix_assignment = False):
    '''
    Runs the local search (with random choices instead of delta)
         with the given input parameters on the starting instances given in the input
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param input_filename: name of the input file (i.e. results of open or close greedy) on which the local search
        should be run
    :param output_filename: name of the output file
    :param iteration_limits: list of iteration limits to be used within the local search
    :param num_consider_per_iterations: list of number of facilities to consider opening / closing in each iteration
    :param depths_greedy_reassign: list of depths of the greedy reassignment within the greedy reassign open algorithm
    :param: time_limit_while_loop: the time limit in seconds used in the while loop in the BFLP local search
    :param assignment_method: method which is used to recompute the assignment if a change is made
    param final_assignment_method: assignment method to run after open facilities decided,
       currently only support relaxation rounding
    :param local_search_methods: BUAP local search method to run on the assignment resulting from assignment_method
        and final_assignment_method; options are "None", "local_random_reassign", "local_random_swap"
    :param num_consider_in_relaxation: number of most preferred facilities to consider in relaxation,
        if -1 all facilities are considered in relaxation
    :param cutoff_BUAP_localsearch: parameter for BUAP local search for minimum preference required to reassign to the facility
    :param time_limit_while_BUAP_localsearch: time limit in the BUAP local search
    :param fix_assignment: boolean indicating whether the assignment should be recomputed from scratch
        using assignment_method every time the open facilities are changed
    :return: list of results of the local search
    '''
    input_results_list = read_results_list(input_filename)
    output_results_list = []
    for result in input_results_list: 
        for num_consider in num_consider_per_iterations:
            for depth in depths_greedy_reassign:
                for iteration_limit in iteration_limits:
                    starting_assignment = result["solution_details"]["assignment"]
                    starting_open_facs = result["solution_details"]["open_facs"]
                    starting_obj = result["solution_details"]["objective_value"]
                    budget_factor = result["model_details"]["budget_factor"]
                    users = result["model_details"]["users"]
                    facs = result["model_details"]["facs"]
                    cap_factor = result["model_details"]["cap_factor"]
                    print("Budget factor: " + str(budget_factor))
                    print("Num consider per iteration: " + str(num_consider))
                    print("Depth: " + str(depth))
                    print("Iteration limit: " + str(iteration_limit))
                    is_feasible, localsearch_result = localsearch_without_change(users_and_facs_df, travel_dict, users, 
                                                                                 facs, starting_assignment,
                                starting_open_facs, starting_obj, budget_factor, cap_factor = cap_factor,
                                num_consider_per_iteration = num_consider, time_limit_while_loop = time_limit_while_loop, 
                                iteration_limit_while_loop = iteration_limit, assignment_method = assignment_method,
                                final_assignment_method = final_assignment_method, 
                                localsearch_method = localsearch_method, num_consider_in_relaxation = num_consider_in_relaxation, 
                                tolerance_relaxation = 2e-3, time_limit_relaxation = 1000,
                                threads_relaxation= 1,
                                cutoff_BUAP_localsearch = cutoff_BUAP_localsearch,
                                time_limit_while_BUAP_localsearch = time_limit_while_BUAP_localsearch,
                                depth_greedy_reassignment = depth, fix_assignment = fix_assignment,
                                output_filename = "test_without_change.json")

                    overall_result = localsearch_result.copy()
                    overall_result["starting_results"] = result.copy()
                    output_results_list.append(overall_result)
                    write_results_list(output_results_list, output_filename)
    return output_results_list

def write_results_localsearch_table(results_list_filename, output_filename,BLFP_MIP_filename, output_abs_path = None):
    '''
    Given a json format of BFLP local search results, saves the relevant results as an excel table.
    :param results_list_filename: file name of the BFLP local search results list, expected to be in own_results folder
    :param output_filename: file name of the output excel file
    :param BFLP_MIP_filename: file name of the corresponding results for the BFLP MIP in order to compare the results
        of the heuristics to what the MIP achieves, expected to be in own_results folder
    :param output_abs_path: file path for the output file, if not given the resulting excel file will be put
        into the own_results folder
    '''
    results_list = read_results_list(results_list_filename)
    BFLP_MIP_list = read_results_list(BLFP_MIP_filename)
    dictionary_BFLP_MIP = {}
    for result in BFLP_MIP_list:
        dictionary_BFLP_MIP[result["model_details"]["budget_factor"]] = result["solution_details"]["objective_value"]
    assignment_method = []
    local_search_method = []
    final_assignment_method = []
    consider_per_iterations = []
    number_of_iterations = []
    depths = []
    budget_factors = []
    running_time = []
    objective_value =  []
    starting_objective_value = []
    MIP_objectives = []
    Delta_MIPs = []
    Delta_MIPs_starting = []
    for result in results_list:
        budget_factor =result["model_details"]["budget_factor"]
        consider_per_iterations.append(result["algorithm_details"]["num_consider_per_iteration"])
        number_of_iterations.append(result["algorithm_details"]["iteration_limit_while_loop"])
        depths.append(result["algorithm_details"]["depth_greedy_reassignment"])
        budget_factors.append(budget_factor)
        running_time.append(result["solution_details"]["solving_time"])
        assignment_method.append(result["algorithm_details"]["assignment_method"])
        local_search_method.append(result["algorithm_details"]["local_search_assignment_method"])
        final_assignment_method.append(result["algorithm_details"]["final_assignment_method"])
        objective_value.append(result["solution_details"]["objective_value"])
        starting_objective_value.append(result["starting_results"]["solution_details"]["objective_value"])
        closest_budget_factor = 0.0
        closest_budget_factor_distance = 1.0
        for b in dictionary_BFLP_MIP.keys():
            if abs(budget_factor - b) < closest_budget_factor_distance:
                closest_budget_factor_distance = abs(budget_factor - b)
                closest_budget_factor = b
        if closest_budget_factor_distance > 1e-4:
            print("Failed to find MIP results. Creating table failed.")
            return
        MIP_objective = dictionary_BFLP_MIP[closest_budget_factor]
        MIP_objectives.append(MIP_objective)
        Delta_MIPs.append(100*(MIP_objective-result["solution_details"]["objective_value"])/ MIP_objective)
        Delta_MIPs_starting.append(100*(MIP_objective-result["starting_results"]["solution_details"]["objective_value"])/ MIP_objective)


    data = {"Budget factor": budget_factors, "Number of facilities considered per iteration (n_c)": consider_per_iterations,
             "Depth of localsearch within greedy reassign":depths, "number_of_iterations": number_of_iterations,
            "Assignment method": assignment_method, "Final assignment metod": final_assignment_method,
            "Local search assignment metod": local_search_method,
            "Run time [s]": running_time, "Starting objective value": starting_objective_value, 
            "Objective value after local search": objective_value, "MIP objective value": MIP_objectives,
            "Delta_MIP of starting solution [%]": Delta_MIPs_starting, "Delta_MIP after BFLP local search [%]": Delta_MIPs 
            }
    df = pd.DataFrame(data=data)

    # save the table
    if not output_abs_path:
        output_abs_path = os.getcwd() + "/own_results"
    if not os.path.exists(output_abs_path):
        os.makedirs(output_abs_path)
    with pd.ExcelWriter(output_abs_path + "/" + output_filename) as writer:
        df.to_excel(writer, index=False)