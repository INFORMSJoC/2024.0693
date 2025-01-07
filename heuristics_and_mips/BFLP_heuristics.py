"""
Functions for running different BFLP heuristics
"""

import time
from utils import *
import numpy as np
from BUAP_heuristics import greedily_assign_users, run_assignment_method, greedy_reassign_open
from BUAP_MIP import *
from math import inf
import random

def Schmitt_Singh_localsearch(users_and_facs_df, travel_dict, users, facs, lb=0.0, budget_factor=1.0, cap_factor=1.5,
                   turnover_factor=0.02, tolerance=5e-3, time_limit=20000, iteration_limit=20):
    """
    Schmitt and Singh local search heuristic, code copied from 
    https://github.com/schmitt-hub/preferential_access_and_fairness_in_waste_management
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
    :param iteration_limit: limit for the number of consecutive iterations without improvement
    :return: dictionary of results
    """
    print('Running Schmitt and Singh heurisitic...')
    start_time = time.time()
    # filter the data frame by the used facilities and sort it by their capacities
    sorted_users_and_facs_df = users_and_facs_df.iloc[facs].sort_values(by=['capacity'], ascending=False)

    # initialize
    nr_of_users = len(users)
    nr_of_facs = len(facs)
    budget = round(budget_factor * nr_of_facs)
    cap_dict = {j: cap_factor * users_and_facs_df.at[j, 'capacity'] for j in facs}
    best_open_facs = sorted_users_and_facs_df.index[:budget]
    best_obj = cap_factor * sum(users_and_facs_df['capacity'])
    best_gap = (best_obj-lb)/best_obj
    best_assignment = {}
    it_ctr = 0
    open_facs = best_open_facs.copy()
    turnover = round(turnover_factor * budget)
    is_feasible = False

    # create dictionary of expected travelers
    travel_matrix = np.zeros((nr_of_users, nr_of_facs))
    for i in range(nr_of_users):
        for j in range(nr_of_facs):
            travel_matrix[i][j] = travel_dict[users[i]][facs[j]]
    exp_travelers_matrix = np.array(users_and_facs_df.loc[users]['population']).reshape(-1, 1) * travel_matrix
    exp_travelers_dict = {users[i]: {facs[j]: exp_travelers_matrix[i][j] for j in range(nr_of_facs)}
                          for i in range(nr_of_users)}
    print(best_obj)
    while time.time()-start_time <= time_limit:
        # assign users
        is_feasible_assignment, assignment_results = greedily_assign_users(users_and_facs_df, travel_dict, users, facs,
                                                                           open_facs, exp_travelers_dict, cap_dict)
        print(assignment_results['obj'])
        # update best solution
        if is_feasible_assignment and assignment_results['obj'] < best_obj:
            is_feasible = True
            best_obj = assignment_results['obj']
            best_gap = (best_obj-lb)/best_obj
            best_open_facs = open_facs.copy()
            best_assignment = assignment_results['assignment']
            it_ctr = 0
        else:
            it_ctr += 1

        # check stop criteria
        if it_ctr == iteration_limit or best_gap < tolerance:
            break
        # update open facilities
        sorted_facs = dict(sorted(assignment_results['utilization'].items(), key=lambda item: item[1]))
        ditched_facs = list(sorted_facs.keys())[:turnover]
        ditched_zipcodes = [area for (area, fac) in assignment_results['assignment'].items() if fac in ditched_facs]
        open_facs = list(sorted_facs.keys())[turnover:]
        access = {j: sum(exp_travelers_dict[i][j] for i in ditched_zipcodes) for j in facs if j not in open_facs}
        sorted_access = dict(sorted(access.items(), key=lambda item: -item[1]))
        open_facs += list(sorted_access.keys())[:turnover]
        open_facs = pd.Index(open_facs)

    solving_time = time.time()-start_time
    if not is_feasible:
        print('no feasible solution could be constructed.')
        return is_feasible, {}

    # write dictionary with results
    results = {"solution_details":
                   {"assignment": best_assignment, "open_facs": list(best_open_facs), "objective_value": best_obj,
                    "lower_bound": lb, "solving_time": solving_time},
               "model_details":
                   {"users": users, "facs": facs, "cap_factor": cap_factor, "budget_factor": budget_factor,
                    "turnover_factor": turnover_factor, "tolerance": tolerance, "time_limit": time_limit,
                    "iteration_limit": iteration_limit}
               }
    return is_feasible, results

def close_greedy_basic(users_and_facs_df, travel_dict, users, facs, budget_factor = 0.7, starting_open_facs = None,
                        cap_factor=1.5, threads=1, tolerance=0.0,cutoff_localsearch = 0.2, num_consider_in_relaxation = -1,
                        num_consider_per_iteration = 5,time_limit_relaxation=1000, assignment_method ="greedy_assign",
                        local_search_method = "None", time_limit_while_localsearch = 1, 
                        final_assignment_method = "relaxation_rounding",output_filename = "test.json"):
    """
    Basic implementation of the close greedy algorithm for the BFLP based on the DROP procedure.
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param budget_factor: proportion of facilities we want to remain open
    :param starting_open_facs: if this is none, we start with all facilities open,
        otherwise we start with the ones that are given
    :param cap_factor: factor by which all capacities are scaled
    :param threads: number of threads used in the relaxation
    :param tolerance: tolerance on the optimality of the solution used in the relaxation
    :param cutoff_localsearch: parameter for localsearch for minimum preference required to reassing/swap to that facility
    :param num_consider_in_relaxation: number of most preferred facilities to consider in relaxation,
        if -1 all facilities are considered in relaxation
    :param num_consider_per_iteration: number of facilities to consider closing in each iteration
    :param time_limit_relaxation: time limit in seconds for the optimization used in the relaxation
    :param assignment_method: which method of assignment used when deciding which facilities are best to close
        options are "relaxation_rounding" or "greedy_assign"
    :param local_search_method: BUAP local search method to run on the assignment resulting from assignment_method
        and final_assignment_method: options are "None", "local_random_reassign", "local_random_swap"
    :param: time_limit_while_localsearch: the time limit in seconds used in the while loop in the BUAP localsearch
    :param final_assignment_method: assignment method to run after open facilities decided 
        if this is different to the one used in each iteration, currently only support relaxation rounding
    :param output_filename: name of the output file 
    :return: boolean indicating feasibility and dictionary containing the results
    """
    start_time = time.time()
    # dictionary of capacities of facilities after scaling
    cap_dict = {j: cap_factor * users_and_facs_df.at[j, 'capacity'] for j in facs}
    nr_of_users = len(users)
    nr_of_facs = len(facs)
    # create dictionary of how many users are expected to travel to each facility
    # using numpy here since it is 10 times faster
    travel_matrix = np.zeros((nr_of_users, nr_of_facs))
    for i in range(nr_of_users):
        for j in range(nr_of_facs):
            travel_matrix[i][j] = travel_dict[users[i]][facs[j]]
    exp_travelers_matrix = np.array(users_and_facs_df.loc[users]['population']).reshape(-1, 1) * travel_matrix
    exp_travelers_dict = {users[i]: {facs[j]: exp_travelers_matrix[i][j] for j in range(nr_of_facs)}
                            for i in range(nr_of_users)}
    
    # number of facilities that should remain open
    budget = round(budget_factor*len(facs))
    # the facilities we start with being open, defaults to all facilities
    open_facs = []
    if starting_open_facs == None or len(starting_open_facs) < budget:
        open_facs = facs.copy()
    else:
        open_facs = starting_open_facs.copy()
    # the number of facilities we need to close
    num_facs_to_close = len(open_facs) - budget
    print("Number to close: " + str(num_facs_to_close))
    # initialise options for relaxation rounding
    options_relaxation = {"model": None, "new_open_facs": [], "new_closed_facs": [], 
                          "time_limit_relaxation": time_limit_relaxation, "tolerance": tolerance, "threads": threads,
                          "num_consider_in_relaxation": num_consider_in_relaxation }
    options_localsearch = {"cutoff": cutoff_localsearch, "time_limit_while_localsearch": time_limit_while_localsearch}
    is_feasible_assignment, assignment_results = run_assignment_method(assignment_method, local_search_method, 
                                                    users_and_facs_df, travel_dict, users, 
                                                    facs, open_facs.copy(), exp_travelers_dict, 
                                                    cap_dict, 
                                                    options_localsearch = options_localsearch,
                                                    options_relaxation=options_relaxation)
    if not is_feasible_assignment:
        return False, {}
    for k in range(num_facs_to_close):
        print("Facility to close " + str(k))
        underused_facs = sorted(open_facs.copy(), 
                                key=lambda j:  assignment_results["utilization"][j])[:min(num_consider_per_iteration, len(open_facs))]
        is_feasible_assignment_best, assignment_results_best, j_best = False, {"obj": inf}, None
        options_relaxation["new_open_facs"] = []
        for j in underused_facs:
            current_open_facs = open_facs.copy()
            current_open_facs.remove(j)
            options_relaxation["new_closed_facs"] = [j]
            is_feasible_assignment_current, assignment_results_current = run_assignment_method(assignment_method, 
                                                                            local_search_method, 
                                                                            users_and_facs_df, 
                                                                            travel_dict, users, 
                                                                            facs, current_open_facs, 
                                                                            exp_travelers_dict, cap_dict,
                                                                            options_localsearch = options_localsearch, 
                                                                            options_relaxation=options_relaxation)
            if is_feasible_assignment_current and assignment_results_current["obj"] < assignment_results_best["obj"]:
                is_feasible_assignment_best = is_feasible_assignment_current
                assignment_results_best = assignment_results_current
                j_best = j
            options_relaxation["new_open_facs"] = [j]
            if assignment_method == "relaxation_rounding":
                options_relaxation["model"] = assignment_results_current["other"]["model"]
        if is_feasible_assignment_best == False:
            return False, {}
        open_facs.remove(j_best)
        if assignment_method == "relaxation_rounding" and  options_relaxation["new_open_facs"][0] != j_best:
             change_open_facs(options_relaxation["model"], options_relaxation["new_open_facs"], [j_best])
        is_feasible_assignment = is_feasible_assignment_best
        assignment_results = assignment_results_best
    if assignment_method != "relaxation_rounding" and final_assignment_method == "relaxation_rounding":
        options_relaxation["new_closed_facs"] = []
        options_relaxation["new_open_facs"] = []
        options_relaxation["model"] = None
        is_feasible_assignment_current, assignment_results_current = run_assignment_method(final_assignment_method, 
                                                                        local_search_method, 
                                                                        users_and_facs_df, 
                                                                        travel_dict, users, 
                                                                        facs, open_facs, 
                                                                        exp_travelers_dict, 
                                                                        cap_dict, 
                                                                        options_localsearch = options_localsearch,
                                                                        options_relaxation=options_relaxation)
        if is_feasible_assignment_current and assignment_results_current["obj"]< assignment_results["obj"]:
            print("Better final assignment")
            is_feasible_assignment = is_feasible_assignment_current
            assignment_results = assignment_results_current

    end_time = time.time()
    if not is_feasible_assignment:
        return False, {}
    result = {"solution_details":
                {"assignment": assignment_results["assignment"], "open_facs": open_facs, 
                 "objective_value": assignment_results["obj"],
                 "solving_time": end_time-start_time},
            "model_details":
                {"users": users, "facs": facs, "cap_factor": cap_factor,
                    "budget_factor": budget_factor,  "starting_open_facs": starting_open_facs
                },
            "algorithm_details": {
                "overall_algorithm": "close_greedily", "assignment_method": assignment_method,
                "local_search_assignment_method": local_search_method,
                "num_consider_per_iteration": num_consider_per_iteration, 
                "cutoff_localsearch": cutoff_localsearch,"time_limit_while_localsearch": time_limit_while_localsearch,
                "final_assignment_method": final_assignment_method, "num_consider_in_relaxaton": num_consider_in_relaxation
            }
            }
    write_results_list([result], output_filename)
    return True, result

def close_greedy_reuse_assignment(users_and_facs_df, travel_dict, users, facs, budget_factor = 0.7, 
                                    starting_open_facs = None, cap_factor=1.5, threads=1, tolerance=0.0,
                                    cutoff_localsearch = 0.2, num_consider_in_relaxation = -1,
                                    num_consider_per_iteration = 5,time_limit_relaxation=1000, 
                                    assignment_method ="greedy_assign", local_search_method = "None", 
                                    time_limit_while_localsearch = 1, num_fix_assignment_iteration = 50,
                                    final_assignment_method = "relaxation_rounding",output_filename = "test.json"):
    """
    First improvement to basic implementation of the close greedy algorithm for the BFLP based on the DROP procedure;
    this reuses the previous iterations assignment and fixes it using the greedy reassign algorithm.
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param budget_factor: proportion of facilities we want to remain open
    :param starting_open_facs: if this is none, we start with all facilities open, 
        otherwise we start with the ones that are given
    :param cap_factor: factor by which all capacities are scaled
    :param threads: number of threads used in the relaxation
    :param tolerance: tolerance on the optimality of the solution used in the relaxation
    :param cutoff_localsearch: parameter for BUAP localsearch for minimum preference required to reassign to the facility
    :param num_consider_in_relaxation: number of most preferred facilities to consider in relaxationx,
        if -1 all facilities are considered in relaxation
    :param num_consider_per_iteration: number of facilities to consider closing in each iteration
    :param time_limit_relaxation: time limit in seconds for the optimization used in the relaxation
    :param assignment_method: which method of assignment is used to create the first assignment,
        options are "relaxation_rounding" or "greedy_assign"
    :param local_search_method: BUAP local search method to run on the assignment resulting from assignment_method
        and final_assignment_method; options are "None", "local_random_reassign", "local_random_swap"
    :param: time_limit_while_localsearch: the time limit in seconds used in the while loop in the localsearch
    :param num_fix_assignment_iteration: number of iterations after which to recompute the assignment from scratch
        using assignment_method; no saving of model for relaxation rounding implemented so this is only for 
        when assignment_method is "greedy_assign"
    :param final_assignment_method: assignment method to run after open facilities decided,
        currently only support relaxation rounding
    :param output_filename: name of the output file 
    :return: boolean indicating feasibility and dictionary containing the results
    """
    start_time = time.time()
    # dictionary of capacities of facilities after scaling
    cap_dict = {j: cap_factor * users_and_facs_df.at[j, 'capacity'] for j in facs}
    nr_of_users = len(users)
    nr_of_facs = len(facs)
    # create dictionary of how many users are expected to travel to each facility
    # using numpy here since it is 10 times faster
    travel_matrix = np.zeros((nr_of_users, nr_of_facs))
    for i in range(nr_of_users):
        for j in range(nr_of_facs):
            travel_matrix[i][j] = travel_dict[users[i]][facs[j]]
    exp_travelers_matrix = np.array(users_and_facs_df.loc[users]['population']).reshape(-1, 1) * travel_matrix
    exp_travelers_dict = {users[i]: {facs[j]: exp_travelers_matrix[i][j] for j in range(nr_of_facs)}
                            for i in range(nr_of_users)}
    
    # number of facilities that should remain open
    budget = round(budget_factor*len(facs))
    # the facilities we start with being open, defaults to all facilities
    open_facs = []
    if starting_open_facs == None or len(starting_open_facs) < budget:
        open_facs = facs.copy()
    else:
        open_facs = starting_open_facs.copy()
    # the number of facilities we need to close
    num_facs_to_close = len(open_facs) - budget
    print("Number to close: " + str(num_facs_to_close))
    # initialise options for relaxation rounding
    options_relaxation = {"model": None, "new_open_facs": [], "new_closed_facs": [], 
                          "time_limit_relaxation": time_limit_relaxation, "tolerance": tolerance, "threads": threads,
                          "num_consider_in_relaxation": num_consider_in_relaxation }
    options_localsearch = {"cutoff": cutoff_localsearch, "time_limit_while_localsearch": time_limit_while_localsearch}
    is_feasible_assignment, assignment_results = run_assignment_method(assignment_method, local_search_method,
                                                    users_and_facs_df, travel_dict, users, 
                                                    facs, open_facs.copy(), exp_travelers_dict, 
                                                    cap_dict, options_localsearch = options_localsearch,
                                                    options_relaxation=options_relaxation)
    if not is_feasible_assignment:
        return False, {}
    for i in range(num_facs_to_close):
        print("closing facility " + str(i))
        if len(open_facs) > num_consider_per_iteration:
            underused_facs = sorted(open_facs.copy(), 
                                    key=lambda j:  assignment_results["utilization"][j])[0:min(num_consider_per_iteration, len(open_facs))]
        else:
            underused_facs = open_facs.copy()
        is_feasible_assignment_best, assignment_results_best, j_best = False, {"obj": inf}, None
        for j in underused_facs:
            is_feasible_assignment_current, assignment_results_current = greedily_assign_users(users_and_facs_df,
                                                                            travel_dict, users, facs, 
                                                                            open_facs,
                                                                            exp_travelers_dict, 
                                                                            cap_dict, 
                                                                            assignment = assignment_results["assignment"],
                                                                            fac_to_close = j)
            if is_feasible_assignment_current and assignment_results_current["obj"] < assignment_results_best["obj"]:
                is_feasible_assignment_best = is_feasible_assignment_current
                assignment_results_best = assignment_results_current.copy()
                j_best = j
        if is_feasible_assignment_best == False:
            return False, {}
        open_facs.remove(j_best)
        is_feasible_assignment = is_feasible_assignment_best
        assignment_results = assignment_results_best.copy()
        print("objective this round " + str( assignment_results["obj"]))
        if i % num_fix_assignment_iteration == 0 and i> 0:
            print("Running assignment method")
            is_feasible_assignment_rerun, assignment_results_rerun = run_assignment_method(assignment_method, 
                                                                        local_search_method, 
                                                                        users_and_facs_df, 
                                                                        travel_dict, users, facs, 
                                                                        open_facs.copy(), 
                                                                        exp_travelers_dict, cap_dict, 
                                                                        options_localsearch = options_localsearch, 
                                                                        options_relaxation=options_relaxation)
            if is_feasible_assignment_rerun and assignment_results_rerun["obj"] < assignment_results["obj"]:
                print("Used new assignment")
                is_feasible_assignment = is_feasible_assignment_rerun
                assignment_results = assignment_results_rerun

    if final_assignment_method == "relaxation_rounding":
        options_relaxation["new_closed_facs"] = []
        options_relaxation["new_open_facs"] = []
        is_feasible_assignment_current, assignment_results_current = run_assignment_method(final_assignment_method, 
                                                                        local_search_method,
                                                                        users_and_facs_df, 
                                                                        travel_dict, users, 
                                                                        facs, open_facs, 
                                                                        exp_travelers_dict, 
                                                                        cap_dict, 
                                                                        options_localsearch = options_localsearch,
                                                                        options_relaxation=options_relaxation)
        if is_feasible_assignment_current and assignment_results_current["obj"]< assignment_results["obj"]:
            print("Better final assignment")
            is_feasible_assignment = is_feasible_assignment_current
            assignment_results = assignment_results_current
    
    end_time = time.time()
    if not is_feasible_assignment:
        return False, {}
    result = {"solution_details":
                {"assignment": assignment_results["assignment"], "open_facs": open_facs, 
                 "objective_value": assignment_results["obj"], "solving_time": end_time-start_time},
            "model_details":
                {"users": users, "facs": facs, "cap_factor": cap_factor,
                    "budget_factor": budget_factor, "starting_open_facs": starting_open_facs
                },
            "algorithm_details": {
                "overall_algorithm": "close_greedy_reuse_assignment", "assignment_method": assignment_method,
                "local_search_assignment_method": local_search_method, 
                "num_consider_per_iteration": num_consider_per_iteration, 
                "cutoff_localsearch": cutoff_localsearch,"time_limit_while_localsearch": time_limit_while_localsearch,
                "final_assignment_method": final_assignment_method, "num_consider_in_relaxaton": num_consider_in_relaxation,
                "num_fix_assignment_iteration": num_fix_assignment_iteration, "fixing_assignment_method": "greedy_reassign"
            }
            }
    write_results_list([result], output_filename)
    return True, result

def close_greedy_final(users_and_facs_df, travel_dict, users, facs, budget_factor = 0.7, 
                        starting_open_facs = None,cap_factor=1.5, threads=1, tolerance=0.0,
                        cutoff_localsearch = 0.2, num_consider_in_relaxation = -1, 
                        num_consider_per_iteration = 50,time_limit_relaxation=1000,
                        assignment_method ="greedy_assign",local_search_method = "None", 
                        time_limit_while_localsearch = 1, num_fix_assignment_iteration = 5,
                        final_assignment_method = "relaxation_rounding",output_filename = "test.json",
                        write_to_file = True):
    """
    Final implementation of the close greedy algorithm for the BFLP with both improvements based on the DROP procedure.
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param budget_factor: proportion of facilities we want to remain open
    :param starting_open_facs: if this is none, we start with all facilities open, 
        otherwise we start with the ones that are given
    :param cap_factor: factor by which all capacities are scaled
    :param threads: number of threads used in the relaxation
    :param tolerance: tolerance on the optimality of the solution used in the relaxation
    :param cutoff_localsearch: parameter for BUAP localsearch for minimum preference required to reassign to the facility
    :param num_consider_in_relaxation: number of most preferred facilities to consider in relaxation,
        if -1 all facilities are considered in relaxation
    :param num_consider_per_iteration: number of facilities to consider closing in each iteration
    :param time_limit_relaxation: time limit in seconds for the optimization used in the relaxation
    :param assignment_method: which method of assignment is used to create the first assignment,
        options are "relaxation_rounding" or "greedy_assign"
    :param local_search_method: BUAP local search method to run on the assignment resulting from assignment_method
        and final_assignment_method; options are "None", "local_random_reassign", "local_random_swap"
    :param: time_limit_while_localsearch: the time limit in seconds used in the while loop in the localsearch
    :param num_fix_assignment_iteration: number of iterations after which to recompute the assignment from scratch
        using assignment_method; no saving of model for relaxation rounding implemented so this is only for 
        when assignment_method is "greedy_assign"
    :param final_assignment_method: assignment method to run after open facilities decided,
       currently only support relaxation rounding
    :param output_filename: name of the output file 
    :param write_to_file: boolean indicating whether the result should be written to file
    :return: boolean indicating feasibility and dictionary containing the results
    """
    start_time = time.time()
    # dictionary of capacities of facilities after scaling
    cap_dict = {j: cap_factor * users_and_facs_df.at[j, 'capacity'] for j in facs}
    nr_of_users = len(users)
    nr_of_facs = len(facs)
    # create dictionary of how many users are expected to travel to each facility
    # using numpy here since it is 10 times faster
    travel_matrix = np.zeros((nr_of_users, nr_of_facs))
    for i in range(nr_of_users):
        for j in range(nr_of_facs):
            travel_matrix[i][j] = travel_dict[users[i]][facs[j]]
    exp_travelers_matrix = np.array(users_and_facs_df.loc[users]['population']).reshape(-1, 1) * travel_matrix
    exp_travelers_dict = {users[i]: {facs[j]: exp_travelers_matrix[i][j] for j in range(nr_of_facs)}
                            for i in range(nr_of_users)}
    
    # number of facilities that should remain open
    budget = round(budget_factor*len(facs))
    open_facs = []
    # the facilities we start with being open, defaults to all facilities
    if starting_open_facs == None or len(starting_open_facs) < budget:
        open_facs = facs.copy()
    else:
        open_facs = starting_open_facs.copy()
    # the number of facilities we need to close
    num_facs_to_close = len(open_facs) - budget
    print("Number to close: " + str(num_facs_to_close))
    # initialise options for relaxation rounding
    options_relaxation = {"model": None, "new_open_facs": [], "new_closed_facs": [], 
                          "time_limit_relaxation": time_limit_relaxation, "tolerance": tolerance, 
                          "threads": threads, "num_consider_in_relaxation": num_consider_in_relaxation }
    options_localsearch = {"cutoff": cutoff_localsearch, "time_limit_while_localsearch": time_limit_while_localsearch}
    is_feasible_assignment, assignment_results = run_assignment_method(assignment_method, local_search_method,
                                                    users_and_facs_df, travel_dict, users,
                                                    facs, open_facs.copy(), exp_travelers_dict,
                                                    cap_dict, options_localsearch = options_localsearch,
                                                    options_relaxation=options_relaxation)
    if not is_feasible_assignment:
        return False, {}
    facs_to_check = open_facs.copy()
    change_in_objective = {j: 0 for j in open_facs}
    for i in range(num_facs_to_close):
        print("closing facility " + str(i))
        is_feasible_assignment_best, assignment_results_best, j_best = False, {"obj": inf}, None
        for j in facs_to_check:
            is_feasible_assignment_current, assignment_results_current = greedily_assign_users(users_and_facs_df,
                                                                            travel_dict, users, facs,
                                                                            open_facs, 
                                                                            exp_travelers_dict,
                                                                            cap_dict, 
                                                                            assignment = assignment_results["assignment"],
                                                                            fac_to_close = j)
            change_in_objective[j] = assignment_results_current['obj'] - assignment_results['obj']
            if is_feasible_assignment_current and assignment_results_current["obj"] < assignment_results_best["obj"]:
                is_feasible_assignment_best = is_feasible_assignment_current
                assignment_results_best = assignment_results_current.copy()
                j_best = j
        if is_feasible_assignment_best == False:
            return False, {}
        open_facs.remove(j_best)
        is_feasible_assignment = is_feasible_assignment_best
        assignment_results = assignment_results_best.copy()
        print("objective this round " + str( assignment_results["obj"]))
        facs_to_check = sorted(open_facs, 
                               key=lambda j: change_in_objective[j])[0:min(len(open_facs), num_consider_per_iteration)]
        if i % num_fix_assignment_iteration == 0 and i> 0:
            print("Running assignment method")
            is_feasible_assignment_rerun, assignment_results_rerun = run_assignment_method(assignment_method,
                                                                        local_search_method, 
                                                                        users_and_facs_df, 
                                                                        travel_dict, users, facs,
                                                                        open_facs.copy(), 
                                                                        exp_travelers_dict, cap_dict, 
                                                                        options_localsearch = options_localsearch, 
                                                                        options_relaxation=options_relaxation)
            if is_feasible_assignment_rerun and assignment_results_rerun["obj"] < assignment_results["obj"]:
                print("Used new assignment")
                is_feasible_assignment = is_feasible_assignment_rerun
                assignment_results = assignment_results_rerun

    if final_assignment_method == "relaxation_rounding":
        options_relaxation["new_closed_facs"] = []
        options_relaxation["new_open_facs"] = []
        is_feasible_assignment_current, assignment_results_current = run_assignment_method(final_assignment_method,
                                                                        local_search_method, 
                                                                        users_and_facs_df, travel_dict,
                                                                        users, facs, open_facs, 
                                                                        exp_travelers_dict, cap_dict,
                                                                        options_localsearch = options_localsearch,
                                                                        options_relaxation=options_relaxation)
        if is_feasible_assignment_current and assignment_results_current["obj"]< assignment_results["obj"]:
            print("Better final assignment")
            is_feasible_assignment = is_feasible_assignment_current
            assignment_results = assignment_results_current
    
    end_time = time.time()
    if not is_feasible_assignment:
        return False, {}
    result = {"solution_details":
                {"assignment": assignment_results["assignment"], "open_facs": open_facs, 
                 "objective_value": assignment_results["obj"], "solving_time": end_time-start_time},
            "model_details":
                {"users": users, "facs": facs, "cap_factor": cap_factor,
                    "budget_factor": budget_factor, "starting_open_facs": starting_open_facs
                },
            "algorithm_details": {
                "overall_algorithm": "close_greedy_final", "assignment_method": assignment_method,
                "local_search_assignment_method": local_search_method, 
                "num_consider_per_iteration": num_consider_per_iteration, "cutoff_localsearch": cutoff_localsearch,
                "time_limit_while_localsearch": time_limit_while_localsearch,
                "final_assignment_method": final_assignment_method, "num_consider_in_relaxaton": num_consider_in_relaxation, "num_fix_assignment_iteration": num_fix_assignment_iteration,
                "fixing_assignment_method": "greedy_reassign"
            }
            }
    if write_to_file:
        write_results_list([result], output_filename)
    return True, result

def open_greedy(users_and_facs_df, travel_dict, users, facs, budget_factor = 0.7, cap_factor=1.5, 
                threads=1, tolerance=0.0,cutoff_localsearch = 0.2, num_consider_in_relaxation = -1,
                num_consider_per_iteration = 50, depth_greedy_reassignment = 1,time_limit_relaxation=1000,
                assignment_method ="greedy_assign", local_search_method = "None", 
                time_limit_while_localsearch = 1, num_fix_assignment_iteration = 5,
                final_assignment_method = "relaxation_rounding",output_filename = "test.json", write_to_file = True):
    """
    Implementation of the open greedy algorithm for the BFLP based on the ADD procedure.
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param budget_factor: proportion of facilities we want to remain open
    :param cap_factor: factor by which all capacities are scaled
    :param threads: number of threads used in the relaxation
    :param tolerance: tolerance on the optimality of the solution used in the relaxation
    :param cutoff_localsearch: parameter for BUAP localsearch for minimum preference required to reassign to the facility
    :param num_consider_in_relaxation: number of most preferred facilities to consider in relaxation,
        if -1 all facilities are considered in relaxation
    :param num_consider_per_iteration: number of facilities to consider closing in each iteration
    :param depth_greedy_reassignment: depth of the greedy reassignment within the greedy reassign open algorithm
    :param time_limit_relaxation: time limit in seconds for the optimization used in the relaxation
    :param assignment_method: which method of assignment is used to create the first assignment,
        options are "relaxation_rounding" or "greedy_assign"
    :param local_search_method: BUAP local search method to run on the assignment resulting from assignment_method
        and final_assignment_method; options are "None", "local_random_reassign", "local_random_swap"
    :param: time_limit_while_localsearch: the time limit in seconds used in the while loop in the localsearch
    :param num_fix_assignment_iteration: number of iterations after which to recompute the assignment from scratch
        using assignment_method; no saving of model for relaxation rounding implemented so this is only for 
        when assignment_method is "greedy_assign"
    :param final_assignment_method: assignment method to run after open facilities decided,
        currently only support relaxation rounding
    :param output_filename: name of the output file 
    :param write_to_file: boolean indicating whether results should be written to file
    :return: boolean indicating feasibility and dictionary containing the results
    """
    start_time = time.time()
    # dictionary of capacities of facilities after scaling
    cap_dict = {j: cap_factor * users_and_facs_df.at[j, 'capacity'] for j in facs}
    nr_of_users = len(users)
    nr_of_facs = len(facs)
    # create dictionary of how many users are expected to travel to each facility
    # using numpy here since it is 10 times faster
    travel_matrix = np.zeros((nr_of_users, nr_of_facs))
    for i in range(nr_of_users):
        for j in range(nr_of_facs):
            travel_matrix[i][j] = travel_dict[users[i]][facs[j]]
    exp_travelers_matrix = np.array(users_and_facs_df.loc[users]['population']).reshape(-1, 1) * travel_matrix
    exp_travelers_dict = {users[i]: {facs[j]: exp_travelers_matrix[i][j] for j in range(nr_of_facs)}
                            for i in range(nr_of_users)}
    
    # number of facilities that should be open
    num_facs_to_open = round(budget_factor*len(facs))
    # the facilities we start with being open, defaults to all facilities
    closed_facs = facs.copy()
    open_facs = []
    print("Number to open: " + str(num_facs_to_open))
    # initialise options for relaxation rounding
    options_relaxation = {"model": None, "new_open_facs": [], "new_closed_facs": [], 
                            "time_limit_relaxation": time_limit_relaxation, "tolerance": tolerance,
                            "threads": threads, "num_consider_in_relaxation": num_consider_in_relaxation }
    options_localsearch = {"cutoff": cutoff_localsearch, "time_limit_while_localsearch": time_limit_while_localsearch}
    sum_capacities = sum(cap_dict[j] for j in facs)
    is_feasible_assignment, assignment_results = False, {'assignment': {i: "unassigned" for i in users}, 
                                                         'utilization': {j: 0 for j in facs}, 
                                                         'obj': sum_capacities, 'other': {'uassigned_users': users.copy()}}
    facs_to_check = closed_facs.copy()
    change_in_objective = {j: 0 for j in open_facs}
    for i in range(num_facs_to_open):
        print("opening facility " + str(i))
        is_feasible_assignment_best, assignment_results_best, j_best = False, {"obj": sum_capacities}, None
        for j in facs_to_check:
            is_feasible_assignment_current, assignment_results_current =  greedy_reassign_open(users_and_facs_df, 
                                                                            travel_dict, users, 
                                                                            facs,open_facs.copy(), 
                                                                            exp_travelers_dict, 
                                                                            cap_dict, 
                                                                            assignment = assignment_results["assignment"].copy(),
                                                                            fac_to_open = j,
                                                                            depth_greedy_reassignment = depth_greedy_reassignment)
            change_in_objective[j] = assignment_results_current['obj'] - assignment_results['obj']
            # only accept this if better than previous best one
            # also only accept either if it is feasible or if our previous assignment is also not feasible
            if (is_feasible_assignment_current or (is_feasible_assignment_current == is_feasible_assignment)) and assignment_results_current["obj"] < assignment_results_best["obj"]:
                is_feasible_assignment_best = is_feasible_assignment_current
                assignment_results_best = assignment_results_current.copy()
                j_best = j
        open_facs.append(j_best)
        closed_facs.remove(j_best)
        is_feasible_assignment = is_feasible_assignment_best
        assignment_results = assignment_results_best.copy()
        print("objective this round " + str( assignment_results["obj"]))
        facs_to_check = sorted(closed_facs, 
                               key=lambda j: change_in_objective[j])[0:min(len(closed_facs), num_consider_per_iteration)]
        if i % num_fix_assignment_iteration == 0 and i> 0:
            print("Running assignment method")
            is_feasible_assignment_rerun, assignment_results_rerun = run_assignment_method(assignment_method, 
                                                                        local_search_method, 
                                                                        users_and_facs_df, 
                                                                        travel_dict, users, facs,
                                                                        open_facs.copy(), 
                                                                        exp_travelers_dict, cap_dict,
                                                                        options_localsearch = options_localsearch, 
                                                                        options_relaxation=options_relaxation)
            if is_feasible_assignment_rerun and assignment_results_rerun["obj"] < assignment_results["obj"]:
                print("Used new assignment")
                is_feasible_assignment = is_feasible_assignment_rerun
                assignment_results = assignment_results_rerun

    if final_assignment_method == "relaxation_rounding":
        options_relaxation["new_closed_facs"] = []
        options_relaxation["new_open_facs"] = []
        is_feasible_assignment_current, assignment_results_current = run_assignment_method(final_assignment_method, 
                                                                        local_search_method,
                                                                        users_and_facs_df, 
                                                                        travel_dict, users, facs, 
                                                                        open_facs, exp_travelers_dict, 
                                                                        cap_dict, 
                                                                        options_localsearch = options_localsearch,
                                                                        options_relaxation=options_relaxation)
        if is_feasible_assignment_current and assignment_results_current["obj"]< assignment_results["obj"]:
            print("Better final assignment")
            is_feasible_assignment = is_feasible_assignment_current
            assignment_results = assignment_results_current
    
    end_time = time.time()
    if not is_feasible_assignment:
        return False, {}
    result = {"solution_details":
                {"assignment": assignment_results["assignment"], "open_facs": open_facs, 
                 "objective_value": assignment_results["obj"], "solving_time": end_time-start_time},
            "model_details":
                {"users": users, "facs": facs, "cap_factor": cap_factor,
                    "budget_factor": budget_factor, 
                },
            "algorithm_details": {
                "overall_algorithm": "open_greedy", "assignment_method": assignment_method,
                "local_search_assignment_method": local_search_method,
                "num_consider_per_iteration": num_consider_per_iteration, "cutoff_localsearch": cutoff_localsearch,
                "time_limit_while_localsearch": time_limit_while_localsearch,
                "final_assignment_method": final_assignment_method, "num_consider_in_relaxaton": num_consider_in_relaxation,
                "num_fix_assignment_iteration": num_fix_assignment_iteration,
                "fixing_assignment_method": "greedy_reassign_open", "depth_greedy_reassignment": depth_greedy_reassignment
            }
            }
    if write_to_file:
        write_results_list([result], output_filename)
    return True, result

def localsearch_without_change(users_and_facs_df, travel_dict, users, facs, starting_assignment,
                                starting_open_facs, starting_obj, budget_factor, cap_factor = 1.5,
                                num_consider_per_iteration = 50, time_limit_while_loop = 120, 
                                iteration_limit_while_loop = 300, assignment_method="greedy_assign",
                                final_assignment_method = "relaxation_rounding", 
                                localsearch_method = "local_random_reassign", num_consider_in_relaxation = 50, 
                                tolerance_relaxation = 2e-3, time_limit_relaxation = 1000,
                                threads_relaxation= 1,
                                cutoff_BUAP_localsearch = 0.2, time_limit_while_BUAP_localsearch = 1,
                                depth_greedy_reassignment = 1, fix_assignment = False,
                                output_filename = "test_without_change.json", write_to_file = True):
    """
    Local search based on ADD / DROP for overall problem with random chocies of which facilities to consider.
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param starting_assignment: the assignment of users on which the local search starts its search
    :param starting_open_facs: the open faciltiies of the solution on which the local search starts its search
    :param starting_obj: the objective function value of the solution on which the local search starts its search
    :param budget_factor: proportion of facilities to open
    :param cap_factor: factor by which all capacities are scaled
    :param num_consider_per_iteration: how many facilties should be considered for the chosen facility
         to be swapped out with
    :param time_limit_while_loop: the maximum time that the main while loop of the local search should run for
    :param iteration_limit_while_loop: the maximum number of iterations the main while loop of
        the local search should run for
    :param assignment_method: which method of assignment is used to fix the assignment after change is made,
        options are "relaxation_rounding" or "greedy_assign"
    :param final_assignment_method: assignment method to run after open facilities decided,
        currently only support relaxation rounding
    :param local_search_method: BUAP local search method to run on the assignment resulting from assignment_method
        and final_assignment_method; options are "None", "local_random_reassign", "local_random_swap"
    :param num_consider_in_relaxation: number of facilities to consider for each user in BUAP relaxation,
         if -1 all facilities are considered in relaxation
    :param tolerance_relaxation: tolerance of the relaxation of the BUAP
    :param time_limit_relaxation: time limit of running the BUAP relaxation
    :param threads_relaxation: number of threads used when solving the relaxation of the BUAP
    :param: cutoff_BUAP_localsearch: parameter for BUAP localsearch for minimum preference required to 
        reassign to the facility
    :param time_limit_while_BUAP_localsearch: the time limit in seconds used in the while loop in the BUAP localsearch
    :param fix_assignment: if true, fixes assignment with assignment_method after change is made
    :param output_filename: name of the output file
    :param write_to_file: boolean indicating whether results should be written to file
    :return: boolean indicating feasibility and dictionary containing the results
    """
    start_time = time.time()
    # dictionary of capacities of facilities after scaling
    cap_dict = {j: cap_factor * users_and_facs_df.at[j, 'capacity'] for j in facs}
    nr_of_users = len(users)
    nr_of_facs = len(facs)
    # create dictionary of how many users are expected to travel to each facility
    # using numpy here since it is 10 times faster
    travel_matrix = np.zeros((nr_of_users, nr_of_facs))
    for i in range(nr_of_users):
        for j in range(nr_of_facs):
            travel_matrix[i][j] = travel_dict[users[i]][facs[j]]
    exp_travelers_matrix = np.array(users_and_facs_df.loc[users]['population']).reshape(-1, 1) * travel_matrix
    exp_travelers_dict = {users[i]: {facs[j]: exp_travelers_matrix[i][j] for j in range(nr_of_facs)}
                            for i in range(nr_of_users)}
    facs_to_consider = facs.copy()
    open_facs = starting_open_facs.copy()
    closed_facs = [x for x in facs if x not in open_facs]
    if len(open_facs) == 0 or len(closed_facs) == 0:
        return False, {}
    
    assignment_results = { 'assignment': starting_assignment, 'obj': starting_obj }
    print("Starting objective" + str(assignment_results['obj']))
    start_while_loop = time.time()
    counter = 0        
    while (len(facs_to_consider) > 0 and time.time() - start_while_loop < time_limit_while_loop and counter < iteration_limit_while_loop):
        print("Counter: " + str(counter))
        counter += 1
        j = random.choice(facs_to_consider)
        facs_to_consider.remove(j)
        # close j and open another facility close to it
        if j in open_facs:
            best_closed_facs = random.sample(closed_facs, num_consider_per_iteration)
            is_feasible_assignment_best, assignment_results_best, j_best = False, {"obj": inf}, None
             # first close the facility and run re-assignment
            is_feasible_assignment_current_after_close, assignment_results_current_after_close = greedily_assign_users(users_and_facs_df,
                                                                                                    travel_dict, users, 
                                                                                                    facs, 
                                                                                                    open_facs.copy(),
                                                                                                    exp_travelers_dict, 
                                                                                                    cap_dict, 
                                                                                                    assignment = assignment_results["assignment"].copy(),
                                                                                                    fac_to_close = j)
            temp_open_facs = open_facs.copy()
            temp_open_facs.remove(j)
            for j_to_open in best_closed_facs:
                # then open facility and run reassignment
                is_feasible_assignment_current, assignment_results_current =  greedy_reassign_open(users_and_facs_df, 
                                                                                travel_dict, users, 
                                                                                facs, 
                                                                                temp_open_facs.copy(), 
                                                                                exp_travelers_dict, 
                                                                                cap_dict, 
                                                                                assignment = assignment_results_current_after_close["assignment"].copy(), 
                                                                                fac_to_open = j_to_open,
                                                                                depth_greedy_reassignment=depth_greedy_reassignment)
                if is_feasible_assignment_current and assignment_results_current["obj"] < assignment_results_best["obj"]:
                    is_feasible_assignment_best = is_feasible_assignment_current
                    assignment_results_best = assignment_results_current.copy()
                    j_best = j_to_open
            # if best found better than current
            print("Difference " + str(assignment_results_best["obj"] - assignment_results['obj']))
            if is_feasible_assignment_best and assignment_results_best["obj"] < assignment_results['obj']:
                print("j " + str(j))
                print("j_best " + str(j_best))
                open_facs.append(j_best)
                open_facs.remove(j)
                closed_facs.append(j)
                closed_facs.remove(j_best)
                assignment_results = assignment_results_best.copy()
                facs_to_consider = facs.copy()
                print("Better objective " + str( assignment_results["obj"]))

        # open j and close another facility close to it
        else:
            best_open_facs = random.sample(open_facs, num_consider_per_iteration)
            is_feasible_assignment_best, assignment_results_best, j_best = False, {"obj": inf}, None
            #first open facility
            is_feasible_assignment_current_after_open, assignment_results_current_after_open = greedy_reassign_open(users_and_facs_df, 
                                                                                                    travel_dict, users, 
                                                                                                    facs, open_facs.copy(), 
                                                                                                    exp_travelers_dict, 
                                                                                                    cap_dict, assignment = assignment_results["assignment"].copy(), 
                                                                                                    fac_to_open = j,
                                                                                                    depth_greedy_reassignment=depth_greedy_reassignment)
            temp_open_facs = open_facs.copy()
            temp_open_facs.append(j)
            for j_to_close in best_open_facs:
                # then close facility and run reassignment
                is_feasible_assignment_current, assignment_results_current =  greedily_assign_users(users_and_facs_df,
                                                                                travel_dict, users, facs, 
                                                                                temp_open_facs.copy(),
                                                                                exp_travelers_dict, cap_dict, 
                                                                                assignment = assignment_results_current_after_open["assignment"].copy(),
                                                                                fac_to_close = j_to_close)
                if is_feasible_assignment_current and assignment_results_current["obj"] < assignment_results_best["obj"]:
                    is_feasible_assignment_best = is_feasible_assignment_current
                    assignment_results_best = assignment_results_current.copy()
                    j_best = j_to_close
            # if best found better than current
            print("Difference " + str(assignment_results_best["obj"] - assignment_results['obj']))
            if is_feasible_assignment_best and assignment_results_best["obj"] < assignment_results['obj']:
                print("j " + str(j))
                print("j_best " + str(j_best))
                open_facs.append(j)
                open_facs.remove(j_best)
                closed_facs.append(j_best)
                closed_facs.remove(j)
                assignment_results = assignment_results_best.copy()
                facs_to_consider = facs.copy()
                print("Better objective" + str( assignment_results["obj"]))
        if fix_assignment and len(facs) == len(facs_to_consider):
            print("Running assignment method")
            options_relaxation = {"model": None, "new_open_facs": [], "new_closed_facs": [], 
                                    "time_limit_relaxation": time_limit_relaxation, "tolerance": tolerance_relaxation,
                                    "threads": threads_relaxation,
                                    "num_consider_in_relaxation": num_consider_in_relaxation }
            options_localsearch = {"cutoff": cutoff_BUAP_localsearch, "time_limit_while_localsearch": time_limit_while_BUAP_localsearch}
            is_feasible_assignment_rerun, assignment_results_rerun = run_assignment_method(assignment_method, 
                                                                        localsearch_method, 
                                                                        users_and_facs_df, 
                                                                        travel_dict, users, 
                                                                        facs, open_facs.copy(), 
                                                                        exp_travelers_dict, cap_dict, 
                                                                        options_localsearch = options_localsearch,
                                                                        options_relaxation=options_relaxation)
            if is_feasible_assignment_rerun and assignment_results_rerun["obj"] < assignment_results["obj"]:
                print("Used new assignment")
                is_feasible_assignment = is_feasible_assignment_rerun
                assignment_results = assignment_results_rerun
    is_feasible_assignment = True
    if final_assignment_method == "relaxation_rounding":
        options_relaxation = {"model": None, "new_open_facs": [], "new_closed_facs": [], 
                              "time_limit_relaxation": time_limit_relaxation, 
                              "tolerance": tolerance_relaxation, "threads": threads_relaxation,
                            "num_consider_in_relaxation": num_consider_in_relaxation }
        options_localsearch = {"cutoff": cutoff_BUAP_localsearch, 
                               "time_limit_while_localsearch": time_limit_while_BUAP_localsearch}
        is_feasible_assignment_current, assignment_results_current = run_assignment_method(final_assignment_method, 
                                                                        localsearch_method,
                                                                        users_and_facs_df, 
                                                                        travel_dict, users, facs,
                                                                        open_facs, exp_travelers_dict,
                                                                        cap_dict, 
                                                                        options_localsearch = options_localsearch,
                                                                        options_relaxation=options_relaxation)
        if is_feasible_assignment_current and assignment_results_current["obj"]< assignment_results["obj"]:
            print("Better final assignment")
            is_feasible_assignment = is_feasible_assignment_current
            assignment_results = assignment_results_current
    if not is_feasible_assignment:
        return False, {}
    end_time = time.time()
    result = {"solution_details":
                {"assignment": assignment_results["assignment"], "open_facs": open_facs,
                 "objective_value": assignment_results["obj"],"solving_time": end_time-start_time},
            "model_details":
                {"users": users, "facs": facs, "cap_factor": cap_factor,
                    "budget_factor": budget_factor, "starting_open_facs": starting_open_facs
                },
            "algorithm_details": {
                "overall_algorithm": "localsearch_without_change", 
                "local_search_assignment_method": localsearch_method,
                "assignment_method": assignment_method,
                "num_consider_per_iteration": num_consider_per_iteration, "cutoff_localsearch": cutoff_BUAP_localsearch,
                "time_limit_while_localsearch": time_limit_while_BUAP_localsearch, 
                "final_assignment_method": final_assignment_method, 
                "num_consider_in_relaxation": num_consider_in_relaxation, 
                "fixing_assignment_method": "greedy_reassign and greedy_reassign_open", 
                "iteration_limit_while_loop": iteration_limit_while_loop, 
                "depth_greedy_reassignment":depth_greedy_reassignment
            }
            }
    if write_to_file:
        write_results_list([result], output_filename)
    return True, result

def localsearch_with_change(users_and_facs_df, travel_dict, users, facs, starting_assignment,
                                starting_open_facs, starting_obj, budget_factor, cap_factor = 1.5,
                                num_consider_per_iteration = 50, time_limit_while_loop = 120, 
                                iteration_limit_while_loop = 300, assignment_method="greedy_assign",
                                final_assignment_method = "relaxation_rounding", 
                                localsearch_method = "local_random_reassign", num_consider_in_relaxation = 50, 
                                tolerance_relaxation = 2e-3, time_limit_relaxation = 1000,
                                threads_relaxation= 1,
                                cutoff_BUAP_localsearch = 0.2, time_limit_while_BUAP_localsearch = 1,
                                depth_greedy_reassignment = 1, fix_assignment = False,
                                output_filename = "test_without_change.json", write_to_file = True):
    """
    Local search based on ADD / DROP for overall problem with choices of which facilities to consider
        based on how good they appeared to close / open in previous iterations.
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param starting_assignment: the assignment of users on which the local search starts its search
    :param starting_open_facs: the open faciltiies of the solution on which the local search starts its search
    :param starting_obj: the objective function value of the solution on which the local search starts its search
    :param budget_factor: proportion of facilities to open
    :param cap_factor: factor by which all capacities are scaled
    :param num_consider_per_iteration: how many facilties should be considered for the chosen facility
         to be swapped out with
    :param time_limit_while_loop: the maximum time that the main while loop of the local search should run for
    :param iteration_limit_while_loop: the maximum number of iterations the main while loop of
        the local search should run for
    :param assignment_method: which method of assignment is used to fix the assignment after change is made,
        options are "relaxation_rounding" or "greedy_assign"
    :param final_assignment_method: assignment method to run after open facilities decided,
        currently only support relaxation rounding
    :param local_search_method: BUAP local search method to run on the assignment resulting from assignment_method
        and final_assignment_method; options are "None", "local_random_reassign", "local_random_swap"
    :param num_consider_in_relaxation: number of facilities to consider for each user in BUAP relaxation,
         if -1 all facilities are considered in relaxation
    :param tolerance_relaxation: tolerance of the relaxation of the BUAP
    :param time_limit_relaxation: time limit of running the BUAP relaxation
    :param threads_relaxation: number of threads used when solving the relaxation of the BUAP
    :param: cutoff_BUAP_localsearch: parameter for BUAP localsearch for minimum preference required to 
        reassign to the facility
    :param time_limit_while_BUAP_localsearch: the time limit in seconds used in the while loop in the BUAP localsearch
    :param fix_assignment: if true, fixes assignment with assignment_method after change is made
    :param output_filename: name of the output file 
    :param write_to_file: boolean indicating whether results should be written to file
    :return: boolean indicating feasibility and dictionary containing the results
    """
    start_time = time.time()
    # dictionary of capacities of facilities after scaling
    cap_dict = {j: cap_factor * users_and_facs_df.at[j, 'capacity'] for j in facs}
    nr_of_users = len(users)
    nr_of_facs = len(facs)
    # create dictionary of how many users are expected to travel to each facility
    # using numpy here since it is 10 times faster
    travel_matrix = np.zeros((nr_of_users, nr_of_facs))
    for i in range(nr_of_users):
        for j in range(nr_of_facs):
            travel_matrix[i][j] = travel_dict[users[i]][facs[j]]
    exp_travelers_matrix = np.array(users_and_facs_df.loc[users]['population']).reshape(-1, 1) * travel_matrix
    exp_travelers_dict = {users[i]: {facs[j]: exp_travelers_matrix[i][j] for j in range(nr_of_facs)}
                            for i in range(nr_of_users)}
    facs_to_consider = facs.copy()
    open_facs = starting_open_facs.copy()
    closed_facs = [x for x in facs if x not in open_facs]
    if len(open_facs) == 0 or len(closed_facs) == 0:
        return False, {}

    assignment_results = { 'assignment': starting_assignment, 'obj': starting_obj }
    print("Starting objective" + str(assignment_results['obj']))
    # initilialise change: note if facility currently close, this is change of opening 
    # while if currently open its the change of closing
    Change = {j: inf for j in facs}
    for j in open_facs:
        is_feasible_assignment_current, assignment_results_current = greedily_assign_users(users_and_facs_df,
                                                                        travel_dict, users, facs, 
                                                                        open_facs.copy(),
                                                                        exp_travelers_dict, cap_dict, 
                                                                        assignment = assignment_results["assignment"].copy(),
                                                                        fac_to_close = j)
        if is_feasible_assignment_current:
            Change[j] = assignment_results_current["obj"] - assignment_results["obj"]
    for j in closed_facs:
        is_feasible_assignment_current, assignment_results_current =  greedy_reassign_open(users_and_facs_df, 
                                                                        travel_dict, users, 
                                                                        facs, open_facs.copy(), 
                                                                        exp_travelers_dict, cap_dict,
                                                                        assignment = assignment_results["assignment"].copy(), 
                                                                        fac_to_open = j,
                                                                        depth_greedy_reassignment=depth_greedy_reassignment)
        if is_feasible_assignment_current:
            Change[j] = assignment_results_current["obj"] - assignment_results["obj"]
    start_while_loop = time.time()
    counter = 0        
    while (len(facs_to_consider) > 0 and time.time() - start_while_loop < time_limit_while_loop and counter < iteration_limit_while_loop):
        print("Counter: " + str(counter))
        counter += 1
        j = random.choice(facs_to_consider)
        facs_to_choose_from_closed = [j1 for j1 in facs_to_consider if j1 in closed_facs]
        facs_to_choose_from_open = [j1 for j1 in facs_to_consider if j1 in open_facs]
        rand = random.random()
        if (rand < len(open_facs) / len(facs) and len(facs_to_choose_from_open) > 0) or len(facs_to_choose_from_closed) == 0:
            j = min(facs_to_choose_from_open, key = lambda j1:Change[j1])
        else:
            j = min(facs_to_choose_from_closed, key = lambda j1:Change[j1])
        facs_to_consider.remove(j)
        # close j and open another facility 
        if j in open_facs:
            best_closed_facs = sorted(closed_facs, 
                                      key = lambda j1:Change[j1])[0:min(len(closed_facs), num_consider_per_iteration)]
            is_feasible_assignment_best, assignment_results_best, j_best = False, {"obj": inf}, None
             # first close the facility and run re-assignment
            is_feasible_assignment_current_after_close, assignment_results_current_after_close = greedily_assign_users(users_and_facs_df,
                                                                                                travel_dict, users, facs, 
                                                                                                open_facs.copy(),
                                                                                                exp_travelers_dict, 
                                                                                                cap_dict, 
                                                                                                assignment = assignment_results["assignment"].copy(),
                                                                                                fac_to_close = j)
            Change[j] = assignment_results_current_after_close["obj"] - assignment_results["obj"]
            temp_open_facs = open_facs.copy()
            temp_open_facs.remove(j)
            for j_to_open in best_closed_facs:
                # then open facility and run reassignment
                is_feasible_assignment_current, assignment_results_current =  greedy_reassign_open(users_and_facs_df, 
                                                                                travel_dict, users, 
                                                                                facs, temp_open_facs.copy(), 
                                                                                exp_travelers_dict, 
                                                                                cap_dict, assignment = assignment_results_current_after_close["assignment"].copy(), 
                                                                                fac_to_open = j_to_open,
                                                                                depth_greedy_reassignment=depth_greedy_reassignment)
                Change[j_to_open] = assignment_results_current["obj"] - assignment_results["obj"]
                if is_feasible_assignment_current and assignment_results_current["obj"] < assignment_results_best["obj"]:
                    is_feasible_assignment_best = is_feasible_assignment_current
                    assignment_results_best = assignment_results_current.copy()
                    j_best = j_to_open
            # if best found better than current
            print("Difference " + str(assignment_results_best["obj"] - assignment_results['obj']))
            if is_feasible_assignment_best and assignment_results_best["obj"] < assignment_results['obj']:
                print("j " + str(j))
                print("j_best " + str(j_best))
                open_facs.append(j_best)
                open_facs.remove(j)
                closed_facs.append(j)
                closed_facs.remove(j_best)
                assignment_results = assignment_results_best.copy()
                facs_to_consider = facs.copy()
                print("Better objective " + str( assignment_results["obj"]))
                # update change values (swap signs since now instead of opening consider closing and vice versa)
                Change[j] = - Change[j]
                Change[j_best] = - Change[j_best]

        # open j and close another facility close to it
        else:
            best_open_facs = sorted(open_facs, key = lambda j1:Change[j1])[0:min(len(open_facs), num_consider_per_iteration)]
            is_feasible_assignment_best, assignment_results_best, j_best = False, {"obj": inf}, None
            #first open facility
            is_feasible_assignment_current_after_open, assignment_results_current_after_open = greedy_reassign_open(users_and_facs_df, 
                                                                                                    travel_dict, users, 
                                                                                                    facs, open_facs.copy(), 
                                                                                                    exp_travelers_dict, 
                                                                                                    cap_dict, assignment = assignment_results["assignment"].copy(), 
                                                                                                    fac_to_open = j,
                                                                                                    depth_greedy_reassignment=depth_greedy_reassignment)
            Change[j] = assignment_results_current_after_open["obj"] - assignment_results["obj"]
            temp_open_facs = open_facs.copy()
            temp_open_facs.append(j)
            for j_to_close in best_open_facs:
                # then open facility and run reassignment
                is_feasible_assignment_current, assignment_results_current =  greedily_assign_users(users_and_facs_df,
                                                                                travel_dict, users, facs, 
                                                                                temp_open_facs.copy(),
                                                                                exp_travelers_dict, cap_dict, 
                                                                                assignment = assignment_results_current_after_open["assignment"].copy(),
                                                                                fac_to_close = j_to_close)
                Change[j_to_close] = assignment_results_current["obj"] - assignment_results["obj"]
                if is_feasible_assignment_current and assignment_results_current["obj"] < assignment_results_best["obj"]:
                    is_feasible_assignment_best = is_feasible_assignment_current
                    assignment_results_best = assignment_results_current.copy()
                    j_best = j_to_close
            # if best found better than current
            print("Difference " + str(assignment_results_best["obj"] - assignment_results['obj']))
            if is_feasible_assignment_best and assignment_results_best["obj"] < assignment_results['obj']:
                print("j " + str(j))
                print("j_best " + str(j_best))
                open_facs.append(j)
                open_facs.remove(j_best)
                closed_facs.append(j_best)
                closed_facs.remove(j)
                assignment_results = assignment_results_best.copy()
                facs_to_consider = facs.copy()
                print("Better objective" + str( assignment_results["obj"]))
                # update change values (swap signs since now instead of opening consider closing and vice versa)
                Change[j] = - Change[j]
                Change[j_best] = - Change[j_best]
        if fix_assignment and len(facs) == len(facs_to_consider):
            print("Running assignment method")
            options_relaxation = {"model": None, "new_open_facs": [], "new_closed_facs": [], 
                                  "time_limit_relaxation": time_limit_relaxation, "tolerance": tolerance_relaxation,
                                    "threads": threads_relaxation, "num_consider_in_relaxation": num_consider_in_relaxation }
            options_localsearch = {"cutoff": cutoff_BUAP_localsearch, 
                                   "time_limit_while_localsearch": time_limit_while_BUAP_localsearch}
            is_feasible_assignment_rerun, assignment_results_rerun = run_assignment_method(assignment_method, 
                                                                                localsearch_method,
                                                                                users_and_facs_df, 
                                                                                travel_dict, users, facs,
                                                                                open_facs.copy(), 
                                                                                exp_travelers_dict, 
                                                                                cap_dict, 
                                                                                options_localsearch = options_localsearch,
                                                                                options_relaxation=options_relaxation)
            if is_feasible_assignment_rerun and assignment_results_rerun["obj"] < assignment_results["obj"]:
                print("Used new assignment")
                is_feasible_assignment = is_feasible_assignment_rerun
                assignment_results = assignment_results_rerun
    is_feasible_assignment = True
    if final_assignment_method == "relaxation_rounding":
        print("In relaxation rounding if")
        options_relaxation = {"model": None, "new_open_facs": [], "new_closed_facs": [], 
                              "time_limit_relaxation": time_limit_relaxation, "tolerance": tolerance_relaxation, 
                              "threads": threads_relaxation, "num_consider_in_relaxation": num_consider_in_relaxation }
        options_localsearch = {"cutoff": cutoff_BUAP_localsearch, 
                               "time_limit_while_localsearch": time_limit_while_BUAP_localsearch}
        is_feasible_assignment_current, assignment_results_current = run_assignment_method(final_assignment_method, 
                                                                            localsearch_method, 
                                                                            users_and_facs_df, 
                                                                            travel_dict, users, facs,
                                                                            open_facs, exp_travelers_dict,
                                                                            cap_dict, 
                                                                            options_localsearch = options_localsearch,
                                                                            options_relaxation=options_relaxation)
        if is_feasible_assignment_current and assignment_results_current["obj"]< assignment_results["obj"]:
            print("Better final assignment")
            is_feasible_assignment = is_feasible_assignment_current
            assignment_results = assignment_results_current
    if not is_feasible_assignment:
        return False, {}
    end_time = time.time()
    result = {"solution_details":
                {"assignment": assignment_results["assignment"], "open_facs": open_facs, 
                 "objective_value": assignment_results["obj"], "solving_time": end_time-start_time},
            "model_details":
                {"users": users, "facs": facs, "cap_factor": cap_factor,
                    "budget_factor": budget_factor, "starting_open_facs": starting_open_facs
                },
            "algorithm_details": {
                "overall_algorithm": "localsearch",  "local_search_assignment_method": localsearch_method,
                "assignment_method": assignment_method,
                "num_consider_per_iteration": num_consider_per_iteration, "cutoff_localsearch": cutoff_BUAP_localsearch,
                "time_limit_while_localsearch": time_limit_while_BUAP_localsearch,
                "final_assignment_method": final_assignment_method,
                "num_consider_in_relaxation": num_consider_in_relaxation, 
                "fixing_assignment_method": "greedy_assign", "iteration_limit_while_loop": iteration_limit_while_loop, 
                "depth_greedy_reassignment":depth_greedy_reassignment
            }
            }
    if write_to_file:
        write_results_list([result], output_filename)
    return True, result