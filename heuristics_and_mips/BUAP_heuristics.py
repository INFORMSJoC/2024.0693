"""
Functions for running different BUAP heuristics and the subroutines used within these heuristics
"""
import time
from utils import *
import numpy as np
from BUAP_MIP import *
import math


def make_assignment_satisfy_capacity_constraints(users_and_facs_df, travel_dict, cap_dict, users, facs, open_facs, assignment):
    """
    Given an assignment that (potentially) does not satisfy capacity constraints, try and fix the assignment
        to satisfy capacity constraints by reassigning users from the over capacity facilities to other facilities.
        This routine is used in the relaxation rounding algorithm.
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param cap_dict: dictionary of capacities of each facility
    :param users: list of the users used in the instance
    :param faacs: list of facilities used in the instance
    :param open_facs: list of the open facilities used in the instance
    :param assignment: assignment of users to facilities that might not satisfy constraints
    :return: in the case of success the function returns the objective function value of the assignment, the assignment
        and a dictionary of 1-u_j; if the function fails as fixing the assignment it returns 0, {},{}
    """
    slacks, not_satisfied, assigned_users = capacity_constraints_slack(users_and_facs_df, travel_dict, cap_dict,
                                                                        open_facs, assignment)
    failed_to_reassign = False
    for j in not_satisfied:
        capacity = cap_dict[j]
        reassignable_users = assigned_users[j]
        reassignable_users.sort(key=lambda i: travel_dict[i][j])
        too_large_users = []
        for i in reassignable_users:
                if cap_dict[j] > capacity:
                    too_large_users.append(i)
        while slacks[j] < 0 and len(reassignable_users) > 0:
            is_too_large = False
            if len(too_large_users) > 0:
                chosen_user = too_large_users[0]
                is_too_large = True
            else:
                chosen_user = reassignable_users[0]
            other_facs = open_facs.copy()
            other_facs.remove(j)
            other_facs.sort( key=lambda k: travel_dict[chosen_user][k], reverse=True)
            for k in other_facs:
                if slacks[k] - travel_dict[chosen_user][k]*users_and_facs_df.at[chosen_user, 'population'] >= 0:
                    assignment[chosen_user] = k
                    slacks[k] = slacks[k] - travel_dict[chosen_user][k]*users_and_facs_df.at[chosen_user, 'population']
                    slacks[j] = slacks[j] + travel_dict[chosen_user][j]*users_and_facs_df.at[chosen_user, 'population']
                    break
            reassignable_users.remove(chosen_user)
            if is_too_large:
                too_large_users.remove(chosen_user)
        if slacks[j] < 0:
            failed_to_reassign = True

    if failed_to_reassign:
        print("failed to reassign")
        return 0, {}, {}

    not_utilised = {j: 1 for j in facs}
    not_utilised.update({j:(slacks[j] / cap_dict[j]) for j in open_facs})
    objective = sum(cap_dict[j] * (not_utilised[j]) ** 2 for j in facs)
    return objective, assignment, not_utilised
 
def reassign_user_better(slacks, user, original_fac, new_fac, cap_dict, exp_travelers_dict):
    """
    Checks if areassigning user from original_fac to new_fac leads to a better objective function
        This is a subroutine of the local search reassign algorithm.
    :param slacks: how much slack is in curren assignment for each facility as a dictionary,
        i.e. C_j - sum_i U_i P_ij x_ij
    :param user: the user we want to reassign
    :param original_fac: the facility the user is currently assigned to
    :param new_fac: the facility we want to assign it to
    :param cap_dict: dictionary of the facilities' capacities
    :param exp_traverlers_dict: number of people we expect to travel
    :return: false if assignment worse or not feasible, true otherwise and the difference in objective function
    """
    if slacks[new_fac] - exp_travelers_dict[user][new_fac] < 0:
        return False, 0
    old_objective_part = cap_dict[original_fac]*(slacks[original_fac]/cap_dict[original_fac])**2 + cap_dict[new_fac]*(slacks[new_fac]/cap_dict[new_fac])**2
    new_objective_part = cap_dict[original_fac]*((slacks[original_fac] + exp_travelers_dict[user][original_fac])/cap_dict[original_fac])**2 + cap_dict[new_fac]*((slacks[new_fac] - exp_travelers_dict[user][new_fac])/cap_dict[new_fac])**2
    difference = new_objective_part - old_objective_part
    if difference < 0:
        return True, difference
    else:
        return False, difference
    
def swap_user_better(slacks, user_1, user_2, fac_1, fac_2, cap_dict, exp_travelers_dict):
    """
    Checks if reassigning user from original_fac to new_fac leads to a better objective function
         This is a subroutine of the local search swap algorithm.
    :param slacks: how much slack is in curren assignment for each facility as a dictionary,
        i.e. C_j - sum_i U_i P_ij x_ij
    :param user_1: the user currently assigned to fac_1
    :param user_2: the user currently assigned to fac_2
    :param fac_1: the facility the user_1 is currently assigned to
    :param fac_2: the facility user_2 is currently assigned to 
    :param cap_dict: dictionary of the facilities' capacities
    :param exp_traverlers_dict: number of people we expect to travel
    :return: false if assignment worse or not feasible, true otherwise and the the new slacks and difference in objective
    """
    new_slack_1 = slacks[fac_1] - exp_travelers_dict[user_2][fac_1] + exp_travelers_dict[user_1][fac_1]
    new_slack_2 = slacks[fac_2] - exp_travelers_dict[user_1][fac_2] + exp_travelers_dict[user_2][fac_2]
    if new_slack_1 < 0 or new_slack_2 < 0:
        return False, 0, 0, 0
    old_objective_part = cap_dict[fac_1]*(slacks[fac_1]/cap_dict[fac_1])**2 + cap_dict[fac_2]*(slacks[fac_2]/cap_dict[fac_2])**2
    new_objective_part = cap_dict[fac_1]*(new_slack_1/cap_dict[fac_1])**2 + cap_dict[fac_2]*(new_slack_2/cap_dict[fac_2])**2 
    difference = new_objective_part - old_objective_part
    if difference < -1:
        return True, new_slack_1, new_slack_2, difference
    else:
        return False, new_slack_1, new_slack_2, difference

def local_search_reassign(users_and_facs_df, travel_dict, users, facs, open_facs, assignment,
                           exp_travelers_dict, cap_dict, cutoff = 0.2, time_limit_while = 1):
    """
    Local search for the BUAP that given an assignment tries to reassign users and implements the reassignment
        if it leads to a better objective function value.
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param open_facs: list of open facilities
    :param assignment: where to start the local search
    :param exp_travelers_dict: dictionary stating the population expected to travel from each user to each facility
    :param cap_dict: dictionary of the facilities' capacities
    :param cutoff: minimum preference to consider reassinging a user
    :param time_limit_while: time limit of the while loop
    :return: Boolean indicating that the assignment is feasible and a dictionary containing assignments, 
        facility utilizations and the objective value.
        If the input assignment is infeasible, the function simply returns False and an empty dictionary.
    """
    slacks, not_satisfied, assigned_users = capacity_constraints_slack(users_and_facs_df, travel_dict,cap_dict,
                                                                        open_facs, assignment)
    if len(not_satisfied)> 0:
        return False, {}
    users_preferred_facs = {i: [j for j in open_facs if travel_dict[i][j]>= cutoff ] for i in users}
    not_utilised = {j:(slacks[j] / cap_dict[j]) for j in open_facs}
    partial_objective = sum(cap_dict[j] * (not_utilised[j]) ** 2 for j in open_facs)
    start_time = time.time()
    improvement = -20
    counter = 0
    while time.time() - start_time < time_limit_while and improvement < -10:
        improvement = 0
        random_facs = open_facs.copy()
        for j in random_facs:
            if(len(assigned_users[j]) <= 1):
                continue
            for i in assigned_users[j]:
                for j_2 in users_preferred_facs[i]:
                    is_better, change_objective = reassign_user_better(slacks, i, j, j_2, cap_dict, exp_travelers_dict)
                    if is_better:
                        assignment[i] = j_2
                        slacks[j] += exp_travelers_dict[i][j]
                        slacks[j_2] -= exp_travelers_dict[i][j_2]
                        assigned_users[j].remove(i)
                        assigned_users[j_2].append(i)
                        partial_objective += change_objective
                        improvement += change_objective
                        break
        counter += 1
        print(improvement)
    closed_facs = [j for j in facs if j not in open_facs]
    objective = partial_objective + sum(cap_dict[j] for j in closed_facs)
    return True, {'assignment': assignment, 'utilization': {j: 1-(slacks[j]/cap_dict[j]) for j in open_facs},
                   'obj': objective, 'other': {}}

def local_search_swap(users_and_facs_df, travel_dict, users, facs, open_facs, assignment,
                       exp_travelers_dict, cap_dict, cutoff = 0.2, time_limit_while = 1):
    """
    Local search for the BUAP that given an assignment tries to swap user assignments and implements the swap
        if it leads to a better objective function value.
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param open_facs: list of open facilities
    :param assignment: where to start the local search
    :param exp_travelers_dict: dictionary stating the population expected to travel from each user to each facility
    :param cap_dict: dictionary of the facilities' capacities
    :param cutoff: minimum preference to consider reassinging a user
    :param time_limit_while: time limit of the while loop
     :return: Boolean indicating that the assignment is feasible and a dictionary containing assignments, 
        facility utilizations and the objective value.
        If the input assignment is infeasible, the function simply returns False and an empty dictionary.
    """
    slacks, not_satisfied, assigned_users = capacity_constraints_slack(users_and_facs_df, travel_dict,cap_dict,
                                                                        open_facs, assignment)
    if len(not_satisfied)> 0:
        return False, {}
    users_preferred_facs = {i: [j for j in open_facs if travel_dict[i][j]>= cutoff ] for i in users}
    not_utilised = {j:(slacks[j] / cap_dict[j]) for j in open_facs}
    partial_objective = sum(cap_dict[j] * (not_utilised[j]) ** 2 for j in open_facs)
    start_time = time.time()
    improvement = -20
    counter = 0
    while time.time() - start_time < time_limit_while and improvement < -10:
        improvement = 0
        random_facs = open_facs.copy()
        for j in random_facs:
            swapped = False
            for i in assigned_users[j]:
                for j_2 in users_preferred_facs[i]:
                    for i_2 in assigned_users[j_2]:
                        if j not in users_preferred_facs[i_2]:
                            continue
                        is_better, new_slack_1, new_slack_2, change_objective = swap_user_better(slacks, i, i_2, j, j_2,
                                                                                                  cap_dict, 
                                                                                                  exp_travelers_dict)
                        if is_better:
                            assignment[i] = j_2
                            assignment[i_2] = j
                            slacks[j] = new_slack_1
                            slacks[j_2] = new_slack_2
                            assigned_users[j].remove(i)
                            assigned_users[j_2].append(i)
                            assigned_users[j_2].remove(i_2)
                            assigned_users[j].append(i_2)
                            partial_objective += change_objective
                            improvement += change_objective
                            swapped = True
                            break
                    if swapped:
                        break
                if swapped:
                    break
        counter += 1
        print(improvement)
    closed_facs = [j for j in facs if j not in open_facs]
    objective = partial_objective + sum(cap_dict[j] for j in closed_facs)
    return True, {'assignment': assignment, 'utilization': {j: 1-(slacks[j]/cap_dict[j]) for j in open_facs}, 
                  'obj': objective, 'other': {}}

def greedily_assign_users(users_and_facs_df, travel_dict, users, facs, open_facs_input, 
                          exp_travelers_dict, cap_dict, assignment = None, fac_to_close = None):
    """
    Heuristic for the BUAP.
    This is the greedy assign and greedy reassign procedures combined into one procedure for the BUAP.
      If no assignment and fac_to_close is given, this simply computes the assignment with the greedy assign procedure.
      If an aissignment and fac_to_close are given it reassign all the users assigned to fac_to_close greedily,
      as in the greedy reassign procedure.
      Code adapted from https://github.com/schmitt-hub/preferential_access_and_fairness_in_waste_management
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param open_facs: list of open facilities
    :param exp_travelers_dict: dictionary stating the population expected to travel from each user to each facility
    :param cap_dict: dictionary of the facilities' capacities
    :param assignment: a previous assignment
    :param fac_to_close: facility to close and reassign users from, if this and assignment given we do not 
    recompute the assignnment but just fix what needs changing based on fac_to_close closing
    :return: boolean indicating whether a feasible assignment could be constructed;
        a dictionary of assignments, facility utilizations and the objective value
    """

    # initialize
    open_facs = open_facs_input.copy()
    if assignment == None and fac_to_close == None:
        cap_left = cap_dict.copy()
        user_assignment = {}
        unassigned_users = users.copy()
        assigned_users = {j: [] for j in open_facs}
    else:
        user_assignment = assignment.copy()
        cap_left = cap_dict.copy()
        assigned_users = {j: [] for j in open_facs}
        for i, j in user_assignment.items():
            assigned_users[j].append(i)
            cap_left[j] = cap_left[j] - exp_travelers_dict[i][j]
        unassigned_users = assigned_users[fac_to_close]
        open_facs.remove(fac_to_close)
        cap_left[fac_to_close] = cap_dict[fac_to_close]
        assigned_users[fac_to_close] = {}
    while unassigned_users:
        users_not_assignable = True
        most_preferred_users = {j: [] for j in open_facs}
        # match every user with their most preferred facility
        for i in unassigned_users:
            possible_facs = [j for j in open_facs if cap_left[j] >= exp_travelers_dict[i][j]]
            if not possible_facs:
                continue
            most_preferred_fac = possible_facs[np.argmax([travel_dict[i][j] for j in possible_facs])]
            most_preferred_prob = travel_dict[i][most_preferred_fac]
            most_preferred_users[most_preferred_fac].append((i, most_preferred_prob))
            users_not_assignable = False
        # if no user could be assigned in this iteration, return without a new feasible assignment
        if users_not_assignable:
            return 0, {}
        # assign users to their most preferred facility in decreasing rank of their preference to this facility
        for j in most_preferred_users:
            sorted_users = sorted(most_preferred_users[j], key=lambda x: -x[1])
            for (i, prob) in sorted_users:
                if cap_left[j] >= exp_travelers_dict[i][j]:
                    unassigned_users.remove(i)
                    user_assignment[i] = j
                    cap_left[j] -= exp_travelers_dict[i][j]
                    assigned_users[j].append(i)

    utilization = {j: sum(exp_travelers_dict[i][j] for i in assigned_users[j]) /
                 cap_dict[j] for j in open_facs}
    closed_facs = [j for j in facs if j not in open_facs]
    objective = sum(cap_dict[j] * (1 - utilization[j]) ** 2 for j in open_facs) + sum(cap_dict[j] for j in closed_facs)
    return 1, {'assignment': user_assignment, 'utilization': utilization, 'obj': objective, 'other': {}}

def greedy_reassign_open(users_and_facs_df, travel_dict, users, facs, open_facs_input, 
                                            exp_travelers_dict, cap_dict, assignment = None, fac_to_open = None,
                                            depth_greedy_reassignment = 1):
    """
    Greedy assignment adapted to work within open greedy algorithm,
        given an assignment opens a facility, reassigns any unassigned users to it
        and runs a focused local searchs
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param open_facs: list of open facilities
    :param exp_travelers_dict: dictionary stating the population expected to travel from each user to each facility
    :param cap_dict: dictionary of the facilities' capacities
    :param assignment: a previous assignment
    :param fac_to_open: facility to open and reassign users to
    :param depth_greedy_reassignment: depth of the greeddy reassignment within the algorithm; i.e. if it is 1
        we only try reassigning users to fac_to_open, if its 2 we addtionally also try to reassign users to the
        facilities that the users that are now assigned to fac_to_open were assigned to, ...
    :return: boolean indicating whether a feasible assignment could be constructed;
        a dictionary of assignments, facility utilizations and the objective value
    """

    # initialize
    open_facs = open_facs_input.copy()
    cap_left = cap_dict.copy()
    if assignment == None and fac_to_open == None:
        user_assignment = {i: "unassigned" for i in users}
        unassigned_users = users.copy()
    else:
        open_facs.append(fac_to_open)
        user_assignment = assignment.copy()
        for i, j in user_assignment.items():
            if j != "unassigned":
                cap_left[j] = cap_left[j] - exp_travelers_dict[i][j]
        unassigned_users = [i for i in users if assignment[i] == "unassigned"]

    while unassigned_users:
        users_not_assignable = True
        most_preferred_users = {j: [] for j in open_facs}
        # match every user with their most preferred facility
        for i in unassigned_users:
            possible_facs = [j for j in open_facs if cap_left[j] >= exp_travelers_dict[i][j]]
            if not possible_facs:
                continue
            most_preferred_fac = possible_facs[np.argmax([travel_dict[i][j] for j in possible_facs])]
            most_preferred_prob = travel_dict[i][most_preferred_fac]
            most_preferred_users[most_preferred_fac].append((i, most_preferred_prob))
            users_not_assignable = False
        # if no user could be assigned in this iteration, break and return the partial assignment
        if users_not_assignable:
            break
        # assign users to their most preferred facility in decreasing rank of their preference to this facility
        for j in most_preferred_users:
            sorted_users = sorted(most_preferred_users[j], key=lambda x: -x[1])
            for (i, prob) in sorted_users:
                if cap_left[j] >= exp_travelers_dict[i][j]:
                    unassigned_users.remove(i)
                    user_assignment[i] = j
                    cap_left[j] -= exp_travelers_dict[i][j]
    utilization = {j: 1- (cap_left[j] / cap_dict[j]) for j in open_facs}
    closed_facs = [j for j in facs if j not in open_facs]
    objective = sum(cap_dict[j] * (1 - utilization[j]) ** 2 for j in open_facs) + sum(cap_dict[j] for j in closed_facs)
    # improve assignment if we are opening facility that still has space, start with opened facility
    # facilities that had users reassigned in previous iteration are considered if they can take up other users
    if fac_to_open != None and cap_left[fac_to_open] > 0:
        facs_to_consider = [fac_to_open]
        counter = 0
        while counter < depth_greedy_reassignment and len(facs_to_consider) > 0:
            new_facs_to_consider = []
            for fac_to_reassign_to in facs_to_consider:
                users_to_test_reassign = [i for i in users if 
                                          cap_left[fac_to_reassign_to] >= exp_travelers_dict[i][fac_to_reassign_to]
                                          and user_assignment[i] != fac_to_reassign_to]
                users_to_test_reassign.sort(key=lambda i: travel_dict[i][fac_to_reassign_to], reverse=True)
                for i in users_to_test_reassign:
                    original_fac = user_assignment[i]
                    if original_fac == "unassigned" and cap_left[fac_to_reassign_to] >= exp_travelers_dict[i][fac_to_reassign_to]:
                        user_assignment[i] = fac_to_reassign_to
                        cap_left[fac_to_reassign_to] -= exp_travelers_dict[i][fac_to_reassign_to]
                    else: 
                        is_better, change_objective = reassign_user_better(cap_left, i, original_fac,fac_to_reassign_to, 
                                                                           cap_dict, exp_travelers_dict)
                        if is_better:
                            user_assignment[i] = fac_to_reassign_to
                            cap_left[original_fac] += exp_travelers_dict[i][original_fac]
                            cap_left[fac_to_reassign_to] -= exp_travelers_dict[i][fac_to_reassign_to]
                            new_facs_to_consider.append(original_fac)
                            if original_fac not in open_facs:
                                print("FAIL: Original facility not in open facilities - something went wrong.")
            facs_to_consider = list(set(new_facs_to_consider.copy()))
            counter += 1
        utilization = {j: 1- (cap_left[j] / cap_dict[j]) for j in open_facs}
        objective = sum(cap_dict[j] * (1 - utilization[j]) ** 2 for j in open_facs) + sum(cap_dict[j] for j in closed_facs)
   
    return len(unassigned_users) == 0, {'assignment': user_assignment, 'utilization': utilization, 'obj': objective,
                                         'other': {'uassigned_users': unassigned_users}}

def relaxation_rounding(users_and_facs_df, travel_dict, users, facs,
                        cap_factor=1.5, threads=1, tolerance=0.0,cutoff = 0.1,
                        time_limit=20000, open_facs = [], budget_factor = 1.0, 
                        define_u = True, num_consider = -1):
    """
    Heuristic for the BUAP.
        This implements the relaxation rounding algorithm: It solves the relaxation and assigns
        a user to the facility for which they have the highest x_ij in the relaxation.
        Then this assignment is fixed if any capacity constraints are exceeded. 
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param cap_factor: factor by which all capacities are scaled
    :param threads: number of threads used in the relaxation
    :param tolerance: tolerance on the optimality of the solution used in the relaxation
    :param cutoff: preference cutoff used when craeting the relaxation model
        only used if num_consider is -1
    :param time_limit: time limit in seconds for the optimization used in the relaxation
    :param open_facs: list of open facilities for this BUAP instance
    :param budget_factor: the proportion of facilties that are open in this BUAP instance
    :param define_u: boolean indicating whether to explicitely define u in the model
    :param num_consider: number of closest facilities to consider for a user (alternative to cutoff value)
        same as n_r in paper; setting this to -1 means we use a cutoff instead
    :return: Boolean indicating whether a feasible solution was achieved; dictionary containing the results.
    """
    start_time = time.time()
    is_feasible, results = solve_BUAP_model_editable(users_and_facs_df, travel_dict, users, facs,
                        cap_factor, cutoff, open_facs, cap_dict = None, continous = True, 
                        define_u = define_u, num_consider = num_consider, 
                        threads=threads, tolerance=tolerance,time_limit = time_limit)
    if not is_feasible:
        print('The model is infeasible')
        return is_feasible, {}
    assignment = results["solution_details"]["assignment"]
    cap_dict = {j: cap_factor * users_and_facs_df.at[j, 'capacity'] for j in facs}

    objective, assignment, not_utilised= make_assignment_satisfy_capacity_constraints(users_and_facs_df, travel_dict, cap_dict, users, facs, open_facs, assignment)


    if len(assignment) == 0:
        print("Failed to fix capacity constraints")
        return False, results

    end_time = time.time()
    results = {"solution_details":
                    {"assignment": assignment,  "objective_value": objective, "solving_time": end_time - start_time},
                "model_details":
                    {"users": users,"open_facs": open_facs, "facs": facs, "cap_factor": cap_factor, "budget_factor": budget_factor, "cutoff": cutoff,
                    "heuristic": "Relaxation rounding", "define_u": define_u, "num_consider": num_consider
                }
            }

    return True, results  

def run_one_iteration_relaxation_rounding_simple_output(m, users_and_facs_df, travel_dict, cap_dict, users, facs, 
                                                        open_facs = [], new_open_facs = [], new_closed_facs = [],
                                                        threads=1, tolerance=0.0,cutoff = 0.0, num_consider = -1,
                                                        time_limit=1000):
    """
    Relaxation rounding implementation used within the close greedy basic algorithm.
    Finds a solution the BUAP given a relaxed model that just needs swapping some facilities from open to closed
        and/or the reverse. If no model is given it creates the model, before solving the relaxation.
        From the relaxed solution, using rounding and some reassignments a BUAP solution is created.
    :param m: relaxed model, if None this function creates the model
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param cap_dict: dictionary of capacities of the facilities
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param open_facs: open facilities currently
    :param new_open_facs: facilities to now open
    :param new_closed_facs: facilities to now close
    :param threads: number of threads used in the relaxation
    :param tolerance: tolerance on the optimality of the solution used in the relaxation
    :param cutoff: preference cutoff used in relaxation model, only used if num_consider is -1
    :param num_consider: number of most preferred facilities to consider for each user,
        same as n_r in paper; setting this to -1 means we use a cutoff instead
    :param time_limit: time limit in seconds for the optimization used in the relaxation
    :return: boolean indicating if feasible, dictionary of assignment, utilization, objective and new model
    """
    if m == None:
        m = build_BUAP_model_editable(users_and_facs_df, travel_dict, users, facs,
                        cap_factor = None, cutoff = cutoff, open_facs = open_facs.copy(), cap_dict = cap_dict, 
                        continous = True, define_u = True, num_consider = num_consider)
    if len(new_open_facs) > 0 or len(new_closed_facs) > 0:
        m = change_open_facs(m, new_open_facs, new_closed_facs)
    is_feasible, result = optimize_BUAP_model_editable(m, threads = threads, tolerance=tolerance,
                                                       time_limit = time_limit)
    m_new = None
    if not is_feasible:
        print("NOT feasible!")
        print("Rebuilding and resolving model")
        # build with all these as this model is going to be used in the next iteration!
        open_facs_all = open_facs.copy()
        open_facs_all.extend(new_closed_facs)
        m_new = build_BUAP_model_editable(users_and_facs_df, travel_dict, users, facs,cap_factor=None,
                        cutoff=cutoff, open_facs=open_facs_all.copy(), cap_dict = cap_dict, continous = True,
                          define_u = True, num_consider = num_consider)
        m_new = change_open_facs(m_new, [], new_closed_facs)
        is_feasible, result = optimize_BUAP_model_editable(m_new, threads = threads, tolerance=tolerance,
                                                           time_limit = time_limit)
        # if still not feasible return this
        if not is_feasible:
            return is_feasible, {'assignment': {}, 'utilization': {}, 'obj': inf,
                           'other': {"model": m }}
    objective,assignment,not_utilised = make_assignment_satisfy_capacity_constraints(users_and_facs_df, travel_dict, 
                                                                                     cap_dict, users, facs, 
                                                                                     result["solution_details"]["open_facs"],
                                                                                     result["solution_details"]["assignment"])
    if len(assignment) == 0:
        return 0, {"other": {"model": m}}
    assignment_results = {'assignment': assignment, 'utilization': {j: 1 - not_utilised[j] for j in facs}, 'obj': objective,
                           'other': {"model": m if m_new == None else m_new }}
    return True, assignment_results

def run_assignment_method(assignment_method, local_search_method, users_and_facs_df, travel_dict, 
                          users, facs, open_facs, exp_travelers_dict, cap_dict, 
                          options_localsearch = {"cutoff": 0.2, "time_limit_while_localsearch": 1}, 
                          options_relaxation = {}):
    """
    Runs the mentioned BUAP method on the input instance and returns the assignment
    :param assignment_method: the method we use to assign, possible inputs: 
        greedy_assign, relaxation_rounding
    :param local_search_method: local search method used after assignment is use,
        possible inputs: None,local_random_reassign, local_random_swap
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param open_facs: list of open facilities
    :param exp_travelers_dict: dictionary stating the population expected to travel from each user to each facility, U_iP_ij
    :param cap_dict: dictionary of the facilities' capacities
    :param options_localsearch:options for localsearch with fields: cutoff_localsearch and time_limit_while_localsearch
    :param options_relaxation: input needed for the relaxation, has fields:
        model, new_open_facs, new_closed_facs, threads, tolerance, time_limit_relaxation, num_consider_in_relaxation
    :return: boolean indicating whether a feasible assignment could be constructed;
        a dictionary of assignments, facility utilizations, the objective value, 
        and other required for that specific method
    """
    is_feasible_assignment, assignment_results = False, {}
    if assignment_method == "greedy_assign":
        is_feasible_assignment, assignment_results = greedily_assign_users(users_and_facs_df, travel_dict, users,
                                                                            facs, open_facs, exp_travelers_dict, cap_dict)
    elif assignment_method == "relaxation_rounding":
        open_facs_before = open_facs.copy()
        print("right before run one iteration")
        is_feasible_assignment, assignment_results = run_one_iteration_relaxation_rounding_simple_output(
                                                            options_relaxation["model"], 
                                                            users_and_facs_df, travel_dict,
                                                            cap_dict, users, facs, 
                                                            open_facs = open_facs_before, 
                                                            new_open_facs = options_relaxation["new_open_facs"],
                                                            new_closed_facs = options_relaxation["new_closed_facs"],
                                                            threads=options_relaxation["threads"], 
                                                            tolerance=options_relaxation["tolerance"],
                                                            cutoff = 0.0, 
                                                            num_consider = options_relaxation["num_consider_in_relaxation"],
                                                            time_limit=options_relaxation["time_limit_relaxation"])
        options_relaxation["model"] = assignment_results["other"]["model"]
    else:
        print("Invalid assignment method")
        return False, {}
    
    if local_search_method == "local_random_reassign":
        is_feasible_assignment, assignment_results = local_search_reassign(users_and_facs_df, travel_dict, users, 
                                                                           facs, open_facs, 
                                                                           assignment_results["assignment"].copy(), 
                                                                           exp_travelers_dict, cap_dict, 
                                                                           cutoff = options_localsearch["cutoff"], 
                                                                           time_limit_while = options_localsearch["time_limit_while_localsearch"])
    elif local_search_method == "local_random_swap":
        is_feasible_assignment, assignment_results = local_search_swap(users_and_facs_df, travel_dict, users,
                                                                        facs, open_facs, 
                                                                        assignment_results["assignment"].copy(), 
                                                                        exp_travelers_dict, cap_dict, 
                                                                        cutoff = options_localsearch["cutoff"], 
                                                                        time_limit_while = options_localsearch["time_limit_while_localsearch"])
    elif local_search_method != "None":
        print("Invald local search method")
        return False, {}
    return is_feasible_assignment, assignment_results