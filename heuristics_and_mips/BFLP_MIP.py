"""
Function for solving the BFLP MIP
"""
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.core.base.constraint import Constraint
import time
from utils import *

def cap_init(m, j):
    return m.users_and_facs_df.at[j, 'capacity'] * m.cap_factor.value

def exp_travelers_init(m, i, j):
    return m.users_and_facs_df.at[i, 'population'] * m.travel_dict[i][j]

def obj_expression(m):
    return sum(m.cap[j] * (1 - m.u[j]) ** 2 for j in m.facs)

def define_utilization(m,j):
    return m.u[j] == sum(m.exp_travelers[(i, j)] * m.x[(i, j)] for (i, j2) in m.travel_pairs if j2 == j) / m.cap[j]


def assign_to_one_cstr(m, i):
    eligible_js = [j for (i2, j) in m.travel_pairs if i2 == i]
    if not eligible_js:
        return Constraint.Skip
        
    expr = sum(m.x[i, j] for j in eligible_js)
    
    if m.strict_assign_to_one:
        return expr == 1
    return expr <= 1
    
def assign_to_open_cstr(m, i, j):
    return m.y[j] >= m.x[(i, j)]

def budget_cstr(m):
    return pyo.summation(m.y) <= m.budget

def build_BFLP_model(users_and_facs_df, travel_dict, users, facs, budget_factor = 1.0,
                        cap_factor = 1.5, cutoff = 0.2, strict_assign_to_one = True,
                        cap_dict = None, continous = False, num_consider = -1):
    """
    Build the BFLP model
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param budget_factor: proportion of facilities to open
    :param cap_factor: factor by which all capacities are scaled
    :param cutoff: minimum preference required for facility user pair to be considered;
      only used if num_consider is -1
    :param strict_assign_to_one: boolean indicating whether each user has to be assigned;
      i.e. whether sum_j x_ij == 1 (true) or sum_j x_ij <= 1 is a constraint
    :param cap_dict: optional dictionary of all the capacities; if not provided (i.e. None) this is computed
      from the other input data  
    :param continous: boolean indicating whether to solve this model continously
    :param num_consider: number of closest facilities to consider for a user out of the open facilities, 
      same as n_r in paper; setting this to -1 means we use a cutoff instead
    :return: the BFLP built model
    """
    print('Build BFLP model...')
    start = time.time()
    m = pyo.ConcreteModel()
    m.users_and_facs_df = users_and_facs_df
    m.travel_dict = travel_dict

    # declare sets, variables and parameters
    m.users = pyo.Set(initialize=users)
    m.facs = pyo.Set(initialize=facs)
    m.cutoff = pyo.Param(initialize=cutoff)
    m.num_consider = pyo.Param(initialize=num_consider)
    m.budget_factor = pyo.Param(initialize=budget_factor)
    m.budget = pyo.Param(initialize=round(m.budget_factor * len(list(m.facs))))
    if num_consider == -1:
        m.travel_pairs = pyo.Set(initialize=[(i, j) for i in m.users for j in m.facs
                                         if m.travel_dict[i][j] > m.cutoff.value])
    else:
        travel_pairs = [(i, j) for i in m.users for j in sorted(m.facs, key=lambda j_1: m.travel_dict[i][j_1], reverse=True)[0:num_consider]]
        travel_pairs = list(set(travel_pairs))
        m.travel_pairs = pyo.Set(initialize=travel_pairs)
    if cap_dict == None:
        m.cap_factor = pyo.Param(initialize=cap_factor)
        m.cap = pyo.Param(m.facs, initialize=cap_init)
    else:
        m.cap_factor = pyo.Param(initialize=cap_dict[facs[0]]/m.users_and_facs_df.at[facs[0], 'capacity'] )
        m.cap = pyo.Param(m.facs, initialize=lambda m, j: cap_dict[j])
    m.exp_travelers = pyo.Param(m.travel_pairs, initialize=exp_travelers_init)
    m.strict_assign_to_one = pyo.Param(initialize=strict_assign_to_one)
    m.continous = pyo.Param(initialize=continous)

    # variables for facilities open
    m.y = pyo.Var(m.facs, bounds=(0, 1))
    # variables for assignment
    if m.continous:
        m.x = pyo.Var(m.travel_pairs, bounds=(0,1))
    else:
        m.x = pyo.Var(m.travel_pairs, within=pyo.Binary)


    m.u = pyo.Var(m.facs, bounds=(0, 1))

    # constraints and objective
    m.define_utilization = pyo.Constraint(m.facs, rule=define_utilization)
    m.assign_to_one_cstr = pyo.Constraint(m.users, rule=assign_to_one_cstr)
    m.assign_to_open_cstr = pyo.Constraint(m.travel_pairs, rule=assign_to_open_cstr)
    m.budget_cstr = pyo.Constraint(rule=budget_cstr)
    m.obj = pyo.Objective(rule=obj_expression, sense=pyo.minimize)
    print('setup complete in ', time.time() - start, 'seconds')

    return m


def optimize_BFLP_model(m, threads=1, tolerance=0.0,time_limit = 20000):
    """
    Solve the BFLP model
    :param m: the model to be optimized
    :param threads: number of threads
    :param tolerance: tolerance on the optimality of the solution
    :param time_limit: how long the model is allowed to run for, 
      this does not include the time to build the model
    :return: boolean indicating whether the model is feasible and results dictionary
    """
    print('Optimize model...')
    opt = pyo.SolverFactory("gurobi")
    opt.options["threads"] = threads
    opt.options["MIPGap"] = tolerance
    opt.options["NodefileStart"] = 0.5
    opt.options["Time_limit"] = time_limit
    opt.options["NodeMethod"] = 2
    opt.options["OptimalityTol"] = 1e-4
    opt.options["FeasibilityTol"] = 1e-4
    # solve with enabling logging of solver
    res = opt.solve(m, tee=True)

    if res.solver.termination_condition == TerminationCondition.infeasible:
        return False, {}

    # create dictionary for the results after postprocessing
    assignment = {}
    results_x = {i : {} for i in m.users}
    open_facs = []
    results_y = {j: 0 for j in m.facs}
    for j in m.facs:
        if m.y[j].value > 1e-4:
            open_facs.append(j)
            results_y[j] = 1
    # assign each user to facility with highest x_i,j
    if m.continous:
        for (i,j) in m.travel_pairs:
            results_x[i][j] = min(m.x[(i,j)].value,1.0)
        assignment =  {i : max(results_x[i], key=results_x[i].get) for i in m.users}
    else: 
        for (i,j) in m.travel_pairs:
            results_x[i][j] = round(m.x[(i,j)].value,2)
        for i in m.users:
            for j in m.facs:
                if (i, j) in m.travel_pairs and m.x[(i, j)].value > 1e-4:
                    assignment[i] = j
    results = {"solution_details":
                {"assignment": assignment, "open_facs": open_facs, "objective_value": pyo.value(m.obj),
                    "lower_bound": res['Problem'][0]['Lower bound'], "solving_time": res.Solver[0]['Time']},
            "model_details":
                {"users": list(m.users), "facs": list(m.facs), "cap_factor": m.cap_factor.value,
                    "budget_factor": m.budget_factor.value, "cutoff": m.cutoff.value,
                    "num_consider_in_model": m.num_consider.value,"strict_assign_to_one": m.strict_assign_to_one.value, 
                    "tolerance": tolerance, "time_limit": time_limit}
            }
    return True, results

def run_BFLP_model(users_and_facs_df, travel_dict, users, facs, budget_factor = 1.0,
                        cap_factor = 1.5, cutoff = 0.2, strict_assign_to_one = True,
                        cap_dict = None, continous = False, num_consider = -1, 
                        threads=1, tolerance=0.0,time_limit = 20000, output_filename = "basic_model.json"):
    """
    Build and solve the BFLP model
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param travel_dict: dictionary of the travel probabilities from each user to each facility
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :param budget_factor: proportion of facilities to open
    :param cap_factor: factor by which all capacities are scaled
    :param cutoff: minimum preference required for facility user pair to be considered;
      only used if num_consider is -1
    :param strict_assign_to_one: boolean indicating whether each user has to be assigned;
      i.e. whether sum_j x_ij == 1 (true) or sum_j x_ij <= 1 is a constraint
    :param cap_dict: optional dictionary of all the capacities; if not provided (i.e. None) this is computed
      from the other input data  
    :param continous: boolean indicating whether to solve this model continously
    :param num_consider: number of closest facilities to consider for a user out of the open facilities, 
      same as n_r in paper; setting this to -1 means we use a cutoff instead
    :param threads: number of threads
    :param tolerance: tolerance on the optimality of the solution
    :param time_limit: how long the model is allowed to run for, 
      this does not include the time to build the model
    :param output_filename: the file name of where the results of this model should be stored
      within the folder own_results
    :return: boolean indicating whether the model is feasible and results dictionary
    """
    start_time = time.time()
    model = build_BFLP_model(users_and_facs_df, travel_dict, users, facs, budget_factor, cap_factor,
                              cutoff, strict_assign_to_one, cap_dict, continous, num_consider)
    is_feasible, results = optimize_BFLP_model(model, threads, tolerance,time_limit)
    results["solution_details"]["solving_time"] = time.time() - start_time
    write_results_list([results], output_filename)
    return is_feasible, results
    