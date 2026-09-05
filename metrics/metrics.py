# metrics/metrics.py

def compute_solution_cost(env, solution, w=100.0):
    """
    Computes cost:
      C = w * K + total_travel_time

    where:
      K = number of routes
      total_travel_time sums depot->first_cust + each inter-customer leg + last_cust->depot.
    """

    # Number of routes:
    K = solution.get_number_of_vehicles_used()
    total_time = 0.0

    # For each route in the solution
    for route in solution.get_routes():
        seq = route.sequence_of_customers
        if not seq:
            continue

        # Depot -> first customer
        total_time += env.time_matrix.loc[0, seq[0]]

        # Inter-customer legs
        for i in range(len(seq) - 1):
            total_time += env.time_matrix.loc[seq[i], seq[i + 1]]

        # Last customer -> depot
        total_time += env.time_matrix.loc[seq[-1], 0]

    return w * K + total_time
