# Simulated annealing scheduler

import numpy as np
from copy import deepcopy

from environment.environment import VRPTWEnvironment
from scheduling.base_scheduler import BaseScheduler
from scheduling.vrptw_model import VRPTWSolution, Route

# Reuse the existing cost function
from metrics.metrics import compute_solution_cost

class Scheduler(BaseScheduler):
    def __init__(self, hyperparams):
        self.hyperparams = hyperparams
        self.t = hyperparams.initial_temperature
        self.alpha = hyperparams.cooling_rate
        self.nbiter_cycle = hyperparams.iterations_per_cycle
        self.max_iterations = hyperparams.max_iterations

    def _generate_initial_solution(self, env: VRPTWEnvironment) -> VRPTWSolution:
        """
        Build a single route containing all the clients in order.
        (Depot 0 is not included here as a 'client'.)
        """
        customers_ids = [c.id for c in env.customers]
        route = Route(vehicle_id=0, sequence_of_customers=customers_ids)
        return VRPTWSolution(routes=[route])

    def _generate_neighbor(self, solution: VRPTWSolution, env: VRPTWEnvironment) -> VRPTWSolution:
        """
        Randomly swap two clients in the single list of the solution.
        """
        new_solution = deepcopy(solution)
        seq = new_solution.routes[0].sequence_of_customers

        if len(seq) < 2:
            return new_solution

        i, j = np.random.choice(len(seq), 2, replace=False)
        seq[i], seq[j] = seq[j], seq[i]
        return new_solution

    def _build_routes(self, seq_of_customers, env: VRPTWEnvironment) -> VRPTWSolution:
        """
        From a linear sequence of clients, build several routes
        that respect the vehicle capacity.
        """
        routes = []
        current_route = []
        current_load = 0
        for cid in seq_of_customers:
            cust = next((c for c in env.customers if c.id == cid), None)
            if not cust:
                continue
            if cust.q + current_load <= env.vehicle_capacity:
                current_route.append(cid)
                current_load += cust.q
            else:
                routes.append(Route(vehicle_id=len(routes), sequence_of_customers=current_route))
                current_route = [cid]
                current_load = cust.q
        if current_route:
            routes.append(Route(vehicle_id=len(routes), sequence_of_customers=current_route))

        return VRPTWSolution(routes=routes)

    def _compute_cost(self, solution: VRPTWSolution, env: VRPTWEnvironment) -> float:
        """
        Build the actual routes (according to capacity),
        then use the official cost function.
        """
        seq = solution.routes[0].sequence_of_customers
        # Split the single sequence into multiple routes
        multi_route_solution = self._build_routes(seq, env)
        # Compute the cost with the existing function (w=1000 by default)
        return compute_solution_cost(env, multi_route_solution)

    def run(self, env: VRPTWEnvironment) -> VRPTWSolution:
        """
        Run simulated annealing
        and return the best (multi-route) solution.
        """
        solution = self._generate_initial_solution(env)
        best_solution = deepcopy(solution)
        best_cost = self._compute_cost(solution, env)

        current_temperature = self.t
        iteration = 0

        while iteration < self.max_iterations:
            for _ in range(self.nbiter_cycle):
                new_solution = self._generate_neighbor(solution, env)
                new_cost = self._compute_cost(new_solution, env)
                delta = new_cost - best_cost

                # Metropolis criterion
                if delta < 0 or np.random.rand() < np.exp(-delta / current_temperature):
                    solution = new_solution
                    if new_cost < best_cost:
                        best_solution = deepcopy(new_solution)
                        best_cost = new_cost

            current_temperature *= self.alpha
            iteration += 1

        # Rebuild the final routes to return the multi-route solution
        final_seq = best_solution.routes[0].sequence_of_customers
        return self._build_routes(final_seq, env)
