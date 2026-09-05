# scheduling/algorithms/genetic_scheduler/scheduler.py

import random
import math
import copy
import numpy as np
from typing import List

from sklearn.cluster import KMeans

from scheduling.base_scheduler import BaseScheduler
from scheduling.vrptw_model import VRPTWSolution, Route
from environment.environment import VRPTWEnvironment, Customer
from .hyper_params import HyperParams


class Scheduler(BaseScheduler):
    def __init__(self, hyperparams: HyperParams):
        """
        The GA Scheduler with cluster-first, route-second logic.
        """
        super().__init__(hyperparams)
        # Set the random seed for reproducibility
        print("\n\nHP:",self.hyperparams,"\n\n")
        random.seed(self.hyperparams.random_seed)
        np.random.seed(self.hyperparams.random_seed)

    def run(self, env: VRPTWEnvironment) -> VRPTWSolution:
        """
        Main entry point.

        Steps:
          1) Estimate #clusters = capacity / mean_demand -> run KMeans for initial solution
          2) Build a GA population (each individual is a set of routes/clusters).
          3) Evolve for n_generations:
             a) Selection
             b) Crossover (PMX-like at the "route" level)
             c) Mutation (swap customers between routes if capacity allows)
             d) For each child, do local BnB on each route to get the best ordering.
          4) Return the best individual found as a VRPTWSolution.
        """
        # 1) Basic data
        customers = env.customers
        depot_id = 0
        n_customers = len(customers)
        capacity = env.vehicle_capacity

        if n_customers == 0:
            # No customers => trivial solution
            return VRPTWSolution(routes=[])

        # 2) Perform KMeans to get initial route definitions
        mean_demand = np.mean([c.q for c in customers])
        if mean_demand == 0:
            mean_demand = 1e-9  # avoid divide-by-zero

        approx_k = max(1, int(round(capacity / mean_demand)))

        # Coordinates: ignoring depot => only cluster customers
        cust_coords = np.array([env.coords[c.id] for c in customers])  

        # if n_customers < approx_k => fallback
        k_for_kmeans = min(n_customers, approx_k)

        # Fit KMeans
        if k_for_kmeans > 1:
            kmeans = KMeans(n_clusters=k_for_kmeans, n_init=5, random_state=self.hyperparams.random_seed)
            labels = kmeans.fit_predict(cust_coords)
        else:
            # All customers in one cluster
            labels = np.zeros(n_customers, dtype=int)

        # Build an initial assignment from the KMeans clusters
        initial_routes = self._build_routes_from_labels(labels, customers, capacity)


        # 3) Create the initial population
        population = []
        population.append(initial_routes)
        for _ in range(self.hyperparams.population_size - 1):
            # Maybe random assignment + small perturbation
            random_routes = self._random_cluster_solution(customers, capacity, k_for_kmeans)
            population.append(random_routes)

        # Local improvement (BnB) for each initial solution
        population = [self._local_optimization(sol, env) for sol in population]

        # Evaluate the population
        def cost_func(routes):
            return self._compute_solution_cost(env, routes)

        # 4) GA loop
        best_solution = None
        best_cost = float("inf")

        for gen in range(self.hyperparams.n_generations):
            
            # a) Selection (tournament or rank-based)
            parents = self._selection(population, cost_func)

            # b) Crossover
            offspring = []
            for i in range(0, len(parents), 2):
                p1 = parents[i]
                if i + 1 < len(parents):
                    p2 = parents[i + 1]
                else:
                    p2 = parents[0]  # or random?

                # With probability, crossover => produce two children
                if random.random() < self.hyperparams.crossover_probability:
                    c1, c2 = self._crossover_pmx_routes(p1, p2, env)
                else:
                    c1, c2 = (p1, p2)

                offspring.append(c1)
                offspring.append(c2)

            # c) Mutation
            for i in range(len(offspring)):
                if random.random() < self.hyperparams.mutation_probability:
                    offspring[i] = self._mutation_exchange(offspring[i], env)

            # d) Local optimization on each child
            new_population = []
            for off_sol in offspring:
                improved_sol = self._local_optimization(off_sol, env)
                new_population.append(improved_sol)

            # Evaluate, keep best
            population = new_population
            for sol in population:
                c = cost_func(sol)
                if c < best_cost:
                    best_cost = c
                    best_solution = sol

        # 5) Convert best solution into VRPTWSolution
        final_solution = self._build_vrptw_solution(best_solution)
        return final_solution

    # --------------------------------------------------------------------------
    # Helper: Build routes from KMeans labels
    # --------------------------------------------------------------------------
    def _build_routes_from_labels(self, labels, customers, capacity):
        """
        Given array of labels (size = len(customers)), group them into routes.
        Returns: a list of routes, each route = [customer_id, ...].
        """
        cluster_dict = {}
        for i, lbl in enumerate(labels):
            if lbl not in cluster_dict:
                cluster_dict[lbl] = []
            cluster_dict[lbl].append(customers[i])

        routes = []
        for lbl, cust_list in cluster_dict.items():
            # Summation of demands
            total_demand = sum(c.q for c in cust_list)
            if total_demand <= capacity:
                # OK as one route
                routes.append([c.id for c in cust_list])
            else:
                # We must split it into multiple routes
                # A simple approach: keep greedily filling a route until capacity is reached
                sorted_by_demand = sorted(cust_list, key=lambda c: c.q, reverse=False)
                current_route = []
                current_load = 0
                for c in sorted_by_demand:
                    if c.q + current_load <= capacity:
                        current_route.append(c.id)
                        current_load += c.q
                    else:
                        # Start a new route
                        routes.append(current_route)
                        current_route = [c.id]
                        current_load = c.q
                if current_route:
                    routes.append(current_route)

        return routes

    def _random_cluster_solution(self, customers, capacity, k):
        """
        Builds a random cluster assignment with k clusters, then splits by capacity if needed.
        Returns a list of routes.
        """
        n_customers = len(customers)
        if k < 1:
            k = 1
        labels = np.random.randint(low=0, high=k, size=n_customers)
        return self._build_routes_from_labels(labels, customers, capacity)

    # --------------------------------------------------------------------------
    # GA Operators
    # --------------------------------------------------------------------------
    def _selection(self, population, cost_func, tournament_size=3):
        """
        Simple tournament selection: pick 'tournament_size' solutions at random,
        choose the best. Repeat to get a new parent.
        """
        new_parents = []
        pop_size = len(population)

        for _ in range(pop_size):
            contenders = random.sample(population, min(tournament_size, pop_size))
            best_contender = min(contenders, key=cost_func)
            new_parents.append(best_contender)

        return new_parents

    def _crossover_pmx_routes(self, routes1, routes2, env: VRPTWEnvironment):

        n1 = len(routes1)
        n2 = len(routes2)
        n = min(n1, n2)
        if n == 0:
            return (routes1, routes2)

        # pick k,l
        k = random.randint(0, n - 1)
        l = random.randint(k, n - 1)

        # We store sub-block
        block1 = routes1[k:l+1]
        block2 = routes2[k:l+1]

        # We need to combine them so that each client appears exactly once
        c1_set = set(sum(block1, [])) 
        c2_set = set(sum(block2, []))
        
        missing_custs = []
        for r in routes2:
            for c in r:
                if c not in c1_set:
                    missing_custs.append(c)

        # Now partition missing_custs into capacity-feasible routes
        remainder_child1 = self._build_routes_capacity_based(missing_custs, env.vehicle_capacity, env)

        # Combine block1 + remainder
        new_child1 = block1 + remainder_child1

        # Child2 
        missing_custs2 = []
        for r in routes1:
            for c in r:
                if c not in c2_set:
                    missing_custs2.append(c)
        remainder_child2 = self._build_routes_capacity_based(missing_custs2, env.vehicle_capacity, env)

        new_child2 = block2 + remainder_child2

        return (new_child1, new_child2)

    def _build_routes_capacity_based(self, customer_ids, capacity, env: VRPTWEnvironment):
        """
        Given a list of customer IDs, partition them into routes respecting capacity.
        A simple approach: sorted ascending by demand, fill greedily.
        """
        cust_obj = {c.id: c for c in env.customers}
        sorted_custs = sorted(customer_ids, key=lambda cid: cust_obj[cid].q)

        routes = []
        current_route = []
        current_load = 0
        for cid in sorted_custs:
            dem = cust_obj[cid].q
            if dem + current_load <= capacity:
                current_route.append(cid)
                current_load += dem
            else:
                # start new route
                routes.append(current_route)
                current_route = [cid]
                current_load = dem
        if current_route:
            routes.append(current_route)
        return routes

    def _mutation_exchange(self, routes, env: VRPTWEnvironment):
        """
        Mutation: pick two routes, see if a client can be exchanged (capacity check).
        If feasible, do the swap. Alternatively, pick a random pair of customers and swap them.
        """
        if len(routes) < 2:
            return routes

        routes_copy = copy.deepcopy(routes)

        r1_idx, r2_idx = random.sample(range(len(routes_copy)), 2)
        route1 = routes_copy[r1_idx]
        route2 = routes_copy[r2_idx]

        if (len(route1) == 0 or len(route2) == 0):
            return routes_copy

        cust_obj = {c.id: c for c in env.customers}

        c1_idx = random.randint(0, len(route1)-1)
        c2_idx = random.randint(0, len(route2)-1)
        c1 = route1[c1_idx]
        c2 = route2[c2_idx]

        # Check capacity
        load1_before = sum(cust_obj[x].q for x in route1)
        load2_before = sum(cust_obj[x].q for x in route2)
        load1_after = load1_before - cust_obj[c1].q + cust_obj[c2].q
        load2_after = load2_before - cust_obj[c2].q + cust_obj[c1].q

        if (load1_after <= env.vehicle_capacity) and (load2_after <= env.vehicle_capacity):
            # swap
            route1[c1_idx], route2[c2_idx] = route2[c2_idx], route1[c1_idx]

        return routes_copy

    # --------------------------------------------------------------------------
    # Local Optimization: BnB on each route
    # --------------------------------------------------------------------------
    def _local_optimization(self, routes, env: VRPTWEnvironment):
        """
        For each route in `routes`, run a small Branch & Bound TSP (with or without time windows).
        We'll then replace the route ordering with the best ordering found.
        Return the updated route list.
        """
        new_routes = []
        for rt in routes:
            # run exact TSP on rt
            # This can be big, but let's do a simple backtracking if len(rt) is not too large
            best_order = self._branch_and_bound(rt, env)
            new_routes.append(best_order)
        return new_routes

    def _branch_and_bound(self, route, env: VRPTWEnvironment):
        """
        Solve TSP on the subset `route` (list of customer IDs).
        """

        if not route:
            return route

        best_perm = route
        best_cost = float("inf")

        # We'll do a simple DFS over permutations
        visited = set()
        route_len = len(route)

        def backtrack(current_path, current_cost):
            nonlocal best_cost, best_perm

            if len(current_path) == route_len:
                # finalize cost: last -> depot
                final_cost = current_cost + env.time_matrix.loc[current_path[-1], 0]
                if final_cost < best_cost:
                    best_cost = final_cost
                    best_perm = current_path[:]
                return

            for cid in route:
                if cid not in visited:
                    # cost from current_path[-1] -> cid
                    from_c = current_path[-1]
                    travel = env.time_matrix.loc[from_c, cid]
                    new_cost = current_cost + travel
                    # bounding
                    if new_cost < best_cost:
                        visited.add(cid)
                        current_path.append(cid)
                        backtrack(current_path, new_cost)
                        current_path.pop()
                        visited.remove(cid)

        # Start from depot
        for cid in route:
            visited.clear()
            visited.add(cid)
            cost_0 = env.time_matrix.loc[0, cid]
            backtrack([cid], cost_0)

        return best_perm

    # --------------------------------------------------------------------------
    # Final: Convert a route-list -> VRPTWSolution
    # --------------------------------------------------------------------------
    def _build_vrptw_solution(self, routes: List[List[int]]) -> VRPTWSolution:
        vrp_routes = []
        for i, rt_list in enumerate(routes):
            # i is the vehicle index
            vrp_routes.append(Route(
                vehicle_id=i,
                sequence_of_customers=rt_list
            ))
        return VRPTWSolution(routes=vrp_routes)

    # --------------------------------------------------------------------------
    # Cost function (similar to the existing "compute_solution_cost" logic)
    # --------------------------------------------------------------------------
    def _compute_solution_cost(self, env: VRPTWEnvironment, routes: List[List[int]]) -> float:
        """
        Here we refine the function, to work directly with rotues. This MUST be changed in the future, to work with the cost defined in /metrics
        """
        w = 100.0
        K = len(routes)
        total_time = 0.0
        for rt in routes:
            if not rt:
                continue
            total_time += env.time_matrix.loc[0, rt[0]]  # depot->first
            for i in range(len(rt)-1):
                total_time += env.time_matrix.loc[rt[i], rt[i+1]]
            total_time += env.time_matrix.loc[rt[-1], 0]  # last->depot
        return w * K + total_time

