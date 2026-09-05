# scheduling/algorithms/sa_scheduler/scheduler.py

import numpy as np
from copy import deepcopy

from environment.environment import VRPTWEnvironment
from scheduling.base_scheduler import BaseScheduler
from scheduling.vrptw_model import VRPTWSolution, Route

# On importe la fonction de calcul de coût déjà existante
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
        Crée une unique route qui contient tous les clients dans l'ordre.
        (Le dépôt 0 n’est pas inclus ici comme 'client'.)
        """
        customers_ids = [c.id for c in env.customers]
        route = Route(vehicle_id=0, sequence_of_customers=customers_ids)
        return VRPTWSolution(routes=[route])

    def _generate_neighbor(self, solution: VRPTWSolution, env: VRPTWEnvironment) -> VRPTWSolution:
        """
        Échange aléatoirement deux clients dans la liste unique de la solution.
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
        À partir d’une séquence linéaire de clients, construit plusieurs routes 
        en respectant la capacité du véhicule.
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
        Construit les routes effectives (selon la capacité) 
        puis utilise la fonction officielle de calcul de coût.
        """
        seq = solution.routes[0].sequence_of_customers
        # On segmente la séquence unique en routes multiples
        multi_route_solution = self._build_routes(seq, env)
        # On calcule le coût avec la fonction existante (w=1000 par défaut)
        return compute_solution_cost(env, multi_route_solution)

    def run(self, env: VRPTWEnvironment) -> VRPTWSolution:
        """
        Exécute le recuit simulé (Simulated Annealing) 
        et renvoie la meilleure solution (multi-route).
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

                # Critère Metropolis
                if delta < 0 or np.random.rand() < np.exp(-delta / current_temperature):
                    solution = new_solution
                    if new_cost < best_cost:
                        best_solution = deepcopy(new_solution)
                        best_cost = new_cost

            current_temperature *= self.alpha
            iteration += 1

        # On reconstruit les routes finales pour retourner la solution multi-route
        final_seq = best_solution.routes[0].sequence_of_customers
        return self._build_routes(final_seq, env)
