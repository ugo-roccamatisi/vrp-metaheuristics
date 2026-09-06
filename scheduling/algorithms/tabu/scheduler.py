# scheduler.py

from scheduling.base_scheduler import BaseScheduler
from scheduling.vrptw_model import VRPTWSolution, Route
from environment.environment import VRPTWEnvironment
from .hyper_params import HyperParams
import random
from collections import deque
from copy import deepcopy
from .list_rules import rule_spt, rule_edd, rule_st, rule_ocr
from metrics.metrics import compute_solution_cost
import numpy as np

class InitialSolution:
    def __init__(self, env: VRPTWEnvironment, num_solutions: int = 5):
        """
        Classe qui génère un pool de solutions initiales pour la recherche tabou en utilisant des règles pré-définies.
        """
        self.env = env
        self.num_solutions = num_solutions
        self.solutions = self._generate_initial_solutions()

    @staticmethod
    def solution_exists(new_solution: VRPTWSolution, solutions: list) -> bool:
        """
        Vérifie si une solution est déjà présente dans une liste de solutions.
        """
        for sol in solutions:
            if new_solution.dict() == sol.dict():
                return True
        return False

    def _generate_initial_solutions(self):
        """
        Génère un ensemble de solutions initiales en utilisant les règles prédéfinies.
        """
        solutions = []
        rules = [rule_spt, rule_edd, rule_st, rule_ocr]

        # Génère des solutions basées sur chaque règle au moins une fois
        for rule in rules:
            solution = rule(self.env)
            if not self.solution_exists(solution, solutions):  # Ajoute seulement si elle est unique
                solutions.append(solution)

        # Génère des solutions aléatoires supplémentaires
        customers_ids = [c.id for c in self.env.customers]
        while len(solutions) < self.num_solutions:
            np.random.shuffle(customers_ids)
            route = Route(vehicle_id=0, sequence_of_customers=customers_ids[:])
            new_solution = VRPTWSolution(routes=[route])
            if not self.solution_exists(new_solution, solutions):  # Vérifie l'unicité
                solutions.append(new_solution)

        return solutions

    def get_random_solution(self) -> VRPTWSolution:
        """
        Retourne une solution initiale aléatoire parmi celles générées.
        """
        return deepcopy(np.random.choice(self.solutions))


class Scheduler(BaseScheduler):
    def __init__(self, hyperparams: HyperParams):
        super().__init__(hyperparams)
        self.tabu_list = deque(maxlen=self.hyperparams.tabu_list_size)

    def run(self, env: VRPTWEnvironment, initial_solutions: list = []) -> VRPTWSolution:
        """
        Exécute l'algorithme de recherche tabou et renvoie la meilleure solution trouvée.
        Si `initial_solutions` est fourni, une solution aléatoire est choisie parmi elles.
        Si `initial_solutions` est vide, une solution est générée par `rule_ocr` par défaut.
        """
        if not initial_solutions:
            # Générer un pool de solutions si aucune n'est fournie
            initial_solutions = InitialSolution(env).solutions
        else:
            # Filtrage des solutions identiques si l'utilisateur en fournit
            unique_solutions = []
            for sol in initial_solutions:
                if not InitialSolution.solution_exists(sol, unique_solutions):
                    unique_solutions.append(sol)
            initial_solutions = unique_solutions

        if not initial_solutions:  # Si malgré tout, aucune solution n'est générée
            initial_solutions = [rule_ocr(env)]  # Utilise OCR par défaut

        # Choisir une solution initiale aléatoire
        current_solution = deepcopy(np.random.choice(initial_solutions))
        best_solution = deepcopy(current_solution)
        best_cost = self.cost(env, best_solution)
        best_iter = 0

        iteration = 0
        while iteration - best_iter < self.hyperparams.max_iterations:
            iteration += 1
            neighborhood = self.neighborhood(current_solution)

            admissible_neighbors = []
            for neighbor in neighborhood:
                if self.is_solution_feasible(env, neighbor) and (neighbor not in self.tabu_list or self.cost(env, neighbor) < best_cost):
                    admissible_neighbors.append(neighbor)

            if not admissible_neighbors:
                break

            next_solution = min(admissible_neighbors, key=lambda sol: self.cost(env, sol))
            next_cost = self.cost(env, next_solution)

            self.tabu_list.append(next_solution)
            current_solution = next_solution

            if next_cost < best_cost:
                best_solution = deepcopy(next_solution)
                best_cost = next_cost
                best_iter = iteration

        return best_solution

    def neighborhood(self, solution: VRPTWSolution) -> list:
        neighbors = []
        if len(solution.routes) < 2:
            # Inter-route swap impossible with a single route: no neighbors,
            # run() then stops cleanly on its own guard.
            return neighbors
        for _ in range(self.hyperparams.neighborhood_size):
            neighbor = deepcopy(solution)
            route1, route2 = random.sample(neighbor.routes, 2)

            if not route1.sequence_of_customers or not route2.sequence_of_customers:
                continue

            idx1 = random.randint(0, len(route1.sequence_of_customers) - 1)
            idx2 = random.randint(0, len(route2.sequence_of_customers) - 1)

            route1.sequence_of_customers[idx1], route2.sequence_of_customers[idx2] = \
                route2.sequence_of_customers[idx2], route1.sequence_of_customers[idx1]

            neighbors.append(neighbor)
        return neighbors

    def cost(self, env: VRPTWEnvironment, solution: VRPTWSolution) -> float:
        return compute_solution_cost(env, solution)

    def is_solution_feasible(self, env: VRPTWEnvironment, solution: VRPTWSolution) -> bool:
        for route in solution.routes:
            demande_totale = 0
            for cid in route.sequence_of_customers:
                c = next((x for x in env.customers if x.id == cid), None)
                if c is None:
                    return False
                demande_totale += c.q
            if demande_totale > env.vehicle_capacity:
                return False
        return True
