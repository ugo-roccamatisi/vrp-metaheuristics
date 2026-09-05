# multi_agent/coordinator.py

from typing import List
from metrics.metrics import compute_solution_cost
from scheduling.vrptw_model import VRPTWSolution
from environment.environment import VRPTWEnvironment
from .blackboard import Blackboard
from .alg_agent import AlgAgent

class Coordinator:
    def __init__(self, env: VRPTWEnvironment, agents: List[AlgAgent], blackboard: Blackboard):
        self.env = env
        self.agents = agents
        self.blackboard = blackboard

    def initialize_solution(self):
        # Example: pick the first agent to generate a solution from scratch
        init_agent = self.agents[0]
        sol = init_agent.propose_improvement(self.env, None)   # solution=None => agent calls _generate_initial_solution
        cost = compute_solution_cost(self.env, sol)
        self.blackboard.update_if_better(sol, cost)

    def run_iteration(self):
        for agent in self.agents:
            # pass the blackboard’s best solution
            current_best_sol = self.blackboard.best_solution
            improved = agent.propose_improvement(self.env, current_best_sol)
            improved_cost = compute_solution_cost(self.env, improved)
            self.blackboard.update_if_better(improved, improved_cost)

    def run_coordinator_loop(self, n_iterations=10):
        if self.blackboard.best_solution is None:
            self.initialize_solution()

        for i in range(n_iterations):
            self.run_iteration()
            print(f"Iteration {i}, best cost={self.blackboard.best_cost}")

        print("Final best cost:", self.blackboard.best_cost)
