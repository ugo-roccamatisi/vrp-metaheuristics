# multi_agent/blackboard.py

from scheduling.vrptw_model import VRPTWSolution

class Blackboard:
    def __init__(self):
        self.best_solution = None
        self.best_cost = float('inf')

    def update_if_better(self, new_solution: VRPTWSolution, new_cost: float) -> bool:
        if new_cost < self.best_cost:
            self.best_solution = new_solution
            self.best_cost = new_cost
            return True
        return False
