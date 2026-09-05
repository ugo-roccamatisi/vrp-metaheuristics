# multi_agent/alg_agent.py
from scheduling.vrptw_model import VRPTWSolution
from environment.environment import VRPTWEnvironment

class AlgAgent:
    def __init__(self, scheduler_class, hyperparams):

        self.scheduler_class = scheduler_class
        self.hyperparams = hyperparams

    def propose_improvement(
        self, 
        env: VRPTWEnvironment, 
        solution: VRPTWSolution
    ) -> VRPTWSolution:

        scheduler = self.scheduler_class(hyperparams=self.hyperparams)
        improved_solution = scheduler.run(env, initial_solution=solution)
        return improved_solution
