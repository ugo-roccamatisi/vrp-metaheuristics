# scheduling/base_scheduler.py
from abc import ABC, abstractmethod
from environment.environment import VRPTWEnvironment
from scheduling.vrptw_model import VRPTWSolution
from pydantic import BaseModel

class BaseScheduler(ABC):
    def __init__(self, hyperparams: BaseModel):
        self.hyperparams = hyperparams

    @abstractmethod
    def run(
        self, 
        env: VRPTWEnvironment, 
        initial_solution: VRPTWSolution = None
    ) -> VRPTWSolution:
        """
        Main entry to execute the scheduling algorithm. 
        - 'initial_solution' is optional; if provided, the scheduler 
          can start from that solution. Otherwise, it calls _generate_initial_solution.
        """
        pass
