# Exemple de fichier : scheduling/algorithms/sa_scheduler/hyper_params.py

from pydantic import BaseModel

class HyperParams(BaseModel):
    initial_temperature: float = 100.0
    cooling_rate: float = 0.95
    iterations_per_cycle: int = 50
    max_iterations: int = 1000
