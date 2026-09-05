# environment/env_model.py
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional
import numpy as np 
import pandas as pd 

class CustomerModel(BaseModel):
    id: int
    earliest_arrival: float
    latest_arrival: float
    service_duration: float
    demand: int

class EnvironmentModel(BaseModel):
    customers: List[CustomerModel]
    vehicle_capacity: int
    coords: List[Tuple[float, float]]
    time_matrix: List[List[float]]

class EnvHyperParams(BaseModel):
    num_customers: int = 10
    vehicle_capacity: int = 15
    earliest_arrival: float = 0.0
    latest_arrival: float = 1000
    service_duration: float = 5.0
    min_demand: int = 1
    max_demand: int = 5
    random_seed: Optional[int] = 42
    coord_min : int = 20 
    coord_max : int = -20
