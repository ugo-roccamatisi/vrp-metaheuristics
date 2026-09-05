# scheduling/algorithms/naive_scheduler/hyper_params.py
from pydantic import BaseModel

class HyperParams(BaseModel):
    sort_by_demand: bool = False
    max_travel_time: float = 100.0
