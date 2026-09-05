# scheduling/algorithms/tabu_scheduler/hyper_params.py

from pydantic import BaseModel
from typing import Literal


class HyperParams(BaseModel):
    max_iterations: int = 100  # Number of iterations for Tabu Search
    tabu_list_size: int = 20  # Max size of the tabu list
    neighborhood_size: int = 30  # Number of neighbors per iteration
    initialization_rule: Literal["random", "SPT", "EDD", "ST", "OCR"] = "random"
    
    # Default params for the OCR rule
    ocr_current_time: float = 0.0  # Current time for OCR calculation
    
    # (Optional) You can add more rule-specific parameters here if needed
    spt_priority_weight: float = 1.0  # Placeholder for weighting service times
    edd_priority_weight: float = 1.0  # Placeholder for weighting due dates
    st_priority_weight: float = 1.0  # Placeholder for weighting slack times
