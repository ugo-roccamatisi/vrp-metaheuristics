# scheduling/vrptw_model.py
from pydantic import BaseModel
from typing import List

class Route(BaseModel):
    vehicle_id: int
    sequence_of_customers: List[int]

class VRPTWSolution(BaseModel):
    routes: List[Route]

    def get_routes(self, env=None):
        return self.routes

    def get_number_of_vehicles_used(self) -> int:
        return len(self.routes)
