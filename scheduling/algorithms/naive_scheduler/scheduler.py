# scheduling/algorithms/naive_scheduler/scheduler.py

from scheduling.base_scheduler import BaseScheduler
from scheduling.vrptw_model import VRPTWSolution, Route
from environment.environment import VRPTWEnvironment
from .hyper_params import HyperParams

class Scheduler(BaseScheduler):
    def __init__(self, hyperparams: HyperParams):
        """
        hyperparams can still include any optional fields 
        (sort_by_demand, etc.), but we won't strictly use them here, 
        since the user specifically wants a nearest-neighbor approach.
        """
        super().__init__(hyperparams)

    def run(self, env: VRPTWEnvironment) -> VRPTWSolution:
        """
        Nearest-Neighbor approach with unlimited vehicles:
          - Start from depot (location=0)
          - While there are unvisited customers:
            - Start a new route (new vehicle)
            - While capacity not exceeded:
              - Pick the closest unvisited customer from current location
              - If none fits capacity, end this route
              - Else, add it to route, subtract demand, continue
        Returns multiple routes in VRPTWSolution.
        """

        unvisited = set(c.id for c in env.customers)
        routes = []

        # Convert list[Customer] -> dict for quick lookup
        customer_dict = {c.id: c for c in env.customers}

        # Helper function to get distance
        def distance(from_loc, to_loc):
            return env.time_matrix.loc[from_loc, to_loc]

        while unvisited:
            route_custs = []
            remaining_capacity = env.vehicle_capacity
            current_loc = 0  # depot

            while True:
                # Among unvisited, find the nearest feasible customer
                feasible_customers = []
                for cid in unvisited:
                    cust = customer_dict[cid]
                    if cust.q <= remaining_capacity:
                        dist = distance(current_loc, cid)
                        feasible_customers.append((cid, dist))

                if not feasible_customers:
                    # No feasible customers -> end this route
                    break

                # Pick the one with minimal distance
                cid_next, dist_next = min(feasible_customers, key=lambda x: x[1])

                # Add that customer to the route
                route_custs.append(cid_next)
                unvisited.remove(cid_next)
                remaining_capacity -= customer_dict[cid_next].q
                current_loc = cid_next

            # Done building one route
            routes.append(Route(
                vehicle_id=len(routes),  # e.g. route index as ID
                sequence_of_customers=route_custs
            ))

        # Build the final VRPTWSolution
        return VRPTWSolution(routes=routes)
