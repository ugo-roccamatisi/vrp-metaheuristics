# scheduling/algorithms/tabu/list_rules.py
from environment.environment import VRPTWEnvironment
from scheduling.vrptw_model import VRPTWSolution, Route

def rule_spt(env: VRPTWEnvironment) -> VRPTWSolution:
    customers = sorted(env.customers, key=lambda c: c.s)
    return distribute_customers(env, customers)

def rule_edd(env: VRPTWEnvironment) -> VRPTWSolution:
    customers = sorted(env.customers, key=lambda c: c.l)
    return distribute_customers(env, customers)

def rule_st(env: VRPTWEnvironment) -> VRPTWSolution:
    customers = sorted(env.customers, key=lambda c: (c.l - c.s))
    return distribute_customers(env, customers)

def rule_ocr(env: VRPTWEnvironment, current_time: float = 0) -> VRPTWSolution:
    customers = sorted(env.customers, key=lambda c: (c.l - current_time) / c.s)
    return distribute_customers(env, customers)

def distribute_customers(env: VRPTWEnvironment, customers) -> VRPTWSolution:
    routes = []
    vehicle_capacity = env.vehicle_capacity
    route = []
    load = 0
    vehicle_id = 0

    for cust in customers:
        if cust.q > env.vehicle_capacity:
            raise ValueError(f"Customer {cust.id} demand exceeds vehicle capacity.")
        if load := sum(env.customers[cid-1].q for cid in route) + cust.q <= env.vehicle_capacity:
            route.append(cust.id)
        else:
            routes.append(Route(vehicle_id=vehicle_id, sequence_of_customers=route))
            vehicle_id += 1
            route = [cust.id]
    if route:
        routes.append(Route(vehicle_id=vehicle_id, sequence_of_customers=route))
    return VRPTWSolution(routes=routes)
