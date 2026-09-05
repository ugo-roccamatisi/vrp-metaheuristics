# verification/verify.py

def verify_solution(env, solution):
    """
    Step-by-step verification of a VRPTW solution, where:
      - solution.routes is a list of Route(vehicle_id, sequence_of_customers)
      - Each route is handled by a 'virtual vehicle' with capacity=env.vehicle_capacity
      - There's no env.vehicles list (unlimited vehicles).

    We check:
      1) No repeated customers
      2) Capacity feasibility
      3) Time-window feasibility
      4) Travel-time validity
      5) Vehicle returns to depot

    Returns:
      (is_feasible, message, steps_log)
    """

    # 1) We'll maintain a set of visited customer IDs
    visited_customers = set()

    # 2) We'll track each step in steps_log for debugging
    steps_log = []

    # 3) For each route, we create a 'virtual vehicle'
    routes = solution.get_routes()  # same as solution.routes
    for route_index, route in enumerate(routes):
        # Virtual vehicle state
        vehicle_id = route_index
        load = env.vehicle_capacity
        current_location = 0  # depot
        accumulated_time = 0.0

        # 4) Traverse each customer in the route
        for step_idx, cust_id in enumerate(route.sequence_of_customers):
            
            print("\n\n route :",route,"\n\n")
            print("\n\n route :",step_idx, cust_id ,"\n\n")

            step_info = {
                "route_index": route_index,
                "vehicle_id": vehicle_id,
                "step_idx": step_idx,
                "from_loc": current_location,
                "to_loc": cust_id,
                "load_before": load,
                "time_before": accumulated_time,
                "feasible": True,
                "reason": "",
            }

            # 4a) Check if this customer was already visited
            if cust_id in visited_customers:
                step_info["feasible"] = False
                step_info["reason"] = f"Customer {cust_id} visited more than once!"
                steps_log.append(step_info)
                return False, step_info["reason"], steps_log
            visited_customers.add(cust_id)

            # 4b) Travel time from current_location -> cust_id

            if cust_id not in env.time_matrix.columns or current_location not in env.time_matrix.index:
                step_info["feasible"] = False
                step_info["reason"] = f"Invalid index in time matrix: from={current_location}, to={cust_id}"
                steps_log.append(step_info)
                return False, step_info["reason"], steps_log

            travel_time = env.time_matrix.loc[current_location, cust_id]
            if travel_time == float('inf') or travel_time < 0:
                step_info["feasible"] = False
                step_info["reason"] = f"No valid path from {current_location} to {cust_id}"
                steps_log.append(step_info)
                return False, step_info["reason"], steps_log

            arrival_time = accumulated_time + travel_time

            # 4c) Check capacity
            cust_obj = next((c for c in env.customers if c.id == cust_id), None)
            if cust_obj is None:
                step_info["feasible"] = False
                step_info["reason"] = f"Customer {cust_id} not found in environment!"
                steps_log.append(step_info)
                return False, step_info["reason"], steps_log

            if cust_obj.q > load:
                step_info["feasible"] = False
                step_info["reason"] = f"Vehicle {vehicle_id} capacity exceeded (need {cust_obj.q}, have {load})."
                steps_log.append(step_info)
                return False, step_info["reason"], steps_log

            # 4d) Time window check
            #     If we arrive earlier than earliest, we wait
            arrival_time = max(arrival_time, cust_obj.e)
            if arrival_time > cust_obj.l:
                step_info["feasible"] = False
                step_info["reason"] = (
                    f"Time window violated at customer {cust_id} "
                    f"(arrival={arrival_time}, latest={cust_obj.l})"
                )
                steps_log.append(step_info)
                return False, step_info["reason"], steps_log

            # Update vehicle state after servicing
            departure_time = arrival_time + cust_obj.s
            accumulated_time = departure_time
            load -= cust_obj.q
            current_location = cust_id

            # 4e) Record step info
            step_info["travel_time"] = travel_time
            step_info["arrival_time"] = arrival_time
            step_info["departure_time"] = departure_time
            step_info["load_after"] = load
            step_info["time_after"] = accumulated_time

            steps_log.append(step_info)

        # 5) Return to depot from the final customer in the route
        if current_location not in env.time_matrix.index:
            step_info = {
                "route_index": route_index,
                "vehicle_id": vehicle_id,
                "step_idx": len(route),
                "from_loc": current_location,
                "to_loc": 0,
                "feasible": False,
                "reason": f"Invalid from_loc {current_location} for time matrix",
            }
            steps_log.append(step_info)
            return False, step_info["reason"], steps_log

        back_time = env.time_matrix.loc[current_location, 0]
        if back_time == float('inf'):
            step_info = {
                "route_index": route_index,
                "vehicle_id": vehicle_id,
                "step_idx": len(route),
                "from_loc": current_location,
                "to_loc": 0,
                "feasible": False,
                "reason": "Cannot return to depot from last customer!",
            }
            steps_log.append(step_info)
            return False, step_info["reason"], steps_log

        accumulated_time += back_time
        # We won't explicitly track load/time once we're back at depot

    # If we finish all routes with no errors, it's feasible
    return True, "Solution is feasible.", steps_log
