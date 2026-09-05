import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from environment.environment import VRPTWEnvironment, Customer
from environment.env_model import EnvHyperParams

def generate_env(hp: EnvHyperParams) -> VRPTWEnvironment:
    np.random.seed(hp.random_seed)

    customers = []
    for cid in range(1, hp.num_customers + 1):
        customers.append(
            Customer(
                customer_id=cid,
                e=hp.earliest_arrival,
                l=hp.latest_arrival,
                s=hp.service_duration,
                q=int(np.random.randint(hp.min_demand, hp.max_demand + 1))
            )
        )

    vehicle_capacity = hp.vehicle_capacity
    num_locations = hp.num_customers + 1

    # Generate random coordinates
    coords = np.random.uniform(low=hp.coord_min, high=hp.coord_max, size=(num_locations, 2))

    # Calculate pairwise distances (Euclidean)
    distance_matrix = cdist(coords, coords, metric='euclidean')

    # Scale distances to match travel times
    time_matrix = distance_matrix 
    np.fill_diagonal(time_matrix, 0)

    # Convert to DataFrame
    time_matrix_df = pd.DataFrame(time_matrix, index=range(num_locations), columns=range(num_locations))

    return VRPTWEnvironment(customers, vehicle_capacity, coords, time_matrix_df)