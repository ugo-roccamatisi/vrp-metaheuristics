# environment/environment.py

import json
import pandas as pd
from environment.env_model import CustomerModel, EnvironmentModel

class Customer:
    def __init__(self, customer_id, e, l, s, q):
        self.id = customer_id
        self.e = e  # earliest time
        self.l = l  # latest time
        self.s = s  # service time
        self.q = q  # demand

class VRPTWEnvironment:

    def __init__(self, customers, vehicle_capacity, coords, time_matrix):
        self.customers = customers  # list[Customer]
        self.vehicle_capacity = vehicle_capacity
        self.time_matrix = time_matrix  # pandas DataFrame
        self.coords = coords

