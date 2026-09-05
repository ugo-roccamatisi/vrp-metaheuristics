from pydantic import BaseModel

class HyperParams(BaseModel):
    population_size: int = 50
    n_generations: int = 50
    crossover_probability: float = 0.8
    mutation_probability: float = 0.1
    random_seed: int = 42