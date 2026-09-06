# Vehicle routing: metaheuristics, multi-agent systems and Q-Learning

The NP-hard vehicle routing problem (VRP), attacked three ways and benchmarked fairly: classic metaheuristics (tabu search, simulated annealing, genetic algorithm), a Top-N loop multi-agent system, and tabular Q-Learning with 2-opt operators. Group project (6 students), Collaborative Intelligence elective at Centrale Lille.

![Routes optimized by tabu search](docs/vrp-routes.png)

## Highlights

- **Shared, reproducible benchmark** with Optuna-tuned hyper-parameters for every algorithm, on 40 and 100 client instances.
- The multi-agent system reduces the initial pool cost by **7 to 26%** depending on agent combinations.
- **Winning strategy: simulated annealing then tabular Q-Learning (2-opt), about 13% better than the best standalone metaheuristic** on the 100-client instance (9,906 vs 11,493).
- Interactive Streamlit demo: live Q-Learning training, step-by-step solution verification, route maps.

## Repository layout

```text
.
├── scheduling/
│   ├── vrptw_model.py             # The VRPTW model (clients, vehicles, time windows, costs)
│   ├── base_scheduler.py          # Common interface all algorithms implement
│   └── algorithms/
│       ├── tabu/                  # Tabu search (scheduler, hyper_params, list_rules)
│       ├── recuit/                # Simulated annealing (scheduler, hyper_params)
│       ├── ag/                    # Genetic algorithm (scheduler, hyper_params)
│       └── naive_scheduler/       # Naive baseline (scheduler, hyper_params)
├── sma/
│   ├── run_agents.py              # Entry point of the multi-agent system
│   ├── alg_agent.py               # One agent wrapping one algorithm
│   ├── blackboard.py              # Shared solution pool
│   └── coordinator.py             # Top-N loop coordination
├── optimization/optimizer.py      # Optuna hyper-parameter search
├── environment/                   # Instance generation and environment model
├── metrics/metrics.py             # Cost metrics
├── verification/verify.py         # Step-by-step solution verification
├── visualization/streamlit_app.py # The Streamlit application
├── report/rapport-ICO.pdf         # Project report (French)
└── docs/                          # Figures
```

The tabular Q-Learning extension (2-opt operators, chained after simulated annealing) is documented in the [project report](report/rapport-ICO.pdf) (French).

## Run it

```bash
pip install -r requirements.txt
streamlit run visualization/streamlit_app.py
```

The Streamlit app lets you generate instances, run and compare the algorithms, watch the Q-Learning agent train, and inspect every solution step by step.

## Team

Group project at Centrale Lille with Lylia Sadoun, Bilal Miali, Adrien Benaroch, Baptiste Neuveux and Guillaume Darlot.

## Gallery

| | |
|---|---|
| ![Multi-agent convergence](docs/vrp-benchmark.png) | ![Live Q-Learning training](docs/vrp-qlearning.png) |
