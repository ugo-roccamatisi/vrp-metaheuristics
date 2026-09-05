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
├── scheduling/        # VRPTW model, base scheduler and the algorithms
│   └── algorithms/    # tabu search, recuit (SA), ag (GA), naive baseline
├── sma/               # Top-N multi-agent system: agents, blackboard, coordinator
├── optimization/      # Optuna hyper-parameter search
├── environment/       # Instance generation and environment model
├── metrics/           # Cost metrics
├── verification/      # Step-by-step solution verification
├── visualization/     # The Streamlit application
├── report/            # Project report (French)
└── docs/              # Figures
```

The tabular Q-Learning extension (2-opt operators, chained after simulated annealing) is documented in the [project report](report/rapport-ICO.pdf) (French).

## Run it

```bash
conda create -n vrptw_env python=3.9.8
conda activate vrptw_env
pip install -r requirements.txt
cd visualization
streamlit run streamlit_app.py
```

## Team

Group project at Centrale Lille with Lylia Sadoun, Bilal Miali, Adrien Benaroch, Baptiste Neuveux and Guillaume Darlot.

## Gallery

| | |
|---|---|
| ![Multi-agent convergence](docs/vrp-benchmark.png) | ![Live Q-Learning training](docs/vrp-qlearning.png) |

