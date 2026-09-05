import optuna
from scheduling.base_scheduler import BaseScheduler
from environment.environment import VRPTWEnvironment
from metrics.metrics import compute_solution_cost

def optimize_hyperparams(
    env: VRPTWEnvironment,
    scheduler_class,
    hp_class,
    param_definitions: dict,
    n_trials: int = 20,
):

    def objective(trial: optuna.Trial):
        # Build a dictionary of hyperparameters
        hp_dict = {}
        for param_name, param_info in param_definitions.items():
            ptype = param_info.get("type", "range")
            if ptype.lower() == "range":
                low_val = param_info.get("low", 0.0)
                high_val = param_info.get("high", 1.0)
                step_val = param_info.get("step", None)

                hp_dict[param_name] = trial.suggest_float(
                    param_name,
                    low_val,
                    high_val,
                    step=step_val
                )
            elif ptype.lower() == "set":
                values = param_info.get("values", [])
                hp_dict[param_name] = trial.suggest_categorical(param_name, values)
            elif ptype.lower() == "fixed":
                hp_dict[param_name] = param_info.get("value")
            else:
                raise ValueError(f"Unknown hyperparam type '{ptype}' for {param_name}")


        scheduler: BaseScheduler = scheduler_class(hyperparams=hp_class(**hp_dict))
        solution = scheduler.run(env)

        cost = compute_solution_cost(env, solution)
        return cost

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial

    print("Best trial ID:", best_trial.number)
    print("Best hyperparams:", best_trial.params)
    print("Best objective (cost):", best_trial.value)

    return study, best_trial
