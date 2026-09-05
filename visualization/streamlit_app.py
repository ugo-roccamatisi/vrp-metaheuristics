# /home/guillaume/ICO/Fil_Rouge/Meta/visualization/streamlit_app.py

import sys
from pathlib import Path
import os

import streamlit as st
import importlib
import pandas as pd
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px

# Adjust paths so we can import from the project root
current_script_path = Path(__file__).resolve()
root_folder = current_script_path.parent.parent
os.chdir(root_folder)
sys.path.append(str(root_folder))

# --- Import your modules ---
from environment.generate_env import generate_env
from environment.env_model import EnvHyperParams
from environment.environment import VRPTWEnvironment
from scheduling.base_scheduler import BaseScheduler
from scheduling.vrptw_model import VRPTWSolution
from metrics.metrics import compute_solution_cost
from verification.verify import verify_solution
from optimization.optimizer import optimize_hyperparams
import optuna
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


ALGO_DIR = "scheduling/algorithms"


st.set_page_config(layout="wide")
st.title("ICO - VRPTW")


# -----------------------------------------------------------------------------
# 1) Helper: Pydantic -> Streamlit form for fixed values
# -----------------------------------------------------------------------------
from typing import get_origin, get_args, Literal

def pydantic_to_streamlit_form(pydantic_model, prefix=""):

    values = {}
    for field, info in pydantic_model.model_fields.items():
        field_type = info.annotation
        default = info.default if info.default is not None else ""
        key = f"{prefix}_{field}"

        origin = get_origin(field_type)

        if field_type == bool:
            values[field] = st.checkbox(field, value=default, key=key)

        elif field_type == int:
            values[field] = st.number_input(field, value=default, key=key)

        elif field_type == float:
            values[field] = st.number_input(field, value=default, format="%.2f", key=key)

        elif origin is Literal:
            literal_options = get_args(field_type)
            values[field] = st.selectbox(
                label=field,
                options=literal_options,
                index=literal_options.index(default) if default in literal_options else 0,
                key=key
            )

        else:
            # fallback for string or other types
            values[field] = st.text_input(field, value=str(default), key=key)

    return pydantic_model(**values)


# -----------------------------------------------------------------------------
# 2) Hyperparameter Search Definition
# -----------------------------------------------------------------------------
def gather_param_search_definition(pydantic_model, prefix="hp"):
    """
    Gather hyperparameter search definitions with fixed, range, or set options,
    based on default values.
    """
    search_definition = {}

    st.markdown("Configure each hyperparameter search type (fixed, range, or set):")
    for field, info in pydantic_model.model_fields.items():
        default_value = info.default
        field_type = info.annotation

        # Title and basic info
        st.markdown(
            f"**{field}** (default={default_value}, type={getattr(field_type, '__name__', str(field_type))})"
        )

        search_option = st.selectbox(
            f"Optimization type for '{field}'",
            options=["fixed", "range", "set"],
            key=f"{prefix}_option_{field}",
            index=0
        )

        # If field_type is int, we will use int for st.number_input
        # If field_type is float, use float.
        is_int = (field_type == int)

        if search_option == "fixed":
            # Just store the default fixed value
            search_definition[field] = {"type": "fixed", "value": default_value}

        elif search_option == "range" and field_type in [int, float]:
            # Provide consistent types
            # Example offset to avoid negative min or zero step
            offset_low = 10 if is_int else 1.0
            offset_high = 10 if is_int else 1.0
            default_low = (default_value - offset_low)
            default_high = (default_value + offset_high)
            default_step = 1 if is_int else 0.1

            col1, col2, col3 = st.columns(3)

            with col1:
                if is_int:
                    low_val = st.number_input(
                        f"Low bound for '{field}'",
                        value=int(default_low),
                        step=1,
                        format="%d",
                        key=f"{prefix}_low_{field}",
                    )
                else:
                    low_val = st.number_input(
                        f"Low bound for '{field}'",
                        value=float(default_low),
                        step=default_step,
                        format="%.4f",
                        key=f"{prefix}_low_{field}",
                    )

            with col2:
                if is_int:
                    high_val = st.number_input(
                        f"High bound for '{field}'",
                        value=int(default_high),
                        step=1,
                        format="%d",
                        key=f"{prefix}_high_{field}",
                    )
                else:
                    high_val = st.number_input(
                        f"High bound for '{field}'",
                        value=float(default_high),
                        step=default_step,
                        format="%.4f",
                        key=f"{prefix}_high_{field}",
                    )

            with col3:
                if is_int:
                    step_val_user = st.number_input(
                        f"Step for '{field}' (0 => continuous in code)",
                        value=1,
                        step=1,
                        format="%d",
                        key=f"{prefix}_step_{field}",
                    )
                    step_val = None if step_val_user == 0 else step_val_user
                else:
                    step_val_user = st.number_input(
                        f"Step for '{field}' (0 => continuous in code)",
                        value=float(default_step),
                        step=float(default_step),
                        format="%.4f",
                        key=f"{prefix}_step_{field}",
                    )
                    step_val = None if step_val_user == 0 else step_val_user

            search_definition[field] = {
                "type": "range",
                "low": float(low_val) if not is_int else int(low_val),
                "high": float(high_val) if not is_int else int(high_val),
                "step": step_val
            }

        elif search_option == "set":
            cat_str = st.text_input(
                f"Comma-separated values for '{field}'",
                value=str(default_value),
                key=f"{prefix}_values_{field}"
            )
            try:
                raw_list = [x.strip() for x in cat_str.split(",") if x.strip()]
                if is_int:
                    values = [int(x) for x in raw_list] if raw_list else [default_value]
                else:
                    values = [float(x) for x in raw_list] if raw_list else [default_value]
                search_definition[field] = {"type": "set", "values": values}

            except ValueError:
                st.error(
                    f"Invalid input for parameter '{field}'. "
                    "Please provide a comma-separated list of numbers."
                )
                continue

        # Divider after each field
        st.divider()

    return search_definition

# -----------------------------------------------------------------------------
# 3) Environment Generation Part
# -----------------------------------------------------------------------------
st.header("Environment Generation")

with st.expander("Environment Hyperparams", expanded=False):
    env_hp = pydantic_to_streamlit_form(EnvHyperParams, prefix="env")

if st.button("Generate Environment"):
    env = generate_env(env_hp)
    st.session_state["env"] = env

env = st.session_state.get("env", None)

if env:
    # The user wants an open expander with two columns showing environment details
    with st.expander("Environment Summary", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Customers")
            df_cust = pd.DataFrame([
                {"ID": c.id, "Earliest": c.e, "Latest": c.l, "Service": c.s, "Demand": c.q}
                for c in env.customers
            ])
            st.dataframe(df_cust)

            st.subheader("Vehicle Capacity (Unlimited # of vehicles)")
            st.write(env.vehicle_capacity)

            st.subheader("Time Matrix")
            st.dataframe(env.time_matrix)

        with col2:
            st.subheader("Coordinates Graph (Nodes Only)")
            G = nx.from_pandas_adjacency(env.time_matrix, create_using=nx.DiGraph)
            pos = {i: (env.coords[i][0], env.coords[i][1]) for i in range(len(env.coords))}

            fig, ax = plt.subplots(figsize=(5, 5))
            nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_size=500)
            nx.draw_networkx_labels(G, pos=pos, ax=ax, font_size=8)
            ax.set_title("VRPTW Environment (Depot=0)")
            ax.set_axis_off()
            st.pyplot(fig)


# -----------------------------------------------------------------------------
# 4) Algorithm Part
# -----------------------------------------------------------------------------
st.header("Algorithm Part")

if env is None:
    st.warning("Please generate the environment first.")
else:
    # Multi select for the alg type
    algorithms = [d for d in os.listdir(ALGO_DIR) if os.path.isdir(os.path.join(ALGO_DIR, d))]
    selected_algos = st.multiselect("Select Algorithm(s)", algorithms)

    # We'll handle each selected algorithm in sequence
    for selected_algo in selected_algos:
        st.subheader(f"Selected Algorithm: {selected_algo}")

        # Dynamic import
        try:
            scheduler_module = importlib.import_module(f"scheduling.algorithms.{selected_algo}.scheduler")
            hyperparams_module = importlib.import_module(f"scheduling.algorithms.{selected_algo}.hyper_params")
        except Exception as e:
            st.error(f"Could not import algorithm '{selected_algo}'. Error: {e}")
            continue

        AlgoHyperParams = hyperparams_module.HyperParams

        # Expender for the alg HPs
        with st.expander(f"Algorithm Hyperparams - {selected_algo}"):
            algo_hp = pydantic_to_streamlit_form(AlgoHyperParams, prefix=f"algo_{selected_algo}")

        # Instantiate scheduler
        SchedulerClass: BaseScheduler = scheduler_module.Scheduler
        scheduler = SchedulerClass(hyperparams=algo_hp)

        # A button to run the scheduling
        if st.button(f"Run Scheduler - {selected_algo}"):
            solution: VRPTWSolution = scheduler.run(env)
            feasible, msg, steps = verify_solution(env, solution)
            cost = compute_solution_cost(env, solution)

            # Then an expander to visualize results in columns

            st.write("**Routes Visualization**")
            st.write(f"**Cost (C)** = {cost:.2f} = w*K + Σ travel_times")
            for rt in solution.routes:
                st.write(f"Vehicle {rt.vehicle_id}: {rt.sequence_of_customers}")

            with st.expander(f"Results - {selected_algo}", expanded=True):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write("**Verification Log**")
                    if feasible:
                        st.success("Solution is feasible.")
                    else:
                        st.error(f"Infeasible solution: {msg}")
                    st.write(pd.DataFrame(steps))

                with col2:
                
                    # Plot the routes via NetworkX
                    def edges_from_path(path):
                        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

                    G_vis = nx.DiGraph()
                    positions = {idx: (coord[0], coord[1]) for idx, coord in enumerate(env.coords)}
                    nx.draw_networkx_nodes(G_vis, positions, node_size=100)
                    nx.draw_networkx_labels(G_vis, positions, font_size=10)

                    for route_idx, route in enumerate(solution.routes):
                        route_color = list(mcolors.TABLEAU_COLORS.values())[route_idx % len(mcolors.TABLEAU_COLORS)]
                        path = [0] + route.sequence_of_customers + [0]
                        edges = edges_from_path(path)
                        G_vis.add_edges_from(edges)

                        nx.draw_networkx_edges(
                            G_vis,
                            positions,
                            edgelist=edges,
                            edge_color=route_color,
                            arrows=True,
                            arrowsize=20,
                            width=2,
                            label=f"Vehicle {route.vehicle_id}"
                        )

                    plt.title("Routes Visualization (Depot=0)")
                    plt.legend()
                    plt.axis('off')
                    st.pyplot(plt)

                with col3:
                    st.write("**Gantt**")
                    gantt_records = []
                    base_time = pd.Timestamp("2025-01-01")

                    for route_idx, route in enumerate(solution.routes):
                        current_time = 0.0
                        from_loc = 0
                        for cust_id in route.sequence_of_customers:
                            travel_t = env.time_matrix.loc[from_loc, cust_id]
                            arrival_t = current_time + travel_t
                            cust_obj = next((c for c in env.customers if c.id == cust_id), None)
                            if cust_obj:
                                arrival_t = max(arrival_t, cust_obj.e)
                                departure_t = arrival_t + cust_obj.s
                            else:
                                departure_t = arrival_t

                            gantt_records.append({
                                "Vehicle": f"Veh{route_idx}",
                                "Start": base_time + pd.to_timedelta(arrival_t, unit="m"),
                                "Finish": base_time + pd.to_timedelta(departure_t, unit="m"),
                                "Customer": f"C{cust_id}"
                            })
                            current_time = departure_t
                            from_loc = cust_id

                    if gantt_records:
                        df_gantt = pd.DataFrame(gantt_records)
                        fig_gantt = px.timeline(
                            df_gantt,
                            x_start="Start",
                            x_end="Finish",
                            y="Vehicle",
                            color="Customer",
                            title="Schedule Gantt Chart"
                        )
                        fig_gantt.update_yaxes(autorange="reversed")
                        st.plotly_chart(fig_gantt)


# -----------------------------------------------------------------------------
# 5) Hyperparameter Optimization with Optuna (Including Visualizations)
# -----------------------------------------------------------------------------
st.header("Hyperparameter Optimization")

if env is None:
    st.warning("Please generate the environment first.")
else:
    # For the sake of demonstration, we let user pick from the same set of algorithms
    # or (in real usage) you might want only one at a time for optimization
    optimization_algorithms = [d for d in os.listdir(ALGO_DIR) if os.path.isdir(os.path.join(ALGO_DIR, d))]
    selected_opt_algo = st.selectbox("Select Algorithm for HP Optimization", [""] + optimization_algorithms)

    if selected_opt_algo:
        try:
            scheduler_module_opt = importlib.import_module(f"scheduling.algorithms.{selected_opt_algo}.scheduler")
            hyperparams_module_opt = importlib.import_module(f"scheduling.algorithms.{selected_opt_algo}.hyper_params")
            AlgoHyperParamsOpt = hyperparams_module_opt.HyperParams
        except Exception as e:
            st.error(f"Could not import algorithm '{selected_opt_algo}' for optimization. Error: {e}")
            st.stop()

        # Let user define search space
        with st.expander("Define Hyperparameter Search Space", expanded=True):
            search_defs = gather_param_search_definition(AlgoHyperParamsOpt)

        # Number of trials
        n_trials = st.number_input("select n_trial", value=10, step=1)

        # Run Optuna
        if st.button("Run Optuna"):
            if not search_defs:
                st.warning("No hyperparameters selected for optimization.")
            else:
                with st.spinner("Running Optuna..."):
                    SchedulerClassOpt: BaseScheduler = scheduler_module_opt.Scheduler
                    study, best_trial = optimize_hyperparams(
                        env=env,
                        scheduler_class=SchedulerClassOpt,
                        hp_class=AlgoHyperParamsOpt,
                        param_definitions=search_defs,
                        n_trials=n_trials,
                    )

                st.success("Optimization completed!")
                # Display best trial results
                st.subheader("Best Trial Results")
                st.write(f"**Best Cost:** {best_trial.value}")
                st.write("**Best Hyperparameters:**")
                st.json(best_trial.params)

                # Optuna Visualizations
                st.subheader("Hyperparameter Optimization Visualizations")

                fig_history = optuna.visualization.plot_optimization_history(study)
                st.plotly_chart(fig_history, use_container_width=True)

                fig_importances = optuna.visualization.plot_param_importances(study)
                st.plotly_chart(fig_importances)

                fig_parallel = optuna.visualization.plot_parallel_coordinate(study)
                st.plotly_chart(fig_parallel)

                st.subheader("Best Solutions : ")
