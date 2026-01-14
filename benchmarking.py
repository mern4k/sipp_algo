import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Callable, Tuple, Any, NamedTuple
import os
from generate_map import load_map_from_file, load_dynamic_obstacles, random_start_goal, manhattan_dist
from utils.sipp import sipp
from utils.wsipp import w_sipp, w_sipp_with_reexpansions, w_sipp_dublicate_states
from utils.focal_sipp import focal_sipp, get_heuristic

class AlgoDefinition(NamedTuple):
    name: str
    func: Callable
    needs_heuristic: bool = False

def collect_metrics_multi_w(
    map_name: str,
    algorithms: List[AlgoDefinition],
    weights: List[float],
    num_tasks: int = 30,
) -> pl.DataFrame:
    results = []
    map_path = f"data/maps/{map_name}.map"        
    grid = load_map_from_file(map_path)
    obs_path = f"out/dynamic_obstacles_{map_name}.txt"
    if not os.path.exists(obs_path):
        obs_path = "out/dynamic_obstacles_arena.txt"
    print(f"Map: {map_name} | Obstacles: {obs_path}")
    dynamic_obstacles = load_dynamic_obstacles(obs_path)
    tasks = []
    for _ in range(num_tasks):
        tasks.append(random_start_goal(grid))

    for _, (s_i, s_j, g_i, g_j) in enumerate(tqdm(tasks, desc=f"Testing {map_name}")):
        true_dist = get_heuristic(grid, g_i, g_j)
        sipp_found, sipp_node, sipp_steps, _, _, _ = sipp(
            grid, s_i, s_j, g_i, g_j, dynamic_obstacles, manhattan_dist, allow_reexpansions=False
        )
        if not sipp_found:
            continue
        sipp_cost = sipp_node.g
        for w in weights:
            for algo in algorithms:
                if algo.needs_heuristic:
                    args = (w, true_dist)
                else:
                    args = (w,)
                found, node, steps, size, open, closed = algo.func(
                    grid, s_i, s_j, g_i, g_j, 
                    dynamic_obstacles, manhattan_dist, *args
                )
                norm_steps = (steps / sipp_steps)
                cost = node.g
                opt_ratio = (cost / sipp_cost) if sipp_cost > 0 else None
                results.append({
                    "map": map_name,
                    "algorithm": algo.name,
                    "weight": w,
                    "steps": steps,
                    "normalized_steps": norm_steps,
                    "optimality_ratio": opt_ratio,
                    "success": found
                })

    return pl.DataFrame(results)

def plot_expansions_by_weight(df: pl.DataFrame):
    pdf = df.drop_nulls(subset=["normalized_steps"]).to_pandas()
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))
    sns.boxplot(
        data=pdf,
        x="weight",         
        y="normalized_steps", 
        hue="algorithm",     
        palette="Set2",
        linewidth=1.2
    )

    plt.axhline(1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.title("ROOM: Normalized node expansions", fontsize=16)
    plt.xlabel("Weight (w)", fontsize=13)
    plt.ylabel("Number of expanded nodes compared to SIPP", fontsize=13)
    plt.legend(title="Algorithm", loc='upper right')
    plt.tight_layout()
    filename = "out/boxplot_expansions_w.png"
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.show()

def plot_suboptimality_by_weight(df: pl.DataFrame):
    pdf = df.drop_nulls(subset=["optimality_ratio"]).to_pandas()
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))
    sns.boxplot(
        data=pdf,
        x="weight",          
        y="optimality_ratio", 
        hue="algorithm",     
        palette="Set3",      
        linewidth=1.2
    )

    #plt.axhline(1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.title("ROOM: Path quality", fontsize=16)
    plt.xlabel("Weight (w)", fontsize=13)
    plt.ylabel("Path suboptimality (compared to SIPP)", fontsize=13)
    plt.legend(title="Algorithm", loc='upper left')
    plt.tight_layout()
    filename = "out/boxplot_suboptimality_w.png"
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.show()

if __name__ == '__main__':
    map = "8room_000"
    weights_list = [1.1, 1.25, 1.5, 2.0, 4.0]
    algos = [
        AlgoDefinition("W-SIPP Dupl", w_sipp_dublicate_states),
        AlgoDefinition("W-SIPP Re-exp", w_sipp_with_reexpansions),
        AlgoDefinition("Focal SIPP", focal_sipp, needs_heuristic=True)
    ]
    file = "out/metrics_w.csv"
    recalc = True

    if not recalc and os.path.exists(file):
        print(f"Loading from {file}...")
        df_results = pl.read_csv(file)
    else:
        print(f"Calculating for weights: {weights_list}...")
        df_results = collect_metrics_multi_w(
            map_name=map,
            algorithms=algos,
            weights=weights_list,
            num_tasks=50
        )
        os.makedirs("out", exist_ok=True)
        df_results.write_csv(file)
    plot_expansions_by_weight(df_results)
    plot_suboptimality_by_weight(df_results)