from utils.search_tree import SippNode
from utils.map import Map
from utils.sipp import DynamicObstacle, sipp
from utils.wsipp import w_sipp, w_sipp_dublicate_states, w_sipp_with_reexpansions
from utils.focal_sipp import focal_sipp
from utils.heuristics import manhattan_dist
from utils.visualization import *
from utils.map import load_map_from_file, are_connected, random_free_cell, random_start_goal
import random
from utils.visualization import create_animation


def extract_node_path(goal_node: SippNode) -> list[SippNode]:
    path = []
    node = goal_node
    while node is not None:
        path.append(node)
        node = node.parent
    path.reverse()
    return path


def extract_timed_path(goal_node: SippNode) -> list[tuple[int, int, int]]:
    node_path = extract_node_path(goal_node)
    timed_path = []
    
    for idx, node in enumerate(node_path):
        arrival_time = int(node.g)
        
        if idx == 0:
            timed_path.append((arrival_time, node.i, node.j))
        else:
            prev_node = node_path[idx - 1]
            prev_time = int(prev_node.g)
            move_duration = 1
            departure_time = arrival_time - move_duration
            
            if departure_time > prev_time:
                for t in range(prev_time + 1, departure_time + 1):
                    timed_path.append((t, prev_node.i, prev_node.j))
            
            timed_path.append((arrival_time, node.i, node.j))
    
    return timed_path


def continue_obstacle(
    obstacle: DynamicObstacle,
    new_segment: list[tuple[int, int, int]]
):
    last_time = obstacle.path[-1][0]
    if len(new_segment) > 0 and new_segment[0][1:] != obstacle.path[-1][1:]:
        print(f"некорректное продолжение пути:(")
    
    shifted = [
        (t + last_time, i, j)
        for t, i, j in new_segment[1:]
    ]
    obstacle.path.extend(shifted)


def plan_obstacle_path(
    task_map: Map,
    start: tuple[int, int],
    goal: tuple[int, int],
    obstacles: list[DynamicObstacle],
    heuristic_func: callable,
    start_time: int = 0,
    w_range=(1.1, 15.0),
    max_tries=5
) -> list[tuple[int, int, int]] | None:
    shifted_obstacles = []
    for obs in obstacles:
        shifted_path = [(t - start_time, i, j) for t, i, j in obs.path if t >= start_time]
        if shifted_path:
            shifted_obstacles.append(DynamicObstacle(shifted_path))
    
    for _ in range(max_tries):
        w = random.uniform(*w_range)
        success, goal_node, *_ = w_sipp_with_reexpansions(
            task_map,
            start[0], start[1],
            goal[0], goal[1],
            shifted_obstacles,
            heuristic_func,
            w
        )
        if success:
            return extract_timed_path(goal_node)
    return None


def generate_dynamic_obstacles(
    task_map: Map,
    heuristic_func: callable,
    max_obstacles: int,
    p_continue: float,
    time_horizon: int
) -> list[DynamicObstacle]:
    obstacles = []
    
    for iter in range(max_obstacles):
        if iter % 10 == 0:
            print(f"Iteration {iter}/{max_obstacles}")

        continue_old = obstacles and random.random() < p_continue
        
        if continue_old:
            obs = random.choice(obstacles)
            last_entry = obs.path[-1]
            start = (last_entry[1], last_entry[2])
            start_time = last_entry[0]
        else:
            obs = None
            start = random_free_cell(task_map)
            start_time = 0
        
        for _ in range(10):
            goal = random_free_cell(task_map)
            if goal != start and are_connected(task_map, start, goal):
                break
        else:
            continue
        
        path = plan_obstacle_path(
            task_map,
            start,
            goal,
            obstacles,
            heuristic_func,
            start_time=start_time
        )
        
        if path is None:
            continue
        
        if obs is None:
            obstacles.append(DynamicObstacle(path))
        else:
            continue_obstacle(obs, path)
    
    # for obs in obstacles:
    #     extend_obstacle_to_horizon(obs, time_horizon)
    
    print(f"Total obstacles: {len(obstacles)}")
    return obstacles


def save_dynamic_obstacles(
    obstacles: list[DynamicObstacle],
    filename: str
):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"obstacles {len(obstacles)}\n")
        for idx, obs in enumerate(obstacles):
            f.write(f"obstacle {idx}\n")
            for t, i, j in obs.path:
                f.write(f"{t} {i} {j}\n")
            f.write("end\n")


def load_dynamic_obstacles(
    filename: str
) -> list[DynamicObstacle]:
    obstacles: list[DynamicObstacle] = []
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    idx = 0
    n = len(lines)
    if idx < n and lines[idx].startswith("obstacles"):
        idx += 1
    
    while idx < n:
        if lines[idx].startswith("obstacle"):
            idx += 1
        path: list[tuple[int, int, int]] = []
        
        while idx < n and lines[idx] != "end":
            parts = lines[idx].split()
            t = int(parts[0])
            i = int(parts[1])
            j = int(parts[2])
            path.append((t, i, j))
            idx += 1
        
        if path:
            obstacles.append(DynamicObstacle(path))
        idx += 1
    
    print(f"Loaded {len(obstacles)} obstacles")
    return obstacles


if __name__ == '__main__':
    map_name = "AR0016SR"
    task_map = load_map_from_file(f"data/maps/{map_name}.map")
    
    dynamic_obstacles = generate_dynamic_obstacles(
        task_map=task_map,
        heuristic_func=manhattan_dist,
        max_obstacles=750,
        p_continue=0.8,
        time_horizon=2000
    )
    
    save_dynamic_obstacles(
        dynamic_obstacles,
        filename=f"out/dynamic_obstacles_{map_name}.txt"
    )
    
    dynamic_obstacles = load_dynamic_obstacles(
        f"out/dynamic_obstacles_{map_name}.txt"
    )
    
    start_i, start_j, goal_i, goal_j = random_start_goal(task_map)
    
    print(f"Planning path from ({start_i}, {start_j}) to ({goal_i}, {goal_j})")
    
    success, goal_node, *_ = sipp(
        task_map,
        start_i,
        start_j,
        goal_i,
        goal_j,
        dynamic_obstacles,
        heuristic_func=manhattan_dist,
        allow_reexpansions=False
    )
    
    if not success:
        print("Path not found")
        exit()
    
    agent_path = extract_node_path(goal_node)
    print(f"Path found! Length: {len(agent_path)}, arrival time: {agent_path[-1].g}")
    
    create_animation(
        filename=f"out/generated_dynamic_obstacles_{map_name}.gif",
        grid_map=task_map,
        start=agent_path[0],
        goal=agent_path[-1],
        path=agent_path,
        dynamic_obstacles=dynamic_obstacles,
        animation_step=0.3,
        scale=10
    )