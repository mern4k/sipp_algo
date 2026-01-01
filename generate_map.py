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
    path = [(int(node.g), node.i, node.j) for node in extract_node_path(goal_node)]
    return path


def continue_obstacle(
    obstacle: DynamicObstacle,
    new_segment: list[tuple[int, int, int]]
):
    last_time = obstacle.path[-1][0]
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
    w_range=(1.1, 15.0),
    max_tries=5
) -> list[tuple[int, int, int]] | None:
    for _ in range(max_tries):
        w = random.uniform(*w_range)
        success, goal_node, *_ = w_sipp_with_reexpansions(
            task_map,
            start[0], start[1],
            goal[0], goal[1],
            obstacles,
            heuristic_func,
            w
        )
        if success:
            return extract_timed_path(goal_node)
    return None


def generate_dynamic_obstacles(
    task_map: Map,
    heuristic_func: callable,
    max_obstacles: int = 100,
    p_continue: float = 0.6
) -> list[DynamicObstacle]:
    obstacles = []
    for _ in range(max_obstacles):
        continue_old = obstacles and random.random() < p_continue
        if continue_old:
            obs = random.choice(obstacles)
            start = obs.path[-1][1:]
        else:
            obs = None
            start = random_free_cell(task_map)
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
            heuristic_func
        )
        if path is None:
            continue
        if obs is None:
            obstacles.append(DynamicObstacle(path))
        else:
            continue_obstacle(obs, path)
    return obstacles


def save_dynamic_obstacles(
    obstacles: list[DynamicObstacle],
    filename: str
):
    max_time = max(
        obs.path[-1][0] for obs in obstacles if obs.path
    )
    with open(filename, "w", encoding="utf-8") as f:
        for idx, obs in enumerate(obstacles):
            f.write(f"{idx}\n")
            for t in range(max_time + 1):
                i, j = obs.location(t)
                f.write(f"{i} {j}\n")
            f.write("stop\n")


def load_dynamic_obstacles(
    filename: str
) -> list[DynamicObstacle]:
    obstacles: list[DynamicObstacle] = []
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    idx = 0
    n = len(lines)
    while idx < n:
        idx += 1
        path: list[tuple[int, int, int]] = []
        t = 0
        while idx < n and lines[idx] != "stop":
            parts = lines[idx].split()
            i = int(float(parts[-2]))
            j = int(float(parts[-1]))
            path.append((t, i, j))
            t += 1
            idx += 1
        idx += 1  
        obstacles.append(DynamicObstacle(path))
    return obstacles


if __name__ == '__main__':
    task_map = load_map_from_file("data/maps/den600d.map")
    # dynamic_obstacles = generate_dynamic_obstacles(
    #     task_map=task_map,
    #     heuristic_func=manhattan_dist,
    #     max_obstacles=6,
    #     p_continue=0.6
    # )

    # save_dynamic_obstacles(
    #     dynamic_obstacles,
    #     filename="out/dynamic_obstacles.txt"
    # )
    dynamic_obstacles = load_dynamic_obstacles(
        "out/dynamic_obstacles.txt"
    )

    start_i, start_j, goal_i, goal_j = random_start_goal(task_map)

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

    create_animation(
        filename="out/generated_dynamic_obstacles_plus_sipp_agent.gif",
        grid_map=task_map,
        start=agent_path[0],
        goal=agent_path[-1],
        path=agent_path,
        dynamic_obstacles=dynamic_obstacles,
        animation_step=0.4,
        scale=6
    )