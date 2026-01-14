from utils.sipp import DynamicObstacle, Constraints, compute_cost, get_arrival_time
from typing import Iterable, Optional, Tuple
from utils.search_tree import SippNode, SearchTreePQDFocal
from utils.map import Map
import numpy as np
from collections import deque
from typing import Callable

def focal_sipp(
    task_map: Map, start_i: int, start_j: int, goal_i: int, goal_j: int, obstacles: list[DynamicObstacle], 
    heuristic_func: callable, w: float, focal_heuristic: callable
) -> Tuple[bool, Optional[SippNode], int, int, Optional[Iterable[SippNode]], Optional[Iterable[SippNode]]]:
    ast = SearchTreePQDFocal(focal_heuristic, w, goal_i, goal_j)
    steps = 0
    width, height = task_map.get_size()
    constraints = Constraints(width, height, obstacles)
    
    for interval in constraints.safe_intervals(start_i, start_j):
        if interval.start == float('inf'):
            continue
        start_node = SippNode(start_i, start_j, g=interval.start, h=heuristic_func(start_i, start_j, goal_i, goal_j), interval=interval)
        ast.add_to_open(start_node)

    while not ast.open_is_empty():
        cur_node = ast.get_best_node_from_open()
        if cur_node is None: 
            break
        if ast.was_expanded(cur_node): 
            continue
        ast.add_to_closed(cur_node)
        if cur_node.i == goal_i and cur_node.j == goal_j and cur_node.interval.end == float('inf'):
            return True, cur_node, steps, len(ast), ast.opened, ast.expanded
        steps += 1
        for row, col in task_map.get_neighbors(cur_node.i, cur_node.j):
            move_duration = compute_cost(cur_node.i, cur_node.j, row, col)
            for neighbor_interval in constraints.safe_intervals(row, col):
                actual_arrival = get_arrival_time(cur_node, row, col, move_duration, neighbor_interval, constraints)
                if actual_arrival is None:
                    continue
                new_node = SippNode(row, col, g=actual_arrival, h=heuristic_func(row, col, goal_i, goal_j), parent=cur_node, interval=neighbor_interval)
                ast.add_to_open(new_node)
    return False, None, steps, len(ast), None, ast.expanded

def get_heuristic(task_map: Map, goal_i: int, goal_j: int) -> Callable[[int, int, int, int], float]:    
    height, width = task_map.get_size()
    dist = np.full((height, width), np.inf)
    queue = deque([(goal_i, goal_j)])
    dist[goal_i, goal_j] = 0
    while queue:
        cur_i, cur_j = queue.popleft()
        for next_i, next_j in task_map.get_neighbors(cur_i, cur_j):
            if dist[next_i, next_j] == np.inf:
                dist[next_i, next_j] = dist[cur_i, cur_j] + compute_cost(cur_i, cur_j, next_i, next_j)
                queue.append((next_i, next_j))
    
    def heuristic(i: int, j: int, gi: int, gj: int) -> float:           
        return float(dist[i][j])
    return heuristic