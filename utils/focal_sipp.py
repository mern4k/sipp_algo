from utils.sipp import DynamicObstacle, Constraints, compute_cost, get_arrival_time
from typing import Iterable, Optional, Tuple
from utils.search_tree import SippNode, SearchTreePQDFocal
from utils.map import Map

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