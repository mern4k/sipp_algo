from utils.sipp import sipp, DynamicObstacle, Constraints, compute_cost, get_arrival_time
from typing import Iterable, Optional, Tuple
from utils.search_tree import SippNode, SippNodeDublicate, SearchTreePQD
from utils.map import Map


def weighted_function(func, w: float):
    def weighted_func(*args, **kwargs):
        return w * func(*args, **kwargs)
    return weighted_func


def w_sipp(
    task_map: Map, start_i: int, start_j: int, goal_i: int, goal_j: int, 
    obstacles: list[DynamicObstacle], heuristic_func: callable, w: float,
) -> Tuple[bool, Optional[SippNode], int, int, Optional[Iterable[SippNode]], Optional[Iterable[SippNode]]]:
    return sipp(task_map, start_i, start_j, goal_i, goal_j, obstacles, weighted_function(heuristic_func, w), False)


def w_sipp_with_reexpansions(
    task_map: Map, start_i: int, start_j: int, goal_i: int, goal_j: int, 
    obstacles: list[DynamicObstacle], heuristic_func: callable, w: float,
) -> Tuple[bool, Optional[SippNode], int, int, Optional[Iterable[SippNode]], Optional[Iterable[SippNode]]]:
    return sipp(task_map, start_i, start_j, goal_i, goal_j, obstacles, weighted_function(heuristic_func, w), True)


def w_sipp_dublicate_states(
    task_map: Map, start_i: int, start_j: int, goal_i: int, goal_j: int, 
    obstacles: list[DynamicObstacle], heuristic_func: callable, w: float,
) -> Tuple[bool, Optional[SippNode], int, int, Optional[Iterable[SippNode]], Optional[Iterable[SippNode]]]:
    ast = SearchTreePQD()
    steps = 0
    width, height = task_map.get_size()
    constraints = Constraints(width, height, obstacles)
    
    for interval in constraints.safe_intervals(start_i, start_j):
        if interval.start == float('inf'):
            continue
        start_node = SippNodeDublicate(start_i, start_j, g=interval.start, h=w*heuristic_func(start_i, start_j, goal_i, goal_j), interval=interval, isOptimal=True)
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
                g_ = actual_arrival
                h_ = heuristic_func(row, col, goal_i, goal_j)
                suboptimal_node = SippNodeDublicate(row, col, g=g_, h=h_, f=g_+w*h_, parent=cur_node, interval=neighbor_interval, isOptimal=False)
                ast.add_to_open(suboptimal_node)
                if cur_node.isOptimal:
                    suboptimal_node = SippNodeDublicate(row, col, g=g_, h=h_, f=w*(g_+h_), parent=cur_node, interval=neighbor_interval, isOptimal=True)
                    ast.add_to_open(suboptimal_node)
    return False, None, steps, len(ast), None, ast.expanded