from utils.sipp import sipp
from typing import Iterable, Optional, Tuple, Type
from utils.search_tree import SippNode, SearchTreePQD
from utils.map import Map
from utils.sipp import DynamicObstacle

def weighted_function(func, w: float):
    def weighted_func(*args, **kwargs):
        return w * func(*args, **kwargs)
    return weighted_func

def w_sipp(
    task_map: Map, start_i: int, start_j: int, goal_i: int, goal_j: int,
    obstacles: list[DynamicObstacle], search_tree: Type[SearchTreePQD], heuristic_func, w: float
) -> Tuple[bool, Optional[SippNode], int, int, Optional[Iterable[SippNode]], Optional[Iterable[SippNode]]]:
    return sipp(task_map, start_i, start_j, goal_i, goal_j, obstacles, search_tree, weighted_function(heuristic_func, w))