from typing import Iterable, Optional, Tuple
from utils.search_tree import TimeAStarNode, SearchTreePQD
from utils.map import Map
from utils.sipp import DynamicObstacle


class TimeConstraints:
    def __init__(self, obstacles: list[DynamicObstacle]):
        self.obstacles = obstacles
        self._max_obs_time = 0
        for obs in obstacles:
            if obs.path:
                self._max_obs_time = max(self._max_obs_time, obs.path[-1][0])
    
    def is_safe_vertex(self, i: int, j: int, t: int) -> bool:
        for obs in self.obstacles:
            pos = obs.last_position(t)
            if pos == (i, j):
                return False
        return True
    
    def is_safe_edge(self, i1: int, j1: int, i2: int, j2: int, t: int) -> bool:
        if not self.is_safe_vertex(i2, j2, t + 1):
            return False
        for obs in self.obstacles:
            obs_start = obs.last_position(t)
            obs_end = obs.last_position(t + 1)
            if obs_start == (i2, j2) and obs_end == (i1, j1):
                return False
        return True
    
    def can_stay_forever(self, i: int, j: int, t: int) -> bool:
        for obs in self.obstacles:
            if not obs.path:
                continue
            
            final_t, final_i, final_j = obs.path[-1]
            if (final_i, final_j) == (i, j):
                return False
            
            for obs_t, obs_i, obs_j in obs.path:
                if (obs_i, obs_j) == (i, j) and obs_t >= t:
                    return False
        
        return True
    
    def get_first_safe_time(self, i: int, j: int, max_time: int) -> Optional[int]:
        for t in range(max_time + 1):
            if self.is_safe_vertex(i, j, t):
                return t
        return None


def time_astar(
    task_map: Map, start_i: int, start_j: int, goal_i: int, goal_j: int,
    obstacles: list[DynamicObstacle], heuristic_func: callable
) -> Tuple[bool, Optional[TimeAStarNode], int, int, Optional[Iterable], Optional[Iterable]]:
    ast = SearchTreePQD()
    steps = 0
    constraints = TimeConstraints(obstacles)
    
    height, width = task_map.get_size()
    max_time_factor = 4
    max_time = max(constraints._max_obs_time * max_time_factor, height * width * 2)
    
    start_time = constraints.get_first_safe_time(start_i, start_j, max_time)
    if start_time is None:
        return False, None, 0, 0, None, None
    
    start_node = TimeAStarNode(
        start_i, start_j, t=start_time, g=start_time, 
        h=heuristic_func(start_i, start_j, goal_i, goal_j)
    )
    ast.add_to_open(start_node)
    
    while not ast.open_is_empty():
        cur_node = ast.get_best_node_from_open()
        if cur_node is None:
            break
        if ast.was_expanded(cur_node):
            continue
        ast.add_to_closed(cur_node)
        if cur_node.i == goal_i and cur_node.j == goal_j:
            if constraints.can_stay_forever(goal_i, goal_j, cur_node.t):
                return True, cur_node, steps, len(ast), ast.opened, ast.expanded
        steps += 1
        if cur_node.t >= max_time:
            continue
        if constraints.is_safe_vertex(cur_node.i, cur_node.j, cur_node.t + 1):
            wait_node = TimeAStarNode(
                cur_node.i, cur_node.j, 
                t=cur_node.t + 1,
                g=cur_node.g + 1,
                h=heuristic_func(cur_node.i, cur_node.j, goal_i, goal_j),
                parent=cur_node
            )
            if not ast.was_expanded(wait_node):
                ast.add_to_open(wait_node)
        for row, col in task_map.get_neighbors(cur_node.i, cur_node.j):
            if constraints.is_safe_edge(cur_node.i, cur_node.j, row, col, cur_node.t):
                new_node = TimeAStarNode(
                    row, col,
                    t=cur_node.t + 1,
                    g=cur_node.g + 1,
                    h=heuristic_func(row, col, goal_i, goal_j),
                    parent=cur_node
                )
                if not ast.was_expanded(new_node):
                    ast.add_to_open(new_node)
    
    return False, None, steps, len(ast), None, ast.expanded