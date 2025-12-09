from collections import namedtuple
from typing import Iterable, Optional, Tuple, Type, Union
from utils.search_tree import SippNode, SearchTreePQD
from utils.map import Map

SafeInterval = namedtuple('SafeInterval', ['start', 'end'])

class DynamicObstacle:
    def __init__(self, path: list[tuple[int, int, int]]):
        self.path = sorted(path)

    def last_position(self, time: int) -> Optional[tuple[int, int]]:
        if not self.path:
            return None
        last_pos = self.path[0][1:]
        for t, i, j in self.path:
            if time < t:
                return last_pos
            last_pos = (i, j)
        return last_pos
    
    def location(self, time: int) -> tuple[float, float]:
        if time <= self.path[0][0]: 
            return self.path[0][1:]
        if time >= self.path[-1][0]: 
            return self.path[-1][1:]
        cur = None
        next = None
        for cur_, next_ in zip(self.path[:-1], self.path[1:]):
            if cur_[0] <= time < next_[0]:
                cur = cur_
                next = next_
                break
        t1, i1, j1 = cur
        t2, i2, j2 = next
        i = i1 + (i2 - i1) * (time - t1) / (t2 - t1)
        j = j1 + (j2 - j1) * (time - t1) / (t2 - t1)
        return (i, j)

class Constraints:
    def __init__(self, width: int, height: int, obstacles: list[DynamicObstacle]):
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self._cache = {}

    def safe_intervals(self, i: int, j: int) -> list[SafeInterval]:
        if (i, j) in self._cache:
            return self._cache[(i, j)]
        limit_time = float('inf')
        collisions = set()
        for obs in self.obstacles:
            last_t = None
            prev_pos = None
            for t, obs_i, obs_j in obs.path:
                if (obs_i, obs_j) == (i, j):
                    if prev_pos == (i, j) and last_t is not None:
                        collisions.update(range(last_t, t))
                    collisions.add(t)
                last_t = t
                prev_pos = (obs_i, obs_j)
            final_t, final_i, final_j = obs.path[-1]
            if (final_i, final_j) == (i, j):
                if final_t < limit_time:
                    limit_time = final_t

        interval = SafeInterval(0, limit_time)
        if not collisions:
            safe_intervals = [interval]            
            self._cache[(i, j)] = safe_intervals
            return safe_intervals
        final_intervals = []
        current_start = interval.start
        sorted_collisions = sorted(list(t for t in collisions if interval.start <= t < interval.end))
        for t_coll in sorted_collisions:
            if t_coll > current_start:
                final_intervals.append(SafeInterval(current_start, t_coll))
            current_start = t_coll + 1
        if current_start < interval.end:
                final_intervals.append(SafeInterval(current_start, interval.end))
        self._cache[(i, j)] = final_intervals
        return self._cache[(i, j)]
    
    def safe_transition(self, from_i: int, from_j: int, to_i: int, to_j: int, departure_time: int) -> bool:
        arrival_time = departure_time + compute_cost(from_i, from_j, to_i, to_j)
        for obs in self.obstacles:
            obs_start = obs.last_position(departure_time)
            obs_end = obs.last_position(arrival_time)
            if obs_start == (to_i, to_j) and obs_end == (from_i, from_j):
                return False
        return True


def compute_cost(i1: int, j1: int, i2: int, j2: int) -> Union[int, float]:
    if abs(i1 - i2) + abs(j1 - j2) == 1: 
        return 1


def sipp(
    task_map: Map, start_i: int, start_j: int, goal_i: int, goal_j: int,
    obstacles: list[DynamicObstacle], search_tree: Type[SearchTreePQD]
) -> Tuple[bool, Optional[SippNode], int, int, Optional[Iterable[SippNode]], Optional[Iterable[SippNode]]]:
    ast, steps = search_tree(), 0
    width, height = task_map.get_size()
    constraints = Constraints(width, height, obstacles)
    def heuristic_func(i, j): 
        return abs(i - goal_i) + abs(j - goal_j)
    for interval in constraints.safe_intervals(start_i, start_j):
        if interval.start == float('inf'): 
            continue
        start_node = SippNode(start_i, start_j, g=interval.start, h=heuristic_func(start_i, start_j), interval=interval)
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
        for ni, nj in task_map.get_neighbors(cur_node.i, cur_node.j):
            move_duration = compute_cost(cur_node.i, cur_node.j, ni, nj)
            for neighbor_interval in constraints.safe_intervals(ni, nj):
                earliest_departure = cur_node.arrival_time
                earliest_arrival = earliest_departure + move_duration
                actual_arrival = max(earliest_arrival, neighbor_interval.start)
                if actual_arrival - move_duration >= cur_node.interval.end:
                    continue
                if actual_arrival < neighbor_interval.end:
                    departure_time = actual_arrival - move_duration
                    if not constraints.safe_transition(cur_node.i, cur_node.j, ni, nj, int(departure_time)):
                        continue                  
                    new_node = SippNode(ni, nj, g=actual_arrival, h=heuristic_func(ni, nj), parent=cur_node, interval=neighbor_interval)
                    if not ast.was_expanded(new_node):
                        ast.add_to_open(new_node)
    return False, None, steps, len(ast), None, ast.expanded

