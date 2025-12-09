from collections import namedtuple
import math
from heapq import heappop, heappush
import os
from random import randint
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

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


class BaseNode:
    def __init__(self, i: int, j: int, g: Union[float, int] = 0, h: Union[float, int] = 0, f: Optional[Union[float, int]] = None, parent: Optional["BaseNode"] = None):
        self.i, self.j, self.g, self.h, self.parent = i, j, g, h, parent
        self.f = g + h if f is None else f
    def __eq__(self, other): 
        return self.i == other.i and self.j == other.j
    def __hash__(self): 
        return hash((self.i, self.j))
    def __lt__(self, other): 
        return self.g < other.g

class SippNode(BaseNode):
    def __init__(self, i, j, g=0, h=0, parent=None, interval=None):
        super().__init__(i=i, j=j, g=g, h=h, parent=parent)
        self.interval = interval
    @property
    def arrival_time(self): 
        return self.g
    def __lt__(self, other):
        return (self.f < other.f) or (self.f == other.f and self.h < other.h)
    def __hash__(self): 
        return hash((self.i, self.j, self.interval))
    def __eq__(self, other): 
        return self.i == other.i and self.j == other.j and self.interval == other.interval

class Map:
    def __init__(self, cells: npt.NDArray):
        self._width, self._height = cells.shape[1], cells.shape[0]
        self._cells = cells
    def in_bounds(self, i: int, j: int) -> bool: 
        return 0 <= j < self._width and 0 <= i < self._height
    def traversable(self, i: int, j: int) -> bool: 
        return not self._cells[i, j]
    def get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        neighbors = []
        for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            ni, nj = i + dx, j + dy
            if self.in_bounds(ni, nj) and self.traversable(ni, nj):
                neighbors.append((ni, nj))
        return neighbors
    def get_size(self) -> Tuple[int, int]: 
        return self._height, self._width

def convert_string_to_cells(cell_str: str) -> npt.NDArray:
    lines = cell_str.replace(" ", "").split("\n")
    return np.array([[1 if char == "#" else 0 for char in line] for line in lines if line], dtype=np.int8)

def compute_cost(i1: int, j1: int, i2: int, j2: int) -> Union[int, float]:
    if abs(i1 - i2) + abs(j1 - j2) == 1: 
        return 1

def draw_rectangle(draw, i, j, scale, color):
    draw.rectangle(((j * scale, i * scale), ((j + 1) * scale - 1, (i + 1) * scale - 1)), fill=color, width=0)


def agent_position(path: List[SippNode], time: float) -> tuple[float, float]:
    if time <= path[0].arrival_time: 
        return (float(path[0].i), float(path[0].j))
    if time >= path[-1].arrival_time: 
        return (float(path[-1].i), float(path[-1].j))
    cur = None
    next = None
    for i in range(len(path) - 1):
        if path[i].arrival_time <= time < path[i+1].arrival_time:
            cur = path[i]
            next = path[i+1]
            break
    t1, i1, j1 = cur.arrival_time, cur.i, cur.j
    t2, i2, j2 = next.arrival_time, next.i, next.j
    start_time = t2 - 1
    if time < start_time:
        return (float(i1), float(j1))
    if t1 == t2: return (float(i1), float(j1))
    i = i1 + (i2 - i1) * (time - start_time)
    j = j1 + (j2 - j1) * (time - start_time)
    return (i, j)

def create_animation(
    filename: str,
    grid_map: Map,
    start: SippNode,
    goal: SippNode,
    path: List[SippNode],
    dynamic_obstacles: List[DynamicObstacle],
):
    scale = 20
    height, width = grid_map.get_size()
    frames = []
    final_time = path[-1].arrival_time
    animation_step = 0.2 
    obstacle_colors = [(randint(30, 230), randint(30, 230), randint(30, 230)) for _ in dynamic_obstacles]
        
    for t in np.arange(0, final_time + 2, animation_step):
        time = min(t, final_time)
        im = Image.new("RGB", (width * scale, height * scale), color="white")
        draw = ImageDraw.Draw(im)
        for i in range(height):
            for j in range(width):
                if not grid_map.traversable(i, j):
                    draw_rectangle(draw, i, j, scale, (70, 80, 80))
        draw_rectangle(draw, start.i, start.j, scale, "green")
        draw_rectangle(draw, goal.i, goal.j, scale, "red")

        radius = scale * 0.28
        agent_i, agent_j = agent_position(path, time)
        cx, cy = (agent_j + 0.5) * scale, (agent_i + 0.5) * scale
        draw.ellipse([(cx - radius, cy - radius), (cx + radius, cy + radius)], fill="blue")
        for i, obstacle in enumerate(dynamic_obstacles):
            obs_i, obs_j = obstacle.location(time)
            if grid_map.in_bounds(obs_i, obs_j):
                color = obstacle_colors[i]
                cx, cy = (obs_j + 0.5) * scale, (obs_i + 0.5) * scale
                draw.rectangle([(cx - radius, cy - radius), (cx + radius, cy + radius)], fill=color)
        draw.text((5, 5), f"Time: {time:.1f}", fill="black")
        frames.append(im)

    print("Saving to", filename)
    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=50, 
        loop=0
    )
    print("Done")

class SearchTreePQD:
    def __init__(self):
        self._open, self._closed, self._enc_open_duplicates = [], {}, 0
    def __len__(self) -> int: 
        return len(self._open) + len(self._closed)
    def open_is_empty(self) -> bool: 
        return not self._open
    def add_to_open(self, item: SippNode): 
        heappush(self._open, item)
    def get_best_node_from_open(self) -> Optional[SippNode]:
        while self._open:
            best_node = heappop(self._open)
            if best_node not in self._closed: 
                return best_node
            self._enc_open_duplicates += 1
        return None
    def add_to_closed(self, item: SippNode): self._closed[item] = item
    def was_expanded(self, item: SippNode) -> bool: 
        return item in self._closed
    @property
    def opened(self): 
        return self._open
    @property
    def expanded(self): 
        return self._closed.values()
    @property
    def number_of_open_dublicates(self): 
        return self._enc_open_duplicates

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

def read_lists_from_file(path: str) -> List[List[Tuple[int, int, int]]]:
    with open(path) as file:
        main_list = []
        curr_list = []
        t = 0
        for line in file:
            if not line.strip():
                continue
            nums = tuple(map(int, line.split()))

            if len(nums) == 1:
                if curr_list:
                    main_list.append(curr_list)
                    curr_list = []
                    t = 0
            else:
                curr_list.append((t, nums[0], nums[1]))
                t += 1

        if curr_list:
            main_list.append(curr_list)

        return main_list
    
if __name__ == '__main__':
    map_0 = """
    . . . . . . . . . . . . . . . . . . . . . # # . . . . . . .  
    . . . . . . . . . . . . . . . . . . . . . # # . . . . . . . 
    . . . . . . . . . . . . . . . . . . . . . # # . . . . . . . 
    . . . # # . . . . . . . . . . . . . . . . # # . . . . . . . 
    . . . # # . . . . . . . . # # . . . . . . # # . . . . . . . 
    . . . # # . . . . . . . . # # . . . . . . # # # # # . . . . 
    . . . # # . . . . . . . . # # . . . . . . # # # # # . . . . 
    . . . # # . . . . . . . . # # . . . . . . . . . . . . . . . 
    . . . # # . . . . . . . . # # . . . . . . . . . . . . . . . 
    . . . # # . . . . . . . . # # . . . . . . . . . . . . . . . 
    . . . # # . . . . . . . . # # . . . . . . . . . . . . . . . 
    . . . # # . . . . . . . . # # . . . . . . . . . . . . . . . 
    . . . . . . . . . . . . . # # . . . . . . . . . . . . . . . 
    . . . . . . . . . . . . . # # . . . . . . . . . . . . . . . 
    . . . . . . . . . . . . . # # . . . . . . . . . . . . . . . 
    """
    map_str = """
        . . . # # . . . . . . . . # # . . . # . . # # . . . . . . .  
        . . . # # # # # . . # . . # # . . . . . . # # . . . . . . . 
        . . . . . . . # . . # . . # # . . . # . . # # . . . . . . . 
        . . . # # . . # . . # . . # # . . . # . . # # . . . . . . . 
        . . . # # . . # . . # . . # # . . . # . . # # . . . . . . . 
        . . . # # . . # . . # . . # # . . . # . . # # # # # . . . . 
        . . . # # . . # . . # . . # # . . . # . . # # # # # . . . . 
        . . . . . . . # . . # . . # # . . . # . . # . . . . . . . . 
        . . . # # . . # . . # . . # # . . . # . . # . . . . . . . . 
        . . . # # . . # . . # . . # # . . . # . . # . . . . . . . . 
        . . . # # . . . . . # . . . . . . . # . . . . . . . . . . . 
        . . . # # # # # # # # # # # # # . # # . # # # # # # # . # # 
        . . . # # . . . . . . . . # # . . . . . . . . . . . . . . . 
        . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        . . . # # . . . . . . . . # # . . . . . . . . . . . . . . .
    """
    cells = convert_string_to_cells(map_str)
    test_map = Map(cells)

    starts = [(1, 28), (2, 29), (3, 20), (3, 20), (0, 0)]
    goals = [(0, 1), (6, 2), (5, 6), (13, 0), (4, 23)]
    durations = [54, 47, 48, 71, 56]

    test_index = 4

    start_node = SippNode(starts[test_index][0], starts[test_index][1])
    goal_node = SippNode(goals[test_index][0], goals[test_index][1])
    #start_node = SippNode(1, 1)
    #goal_node = SippNode(6, 27)

    
    obs1_path = [(t, 1, 5-t) for t in range(3)]
    obstacle1 = DynamicObstacle(obs1_path)
    obs2_path = [(t, 2, 8-t) for t in range(7)]
    obstacle2 = DynamicObstacle(obs2_path)
    obs3_path = [(t, 2, 17-t) for t in range(17)]
    obstacle3 = DynamicObstacle(obs3_path)
    obs4_path = [(t, 3, 23-t) for t in range(23)]
    obstacle4 = DynamicObstacle(obs4_path)
    obs5_path = [(t, 0, 12-t) for t in range(13)]
    obstacle5 = DynamicObstacle(obs5_path)

    #dynamic_obstacles = [obstacle1, obstacle2, obstacle3, obstacle4, obstacle5]
    dynamic_obstacles = [DynamicObstacle(obs) for obs in read_lists_from_file(os.path.join("obstacles.txt"))]

    path_found, last_node, steps, tree_size, open_nodes, closed_nodes = sipp(
        test_map, start_node.i, start_node.j, goal_node.i, goal_node.j, 
        dynamic_obstacles, SearchTreePQD
    )
    
    if path_found:
        path = []
        curr = last_node
        while curr is not None:
            path.append(curr)
            curr = curr.parent
        path.reverse()

        correct = last_node.arrival_time == durations[test_index]
        
        print(f"Path found. Arrival time: {last_node.arrival_time}. Steps: {steps}. Tree size: {tree_size}. Correct: {correct}")
        create_animation("sipp_animation.gif", test_map, start_node, goal_node, path, dynamic_obstacles)
    else:
        print("Path not found.")