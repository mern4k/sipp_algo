from collections import namedtuple
import math
from heapq import heappop, heappush
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

# Простая структура для безопасного интервала
SafeInterval = namedtuple('SafeInterval', ['start', 'end'])

class DynamicObstacle:
    """Представляет одно динамическое препятствие."""
    def __init__(self, path: list[tuple[int, int, int]]):
        self.path = sorted(path)

    def get_location_at(self, time: int) -> tuple[int, int]:
        """Возвращает позицию (i, j) препятствия в заданное время."""
        if not self.path:
            return (-1, -1)
        last_pos = self.path[0][1:]
        for t, i, j in self.path:
            if time < t:
                return last_pos
            last_pos = (i, j)
        return last_pos

class Constraints:
    def __init__(self, width: int, height: int, obstacles: list[DynamicObstacle]):
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self._cache = {}

    def get_safe_intervals(self, i: int, j: int) -> list[SafeInterval]:
        if (i, j) in self._cache:
            return self._cache[(i, j)]

        safe_intervals = [SafeInterval(0, float('inf'))]
        
        collision_times = set()
        for obs in self.obstacles:
            last_t = -1
            prev_pos = (-1, -1)
            for t, obs_i, obs_j in obs.path:
                if (obs_i, obs_j) == (i, j):
                    # Если препятствие было в этой клетке в предыдущий момент времени,
                    # заполняем все промежуточные временные шаги
                    if prev_pos == (i, j) and last_t != -1:
                        for time_step in range(last_t, t):
                           collision_times.add(time_step)
                    collision_times.add(t)
                last_t = t
                prev_pos = (obs_i, obs_j)

        if not collision_times:
             self._cache[(i,j)] = safe_intervals
             return safe_intervals
             
        final_intervals = []
        for interval in safe_intervals:
            current_start = interval.start
            sorted_collisions = sorted(list(t for t in collision_times if interval.start <= t < interval.end))
            
            for t_coll in sorted_collisions:
                if t_coll > current_start:
                    final_intervals.append(SafeInterval(current_start, t_coll))
                current_start = t_coll + 1
            
            if current_start < interval.end:
                 final_intervals.append(SafeInterval(current_start, interval.end))

        # Если после всех коллизий не осталось интервалов, это может быть ошибкой,
        # но для простоты добавим интервал после последней коллизии.
        if not final_intervals and collision_times:
             final_intervals.append(SafeInterval(max(collision_times) + 1, float('inf')))

        self._cache[(i, j)] = final_intervals
        return self._cache[(i, j)]
    
    def is_transition_safe(self, from_i: int, from_j: int, to_i: int, to_j: int, departure_time: int) -> bool:
        """
        Проверяет, является ли переход из (from_i, from_j) в (to_i, to_j) безопасным,
        учитывая обмен местами с динамическими препятствиями.
        """
        arrival_time = departure_time + 1

        for obs in self.obstacles:
            obs_pos_start = obs.get_location_at(departure_time)
            obs_pos_end = obs.get_location_at(arrival_time)
            if obs_pos_start == (to_i, to_j) and obs_pos_end == (from_i, from_j):
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
        if not isinstance(other, BaseNode): 
            return NotImplemented
        return (self.f < other.f) or (self.f == other.f and self.h < other.h)
    def __hash__(self): 
        return hash((self.i, self.j, self.interval))
    def __eq__(self, other): 
        return isinstance(other, SippNode) and self.i == other.i and self.j == other.j and self.interval == other.interval

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
    def get_size(self) -> Tuple[int, int]: return self._height, self._width

def convert_string_to_cells(cell_str: str) -> npt.NDArray:
    lines = cell_str.replace(" ", "").split("\n")
    return np.array([[1 if char == "#" else 0 for char in line] for line in lines if line], dtype=np.int8)

def compute_cost(i1: int, j1: int, i2: int, j2: int) -> Union[int, float]:
    if abs(i1 - i2) + abs(j1 - j2) == 1: 
        return 1
    raise ValueError("Only cardinal moves are supported.")

def draw_rectangle(draw, i, j, scale, color):
    draw.rectangle(((j * scale, i * scale), ((j + 1) * scale - 1, (i + 1) * scale - 1)), fill=color, width=0)


def get_agent_position_at_time(path: List[SippNode], time: int) -> Tuple[int, int]:
    """Находит позицию агента в момент времени t на основе пути."""
    if not path:
        return (-1, -1)
    
    # Находим последний узел, в который агент уже прибыл
    last_arrived_node = path[0]
    for node in path:
        if node.arrival_time <= time:
            last_arrived_node = node
        else:
            break
    return last_arrived_node.i, last_arrived_node.j


def create_animation(
    filename: str,
    grid_map: Map,
    start: SippNode,
    goal: SippNode,
    path: List[SippNode],
    dynamic_obstacles: List[DynamicObstacle],
):
    """Создает и сохраняет анимацию движения в GIF файл."""
    scale = 20
    height, width = grid_map.get_size()
    frames = []
    
    # Загружаем шрифт для отображения времени
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    final_time = path[-1].arrival_time
    print(f"Generating animation for {final_time+1} frames...")

    for t in range(final_time + 2): # +2 чтобы показать финальное состояние
        time = min(t, final_time) # Остаемся в финальном состоянии после прибытия
        
        im = Image.new("RGB", (width * scale, height * scale), color="white")
        draw = ImageDraw.Draw(im)

        # Рисуем статические препятствия
        for i in range(height):
            for j in range(width):
                if not grid_map.traversable(i, j):
                    draw_rectangle(draw, i, j, scale, (70, 80, 80))

        # Рисуем старт и цель
        draw_rectangle(draw, start.i, start.j, scale, (40, 180, 99))
        draw_rectangle(draw, goal.i, goal.j, scale, (231, 76, 60))

        # Рисуем положение агента в момент времени t
        agent_i, agent_j = get_agent_position_at_time(path, time)
        draw_rectangle(draw, agent_i, agent_j, scale, (52, 152, 219))

        # Рисуем положение динамических препятствий в момент времени t
        obstacle_colors = [(199, 12, 42), (80, 20, 180), (255, 140, 0)]
        for i, obstacle in enumerate(dynamic_obstacles):
            obs_i, obs_j = obstacle.get_location_at(time)
            if grid_map.in_bounds(obs_i, obs_j):
                color = obstacle_colors[i % len(obstacle_colors)]
                draw_rectangle(draw, obs_i, obs_j, scale, color)
        
        # Добавляем текст с текущим временем
        draw.text((5, 5), f"Time: {time}", fill="black", font=font)
        
        frames.append(im)

    print("Saving animation to", filename)
    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=200,  # мс на кадр (200 = 5 FPS)
        loop=0
    )
    print("Done!")

class SearchTreePQD:
    def __init__(self):
        self._open, self._closed, self._enc_open_duplicates = [], {}, 0
    def __len__(self) -> int: 
        return len(self._open) + len(self._closed)
    def open_is_empty(self) -> bool: 
        return not self._open
    def add_to_open(self, item: SippNode): heappush(self._open, item)
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
    for interval in constraints.get_safe_intervals(start_i, start_j):
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
        if cur_node.i == goal_i and cur_node.j == goal_j:
            return True, cur_node, steps, len(ast), ast.opened, ast.expanded
        steps += 1
        for ni, nj in task_map.get_neighbors(cur_node.i, cur_node.j):
            move_duration = compute_cost(cur_node.i, cur_node.j, ni, nj)
            for neighbor_interval in constraints.get_safe_intervals(ni, nj):
                # SIPP позволяет агенту ждать в текущем узле
                # Мы можем отправиться в любой момент времени t >= cur_node.arrival_time,
                # пока мы остаемся в безопасном интервале cur_node.interval
                
                wait_end_time = cur_node.interval.end
                earliest_departure = cur_node.arrival_time
                
                earliest_arrival_at_neighbor = earliest_departure + move_duration
                actual_arrival = max(earliest_arrival_at_neighbor, neighbor_interval.start)
                
                # Проверяем, что можно дождаться начала интервала соседа, не покинув свой интервал
                if actual_arrival - move_duration >= wait_end_time:
                    continue

                if actual_arrival < neighbor_interval.end:
                    departure_time = actual_arrival - move_duration
                    if not constraints.is_transition_safe(cur_node.i, cur_node.j, ni, nj, int(departure_time)):
                        continue # Если переход небезопасен, пропускаем этот вариант
                    
                    new_node = SippNode(ni, nj, g=actual_arrival, h=heuristic_func(ni, nj), parent=cur_node, interval=neighbor_interval)
                    if not ast.was_expanded(new_node):
                        ast.add_to_open(new_node)
    return False, None, steps, len(ast), None, ast.expanded

if __name__ == '__main__':
    map_str = """
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
    cells = convert_string_to_cells(map_str)
    test_map = Map(cells)
    start_node = SippNode(1, 1)
    goal_node = SippNode(13, 28)
    
    obs1_path = [(t, 1, 5) for t in range(10)]
    obstacle1 = DynamicObstacle(obs1_path)
    obs2_path = [(t, 2, 8-t) for t in range(7)]
    obstacle2 = DynamicObstacle(obs2_path)
    obs3_path = [(t, 2, 17-t) for t in range(17)]
    obstacle3 = DynamicObstacle(obs3_path)
    obs4_path = [(t, 3, 23-t) for t in range(23)]
    obstacle4 = DynamicObstacle(obs4_path)
    obs5_path = [(t, 0, 12-t) for t in range(13)]
    obstacle5 = DynamicObstacle(obs5_path)

    dynamic_obstacles = [obstacle1, obstacle2, obstacle3, obstacle4, obstacle5]

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
        
        print(f"Path found! Arrival time: {last_node.arrival_time}. Steps: {steps}. Tree size: {tree_size}")
        create_animation("sipp_animation.gif", test_map, start_node, goal_node, path, dynamic_obstacles)
    else:
        print("Path not found!")