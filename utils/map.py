from typing import List, Tuple
import numpy.typing as npt
import numpy as np 
from collections import deque
import random


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
    

def load_map_from_file(path: str) -> Map:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    height = None
    width = None
    map_start = None
    for idx, line in enumerate(lines):
        if line.startswith("height"):
            height = int(line.split()[1])
        elif line.startswith("width"):
            width = int(line.split()[1])
        elif line == "map":
            map_start = idx + 1
            break
    if height is None or width is None or map_start is None:
        raise ValueError("Некорректный формат карты")

    map_lines = lines[map_start:map_start + height]
    if len(map_lines) != height:
        raise ValueError("Некорректная высота карты")
    
    cells = np.zeros((height, width), dtype=bool)
    OBSTACLES = {'@', 'O', 'T'}
    FREE = {'.', 'G', 'S', 'W'}

    for i, row in enumerate(map_lines):
        if len(row) != width:
            raise ValueError(f"Некорректная ширина карты")
        for j, ch in enumerate(row):
            if ch in OBSTACLES:
                cells[i, j] = True
            elif ch in FREE:
                cells[i, j] = False
            else:
                raise ValueError(f"Некорректный символ в карте: '{ch}' в позиции ({i}, {j})")
    return Map(cells)


def random_free_cell(task_map: Map) -> tuple[int, int]:
    h, w = task_map.get_size()
    while True:
        i = random.randrange(h)
        j = random.randrange(w)
        if task_map.traversable(i, j):
            return i, j
        

def are_connected(task_map: Map, start, goal) -> bool:
    q = deque([start])
    visited = {start}
    while q:
        i, j = q.popleft()
        if (i, j) == goal:
            return True
        for ni, nj in task_map.get_neighbors(i, j):
            if (ni, nj) not in visited:
                visited.add((ni, nj))
                q.append((ni, nj))
    return False


def random_start_goal(
    task_map: Map,
    max_tries: int = 100
) -> tuple[int, int, int, int]:
    start = random_free_cell(task_map)
    for _ in range(max_tries):
        goal = random_free_cell(task_map)
        if goal != start and are_connected(task_map, start, goal):
            return start[0], start[1], goal[0], goal[1]
    raise RuntimeError("Не получилось найти корректные старт и цель")