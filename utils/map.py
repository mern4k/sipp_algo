from typing import List, Tuple
import numpy.typing as npt

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