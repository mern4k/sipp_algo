from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict
import math
import numpy as np

INF = float('inf')
EPS = 1e-9


@dataclass
class SafeInterval:
    begin: float = 0.0
    end: float = INF
    id: int = 0


@dataclass
class Agent:
    id: str = ""
    id_num: int = -1
    start_i: int = -1
    start_j: int = -1
    goal_i: int = -1
    goal_j: int = -1
    size: float = 0.5
    mspeed: float = 1.0


class Vector2D:
    def __init__(self, i: float = 0.0, j: float = 0.0):
        self.i = i
        self.j = j

    def __add__(self, other):
        return Vector2D(self.i + other.i, self.j + other.j)

    def __sub__(self, other):
        return Vector2D(self.i - other.i, self.j - other.j)

    def __neg__(self):
        return Vector2D(-self.i, -self.j)

    def __truediv__(self, num: float):
        return Vector2D(self.i / num, self.j / num)

    def __mul__(self, num: float):
        return Vector2D(self.i * num, self.j * num)

    def dot(self, other) -> float:
        return self.i * other.i + self.j * other.j

    def norm2(self) -> float:
        return self.dot(self)


class Point:
    def __init__(self, i: float = 0.0, j: float = 0.0):
        self.i = i
        self.j = j

    def __sub__(self, p: 'Point'):
      return Point(self.i - p.i, self.j - p.j)

    def __eq__(self, p: 'Point'):
        return abs(self.i - p.i) < EPS and abs(self.j - p.j) < EPS

    def classify(self, p0: 'Point', p1: 'Point') -> int:
        p2 = Point(self.i, self.j)
        a = Point(p1.i - p0.i, p1.j - p0.j)
        b = Point(p2.i - p0.i, p2.j - p0.j)
        sa = a.i * b.j - b.i * a.j
        if sa > 0.0:
            return 1
        if sa < 0.0:
            return 2
        if (a.i * b.i < 0.0) or (a.j * b.j < 0.0):
            return 3
        if (a.i * a.i + a.j * a.j) < (b.i * b.i + b.j * b.j):
            return 4
        if p0 == p2:
            return 5
        if p1 == p2:
            return 6
        return 7


class Section:
    def __init__(self, i1: int = -1, j1: int = -1, i2: int = -1, j2: int = -1, g1: float = -1.0, g2: float = -1.0):
        self.i1 = i1
        self.j1 = j1
        self.i2 = i2
        self.j2 = j2
        self.size = 0.0
        self.g1 = g1
        self.g2 = g2
        self.mspeed = 1.0

    @classmethod
    def from_nodes(cls, a: 'Node', b: 'Node'):
        s = cls(a.i, a.j, b.i, b.j, a.g, b.g)
        return s

    def __eq__(self, other: 'Section'):
        return (self.i1 == other.i1 and self.j1 == other.j1 and abs(self.g1 - other.g1) < EPS)


class Node:
    def __init__(
        self,
        i: int,
        j: int,
        interval: SafeInterval = SafeInterval(),
        g: Union[float, int] = 0,
        h: Union[float, int] = 0,
        f: Optional[Union[float, int]] = None,
        parent: Optional["Node"] = None,
    ):
        self.i = i
        self.j = j
        self.interval = interval
        self.g = g
        self.h = h
        if f is None:
            self.f = self.g + h
        else:
            self.f = f
        self.parent = parent

    def __eq__(self, other):
        """
        Checks if two search nodes are the same, which is needed 
        to detect duplicates in the search tree.
        """
        return self.i == other.i and self.j == other.j and self.interval == other.interval

    def __hash__(self):
        """
        Makes the Node object hashable, allowing it to be used in sets/dictionaries.
        """
        return hash((self.i, self.j, self.interval.begin, self.interval.end, self.interval.id))

    def __lt__(self, other):
        """
        Compares the keys (i.e., the f-values) of two nodes, 
        needed for sorting/extracting the best element from OPEN.
        """
        if self.f == other.f:
            return self.g > other.g
        return self.f < other.f
    

class Obstacle:
    def __init__(self, id_: str = "", size: float = 0.5, mspeed: float = 1.0):
        self.id = id_
        self.size = size
        self.mspeed = mspeed
        self.sections: List[Node] = []


class DynamicObstacles:
    def __init__(self):
        self.obstacles: List[Obstacle] = []

    def get_obstacles(self) -> List[Obstacle]:
        return self.obstacles

    def add_obstacle(self, obs: Obstacle):
        self.obstacles.append(obs)

    def get_sections(self, num: int) -> List[Node]:
        return self.obstacles[num].sections

    def get_size(self, num: int) -> float:
        return self.obstacles[num].size

    def get_mspeed(self, num: int) -> float:
        return self.obstacles[num].mspeed

    def get_id(self, num: int) -> str:
        return self.obstacles[num].id

    def get_number_of_obstacles(self) -> int:
        return len(self.obstacles)
    

class LineOfSight:
    def __init__(self, size: float = 0.5):
        self.size = size

    def set_size(self, s: float):
        self.size = s

    def bresenham(self, i0: int, j0: int, i1: int, j1: int) -> List[Tuple[int, int]]:
        cells = []
        di = abs(i1 - i0)
        dj = abs(j1 - j0)
        si = 1 if i0 < i1 else -1
        sj = 1 if j0 < j1 else -1
        err = di - dj
        i, j = i0, j0
        while True:
            cells.append((i, j))
            if i == i1 and j == j1:
                break
            e2 = 2 * err
            if e2 > -dj:
                err -= dj
                i += si
            if e2 < di:
                err += di
                j += sj
        return cells

    def getCells(self, i: int, j: int) -> List[Tuple[int, int]]:
        radius = max(1, int(math.ceil(self.size)))
        cells = []
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                cells.append((i + di, j + dj))
        return cells

    def checkLine(self, i0: int, j0: int, i1: int, j1: int, map_obj: 'Map') -> bool:
        for (ci, cj) in self.bresenham(i0, j0, i1, j1):
            if not map_obj.in_bounds(ci, cj) or not map_obj.traversable(ci, cj):
                    return False
        return True

    def getCellsCrossedByLine(self, i0: int, j0: int, i1: int, j1: int, map_obj: 'Map') -> List[Tuple[int, int]]:
        return [(ci, cj) for (ci, cj) in self.bresenham(i0, j0, i1, j1) if map_obj.in_bounds(ci, cj)]
    

class Map:
    def __init__(self, cells: np.ndarray):
        if cells is None:
            cells = np.zeros((0, 0), dtype=np.int8)
        self._cells = np.array(cells)
        self._height = self._cells.shape[0]
        self._width = self._cells.shape[1]

    def in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self._height and 0 <= j < self._width

    def traversable(self, i: int, j: int) -> bool:
        return self._cells[i, j] == 0

    def get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        neighbors = []
        delta = ((0, 1), (1, 0), (0, -1), (-1, 0))
        for di, dj in delta:
            ni, nj = i + di, j + dj
            if self.in_bounds(ni, nj) and self.traversable(ni, nj):
                neighbors.append((ni, nj))
        return neighbors

    def get_size(self) -> Tuple[int, int]:
        return self._height, self._width

    def get_valid_moves(self, i: int, j: int, k: int = 2, size: float = 0.5) -> List[Node]:
        los = LineOfSight(size)
        moves = []
        if k == 2:
            moves = [Node(0, 1, SafeInterval(), 1.0), Node(1, 0, SafeInterval(), 1.0), Node(-1, 0, SafeInterval(), 1.0), Node(0, -1, SafeInterval(), 1.0)]
        else: # add 2^k-connections, skipped rn
            moves = [Node(0, 1, SafeInterval(), 1.0), Node(1, 0, SafeInterval(), 1.0), Node(0, -1, SafeInterval(), 1.0), Node(-1, 0, SafeInterval(), 1.0)]
        valid = []
        for m in moves:
            ni, nj = i + m.i, j + m.j
            if self.in_bounds(ni, nj) and self.traversable(ni, nj) and los.checkLine(i, j, ni, nj, self):
                move_copy = Node(ni, nj, SafeInterval(), m.g)
                valid.append(move_copy)
        return valid
    

class Constraints:
    def __init__(self, width: int, height: int, agentsize: float = 0.5, inflateintervals: float = 0.0):
        self.width = width
        self.height = height
        self.agentsize = agentsize
        self.inflateintervals = inflateintervals
        self.resetSafeIntervals(width, height)
        self.constraints: List[List[List[Section]]] = [[[] for _ in range(width)] for _ in range(height)]

    def minDist(self, A: Point, C: Point, D: Point) -> float:
        classA = A.classify(C, D)
        if classA == 3:
            return math.hypot(A.i - C.i, A.j - C.j)
        elif classA == 4:
            return math.hypot(A.i - D.i, A.j - D.j)
        else:
            num = abs((C.i - D.i) * A.j + (D.j - C.j) * A.i + (C.j * D.i - D.j * C.i))
            den = math.hypot(C.i - D.i, C.j - D.j)
        return num / den if den > EPS else math.hypot(A.i - C.i, A.j - C.j)

    def resetSafeIntervals(self, width: int, height: int):
        self.safe_intervals: List[List[List[SafeInterval]]] = [[[] for _ in range(width)] for _ in range(height)]
        for i in range(height):
            for j in range(width):
                self.safe_intervals[i][j].append(SafeInterval(0.0, INF, 0))

    def updateCellSafeIntervals(self, cell: Tuple[int, int]):
        ci, cj = cell
        if len(self.safe_intervals[ci][cj]) > 1:
            return
        los = LineOfSight(self.agentsize)
        cells = los.getCells(ci, cj)
        secs: List[Section] = []
        for (x, y) in cells:
            if 0 <= x < self.height and 0 <= y < self.width:
                for sec in self.constraints[x][y]:
                    if sec not in secs:
                        secs.append(sec)
        for sec in secs:
            radius = self.agentsize + sec.size
            i0, j0, i1, j1, i2, j2 = sec.i1, sec.j1, sec.i2, sec.j2, ci, cj
            if i0 == i1 and j0 == j1 and i0 == i2 and j0 == j2:
                mindist = 0.0
            else:
                mindist = self.minDist(Point(i2, j2), Point(i0, j0), Point(i1, j1))
            if mindist >= radius:
                continue
            dist = abs((i0 - i1) * j2 + (j1 - j0) * i2 + (j0 * i1 - i0 * j1)) / max(EPS, math.hypot(i0 - i1, j0 - j1))
            da = (i0 - i2) * (i0 - i2) + (j0 - j2) * (j0 - j2)
            db = (i1 - i2) * (i1 - i2) + (j1 - j2) * (j1 - j2)
            ha = math.sqrt(max(0.0, da - dist * dist))
            size = math.sqrt(max(0.0, radius * radius - dist * dist))
            interval = SafeInterval()
            cls = Point(i2, j2).classify(Point(i0, j0), Point(i1, j1))
            if cls == 3:
                interval.begin = sec.g1
                interval.end = sec.g1 + (size - ha) / max(EPS, sec.mspeed)     
            elif cls == 4:
                interval.begin = sec.g2 - size / max(EPS, sec.mspeed) + math.sqrt(max(0.0, db - dist * dist)) / max(EPS, sec.mspeed)
                interval.end = sec.g2
            elif da < radius * radius:
                if db < radius * radius:
                    interval.begin = sec.g1
                    interval.end = sec.g2
                else:
                    hb = math.sqrt(max(0.0, db - dist * dist))
                    interval.begin = sec.g1
                    interval.end = sec.g2 - hb / max(EPS, sec.mspeed) + size / max(EPS, sec.mspeed)
            else:
                if db < radius * radius:
                    interval.begin = sec.g1 + ha / max(EPS, sec.mspeed) - size / max(EPS, sec.mspeed)
                    interval.end = sec.g2
                else:
                    interval.begin = sec.g1 + ha / max(EPS, sec.mspeed) - size / max(EPS, sec.mspeed)
                    interval.end = sec.g1 + ha / max(EPS, sec.mspeed) + size / max(EPS, sec.mspeed)
            new_intervals: List[SafeInterval] = []
            iins = False
            for old in self.safe_intervals[ci][cj]:
                if old.end <= interval.begin + EPS or old.begin >= interval.end - EPS:
                    new_intervals.append(old)
                else:
                    if old.begin < interval.begin:
                        new_intervals.append(SafeInterval(old.begin, interval.begin))
                    if old.end > interval.end:
                        new_intervals.append(SafeInterval(interval.end, old.end))
            if new_intervals:
                self.safe_intervals[ci][cj] = new_intervals
            for idx, it in enumerate(self.safe_intervals[ci][cj]):
                it.id = idx       

    def getSafeIntervals(self, curNode: Node, close: Optional[Dict[int, List[Node]]] = None, w: Optional[int] = None) -> List[SafeInterval]:
        return list(self.safe_intervals[curNode.i][curNode.j])

    def addStartConstraint(self, i: int, j: int, size: int, cells: List[Tuple[int, int]], agentsize: float):
        sec = Section(i, j, i, j, 0.0, float(size))
        sec.size = agentsize
        for (ci, cj) in cells:
            self.constraints[ci][cj].insert(0, sec)

    def removeStartConstraint(self, cells: List[Tuple[int, int]], start_i: int, start_j: int):
        for (ci, cj) in cells:
            newlist = []
            for c in self.constraints[ci][cj]:
                if not (c.i1 == start_i and c.j1 == start_j and c.g1 < EPS):
                    newlist.append(c)
            self.constraints[ci][cj] = newlist

    def addConstraints(self, sections: List[Node], size: float, mspeed: float, map_obj: Map):
        los = LineOfSight(size)
        sec = Section(sections[-1].i, sections[-1].j, sections[-1].i, sections[-1].j, sections[-1].g, INF)
        sec.size = size
        sec.mspeed = mspeed
        cells = los.getCellsCrossedByLine(sec.i1, sec.j1, sec.i2, sec.j2, map_obj)
        for (ci, cj) in cells:
            self.constraints[ci][cj].append(sec)
        for a in range(1, len(sections)):
            cells = los.getCellsCrossedByLine(sections[a - 1].i, sections[a - 1].j, sections[a].i, sections[a].j, map_obj)
            s = Section.from_nodes(sections[a - 1], sections[a])
            s.size = size
            s.mspeed = mspeed
            for (ci, cj) in cells:
                self.constraints[ci][cj].append(s)  

    def hasCollision(self, curNode: Node, startTimeA: float, constraint: Section) -> Tuple[bool, bool]:
        endTimeA = startTimeA + (curNode.g - (curNode.parent.g if curNode.parent is not None else 0.0))
        startTimeB = constraint.g1
        endTimeB = constraint.g2
        if startTimeA > endTimeB or startTimeB > endTimeA:
            return False, False
        A = Vector2D(curNode.parent.i if curNode.parent else curNode.i, curNode.parent.j if curNode.parent else curNode.j)
        if curNode.parent:
            denomA = (curNode.g - curNode.parent.g) if (curNode.g - curNode.parent.g) > EPS else EPS
            VA = Vector2D((curNode.i - curNode.parent.i) / denomA, (curNode.j - curNode.parent.j) / denomA)
        else:
            VA = Vector2D(0.0, 0.0)
        B = Vector2D(constraint.i1, constraint.j1)
        denomB = (constraint.g2 - constraint.g1) if (constraint.g2 - constraint.g1) > EPS else EPS
        VB = Vector2D((constraint.i2 - constraint.i1) / denomB, (constraint.j2 - constraint.j1) / denomB)

        if startTimeB > startTimeA:
            A = A + VA * (startTimeB - startTimeA)
            startTimeA = startTimeB
        elif startTimeB < startTimeA:
            B = B + VB * (startTimeA - startTimeB)
            startTimeB = startTimeA

        r = constraint.size + self.agentsize + self.inflateintervals
        w = Vector2D(B.i - A.i, B.j - A.j)
        c = w.dot(w) - r * r
        goal_collision = False
        if c < 0:
            if constraint.g2 == INF:
                goal_collision = True
            return True, goal_collision
        v = Vector2D(VA.i - VB.i, VA.j - VB.j)
        a = v.dot(v)
        b = w.dot(v)
        dscr = b * b - a * c
        if dscr <= 0 or a <= EPS:
            return False, False
        ctime = (b - math.sqrt(dscr)) / a
        if ctime > -EPS and ctime < min(endTimeB, endTimeA) - startTimeA + EPS:
            if constraint.g2 == INF:
                goal_collision = True
            return True, goal_collision
        return False, False   