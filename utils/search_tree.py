from heapq import heappop, heappush
from typing import Optional, Union


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
    def __init__(self, i, j, g=0, h=0, f=None, parent=None, interval=None):
        super().__init__(i=i, j=j, g=g, h=h, f=f, parent=parent)
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


class SippNodeDublicate(SippNode):
    def __init__(self, i, j, isOptimal, g=0, h=0, f=None, parent=None, interval=None):
        super().__init__(i=i, j=j, g=g, h=h, f=f, parent=parent, interval=interval)
        self.isOptimal = isOptimal
    def __hash__(self): 
        return hash((self.i, self.j, self.interval, self.isOptimal))
    def __eq__(self, other): 
        return self.i == other.i and self.j == other.j and self.interval == other.interval and self.isOptimal == other.isOptimal
    

class SearchTreePQD:
    def __init__(self):
        self._open, self._closed, self._enc_open_dublicates = [], {}, 0
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
            self._enc_open_dublicates += 1
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
        return self._enc_open_dublicates
    
    
class SearchTreePQDReexp(SearchTreePQD):
    def __init__(self):
        super().__init__()
        self._reexpanded = set() 
        self._reopened = set() 
        self._number_of_reexpansions = 0  

    def add_to_open(self, item: SippNode):
        if item in self._closed:
            if self._closed[item] > item:
                del self._closed[item]
                self._reopened.add(item)
            else:
                return
        heappush(self._open, item)

    def get_best_node_from_open(self) -> Optional[SippNode]:
        while self._open:
            best_node = heappop(self._open)
            if best_node in self._closed:
                self._enc_open_dublicates += 1
                continue
            if best_node in self._reopened:
                self._reexpanded.add(best_node)
                self._number_of_reexpansions += 1
            return best_node
        return None

    def was_expanded(self, item: SippNode) -> bool:
        return (item in self._closed or item in self._reopened)

    @property
    def reexpanded(self):
        return self._reexpanded

    @property
    def number_of_reexpansions(self):
        return self._number_of_reexpansions
    

class SearchTreePQDFocal(SearchTreePQDReexp):
    def __init__(self, heuristic_func: callable, w: float, goal_i, goal_j):
        super().__init__()
        self._heuristic_func = heuristic_func
        self._w = w
        self._focal = []
        self._focal_bound = float('inf')
        self._goal_i = goal_i
        self._goal_j = goal_j

    def focal_is_empty(self) -> bool: 
        return not self._focal
    
    def _expand_focal(self):
        if self.open_is_empty() or not self.focal_is_empty():
            return
        new_bound = self._w * self._open[0].f
        if new_bound <= self._focal_bound:
            return
        self._focal_bound = new_bound
        for item in self._open:
            if item.f <= self._focal_bound and item not in self._closed:
                heappush(self._focal, (self._heuristic_func(item.i, item.j, self._goal_i, self._goal_j), item))

    def add_to_open(self, item: SippNode):
        if item in self._closed:
            if self._closed[item] > item:
                del self._closed[item]
                self._reopened.add(item)
            else:
                return
        heappush(self._open, item)
        if item.f <= self._focal_bound:
            heappush(self._focal, (self._heuristic_func(item.i, item.j, self._goal_i, self._goal_j), item))

    def get_best_node_from_open(self) -> Optional[SippNode]:
        while self._open:
            if self.focal_is_empty():
                self._expand_focal()
                if self.focal_is_empty():
                    return None
            best_node = heappop(self._focal)[1]
            if best_node in self._closed:
                self._enc_open_dublicates += 1
                continue
            if best_node in self._reopened:
                self._reexpanded.add(best_node)
                self._number_of_reexpansions += 1
            return best_node
        return None