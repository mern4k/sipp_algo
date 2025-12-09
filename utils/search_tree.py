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