import os
from typing import  List, Tuple
import numpy as np
import numpy.typing as npt
from utils.search_tree import SippNode, SearchTreePQD
from utils.map import Map
from utils.sipp import DynamicObstacle, sipp
from utils.visualization import create_animation

def manhattan_dist(i1, j1, i2, j2):
        return abs(i1 - i2) + abs(j1 - j2)

def convert_string_to_cells(cell_str: str) -> npt.NDArray:
    lines = cell_str.replace(" ", "").split("\n")
    return np.array([[1 if char == "#" else 0 for char in line] for line in lines if line], dtype=np.int8)

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
    dynamic_obstacles = [DynamicObstacle(obs) for obs in read_lists_from_file(os.path.join("samples/obstacles.txt"))]

    path_found, last_node, steps, tree_size, open_nodes, closed_nodes = sipp(
        test_map, start_node.i, start_node.j, goal_node.i, goal_node.j, 
        dynamic_obstacles, SearchTreePQD, manhattan_dist
    )
    
    if path_found:
        path = []
        curr = last_node
        while curr is not None:
            path.append(curr)
            curr = curr.parent
        path.reverse()

        correct = last_node.arrival_time == durations[test_index]
        
        print(f"Path found. Arrival time: {last_node.arrival_time}. Steps: {steps}. Nodes created: {tree_size}. Correct: {correct}")
        create_animation("out/sipp_animation.gif", test_map, start_node, goal_node, path, dynamic_obstacles)
    else:
        print("Path not found.")