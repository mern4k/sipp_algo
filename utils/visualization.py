from utils.search_tree import SippNode
from utils.map import Map
from utils.sipp import DynamicObstacle
from typing import List
from random import randint
from PIL import Image, ImageDraw, ImageFont
import numpy as np

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