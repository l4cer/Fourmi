import os

# Hide PyGame welcome message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import time

import numpy as np
import pygame as pg

from mpi4py import MPI

from maze import Maze
from colony import Colony

from display import display

from constants import *


# Initializes the communication
comm = MPI.COMM_WORLD.Dup()
nbp  = comm.size
rank = comm.rank


def main(seed: int = 123) -> None:
    if nbp < 2:
        if rank == 0:
            print("At least two processes must be used")

        return

    np.random.seed(seed)

    if rank == 0:
        pg.init()
        screen = pg.display.set_mode((8 * HEIGHT, 8 * WIDTH))

    pos_nest = np.array([0, 0])
    pos_food = np.array([WIDTH - 1, HEIGHT - 1])

    maze = Maze(WIDTH, HEIGHT)
    np.random.seed(seed * (rank + 1))

    per_rank = NUM_ANTS // (nbp - 1)

    loc_pose = np.zeros((per_rank, 3), dtype=np.int16)

    glb_pose = np.zeros((nbp * per_rank, 3), dtype=np.int16)
    glb_pose[:, :2] = pos_nest.astype(np.int16)

    loc_pheromones = np.zeros(
        (maze.pheromones.shape[0], maze.pheromones.shape[1])
    )

    glb_pheromones = np.zeros(
        (nbp, maze.pheromones.shape[0], maze.pheromones.shape[1])
    )

    colony = Colony(per_rank, pos_nest) if rank > 0 else None

    running = True
    food_counter = 0

    start = time.time()

    while running:
        if rank == 0:
            display(screen, maze, glb_pose[per_rank:])
            pg.display.update()

        if rank > 0:
            loc_pheromones = colony.update(maze, pos_food)
            loc_pose = colony.pose_ants

        comm.Gather(loc_pose, glb_pose, root=0)
        comm.Allgather(loc_pheromones, glb_pheromones)

        maze.pheromones = np.max(glb_pheromones, axis=0)
        maze.pheromones[pos_food[0] + 1, pos_food[1] + 1] = 1.0

        collected = colony.food_collected if rank > 0 else 0
        collected = comm.reduce(collected, op=MPI.SUM, root=0)

        if rank == 0:
            food_counter += collected

            if food_counter >= 1000:
                running = False

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False

        # Notifies game running status
        running = comm.scatter([running] * nbp, root=0)

    if rank == 0:
        elapsed = time.time() - start
        print(f"{elapsed:.4f} s until collect {food_counter} foods")

        pg.image.save(screen, f"sim_{nbp}_{100.0 * elapsed:.0f}_ms.png")
        pg.quit()


if __name__ == "__main__":
    main()
