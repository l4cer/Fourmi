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

comp_comm = comm.Split(rank != 0, rank)
comp_nbp  = comp_comm.Get_size()
comp_rank = comp_comm.Get_rank()


def main(seed: int = 123) -> None:
    if nbp < 2:
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

    per_rank = NUM_ANTS // comp_nbp

    pose = np.zeros((comp_nbp * per_rank, 3), dtype=np.int16)
    pose[:, :2] = pos_nest.astype(np.int16)

    colony = Colony(per_rank, pos_nest) if rank > 0 else None

    running = True
    food_counter = 0

    elapsed = np.array([])

    while running:
        if rank == 0:
            start = time.time()
            display(screen, maze, pose)
            pg.display.update()

        if rank > 0:
            comp_comm.Gather(colony.pose_ants, pose, root=0)
            comp_comm.Allreduce(
                colony.update(maze, pos_food), maze.pheromones, op=MPI.MAX)

            collected = comp_comm.reduce(
                colony.food_collected, op=MPI.SUM, root=0)

        if rank == 1:
            comm.Send(pose, dest=0, tag=1)
            comm.Send(maze.pheromones, dest=0, tag=2)
            comm.send(collected, dest=0, tag=3)

        if rank == 0:
            comm.Recv(pose, source=1, tag=1)
            comm.Recv(maze.pheromones, source=1, tag=2)

            food_counter += comm.recv(source=1, tag=3)
            elapsed = np.append(elapsed, time.time() - start)

            print(f"{(1.0 / elapsed).mean():.2f} FPS | food: {food_counter}", end="\r")

            if food_counter >= 10_000:
                running = False

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False

        # Notifies game running status
        running = comm.scatter([running] * nbp, root=0)

    if rank == 0:
        print(f"{(1.0 / elapsed).mean():.2f} FPS | food: {food_counter}")

        pg.image.save(screen, f"sim_{nbp}_{(1.0 / elapsed).mean():.0f}.png")
        pg.quit()


if __name__ == "__main__":
    main()
