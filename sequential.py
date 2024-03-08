import os

# Hide PyGame welcome message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import time

import numpy as np
import pygame as pg

from maze import Maze
from colony import Colony

from display import display

from constants import *


def main(seed: int = 123) -> None:
    np.random.seed(seed)

    pg.init()
    screen = pg.display.set_mode((8 * HEIGHT, 8 * WIDTH))

    pos_nest = np.array([0, 0])
    pos_food = np.array([WIDTH - 1, HEIGHT - 1])

    maze = Maze(WIDTH, HEIGHT)
    colony = Colony(NUM_ANTS, pos_nest)

    running = True
    food_counter = 0

    elapsed = np.array([])

    while running:
        start = time.time()
        display(screen, maze, colony.pose_ants)
        pg.display.update()

        maze.pheromones = colony.update(maze, pos_food)

        collected = colony.food_collected
        food_counter += collected

        elapsed = np.append(elapsed, time.time() - start)

        print(f"{(1.0 / elapsed).mean():.2f} FPS | food: {food_counter}", end="\r")

        if food_counter >= 10_000:
            running = False

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

    print(f"{(1.0 / elapsed).mean():.2f} FPS | food: {food_counter}")

    pg.image.save(screen, f"sim_1_{(1.0 / elapsed).mean():.0f}.png")
    pg.quit()


if __name__ == "__main__":
    main()
