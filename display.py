import numpy as np
import pygame as pg

from maze import Maze


ant_sprite = None
image_maze = None


def init(maze: Maze) -> None:
    global ant_sprite, image_maze

    image = pg.image.load("sprites/ants.png").convert_alpha()
    ant_sprite = [
        pg.Surface.subsurface(image, 8 * i, 0, 8, 8) for i in range(4)]

    image = pg.image.load("sprites/cases.png").convert_alpha()
    cases = [
        pg.Surface.subsurface(image, 8 * i, 0, 8, 8) for i in range(16)]

    image_maze = pg.Surface(
        (8 * maze.map.shape[1], 8 * maze.map.shape[0]), flags=pg.SRCALPHA)

    for i in range(maze.map.shape[0]):
        for j in range(maze.map.shape[1]):
            image_maze.blit(cases[maze.map[i, j]], (8 * j, 8 * i))


def display(screen: pg.Surface, maze: Maze, pose_ants: np.ndarray) -> pg.Surface:
    global ant_sprite, image_maze

    if ant_sprite is None or image_maze is None:
        init(maze)

    for i in range(maze.map.shape[0]):
        for j in range(maze.map.shape[1]):
            t = maze.pheromones[i + 1, j + 1]**0.08

            blue = np.array([0, 0, 128])
            yellow = np.array([255, 255, 128])

            screen.fill(blue * (1 - t) + yellow * t, (8 * j, 8 * i, 8, 8))

    for pose in pose_ants:
        screen.blit(
            ant_sprite[pose[2]], (8 * pose[1], 8 * pose[0]))

    screen.blit(image_maze, (0, 0))
