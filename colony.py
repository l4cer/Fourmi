import numpy as np

from maze import Maze

from constants import *


class Colony:
    def __init__(self, num_ants: int, pos_nest: np.ndarray) -> None:
        self.num_ants = num_ants
        self.pos_nest = pos_nest.astype(np.int16)

        self.is_loaded = np.zeros(num_ants, dtype=bool)

        self.cur_age = np.zeros(num_ants, dtype=np.int32)
        self.max_age = np.random.uniform(
            0.75 * MAX_AGE, MAX_AGE, num_ants).astype(np.int32)

        self.path = np.zeros((num_ants, MAX_AGE + 1, 2), dtype=np.int16)
        self.path[:, 0, 0] = pos_nest[0]
        self.path[:, 0, 1] = pos_nest[1]

        self.direction = -np.ones(num_ants, dtype=np.int8)

    @property
    def pos_ants(self) -> np.ndarray:
        return self.path[np.arange(self.num_ants), self.cur_age, :]

    @property
    def pose_ants(self) -> np.ndarray:
        return np.concatenate(
            (self.pos_ants, self.direction[:, np.newaxis]), axis=1)

    def return_nest(self) -> None:
        self.cur_age[self.is_loaded] -= 1

        tmp = self.pos_ants[self.is_loaded] == self.pos_nest
        in_nest = np.nonzero(self.is_loaded)[0][tmp[:, 0] & tmp[:, 1]]

        self.cur_age[in_nest] = 0
        self.is_loaded[in_nest] = False

        self.food_collected = len(in_nest)

    def explore(self, maze: Maze, pos_food: np.ndarray) -> None:
        cur_pos = self.pos_ants
        cur_cell = maze.map[cur_pos[:, 0], cur_pos[:, 1]]

        pheromones_N = maze.pheromones[cur_pos[:, 0] + 0, cur_pos[:, 1] + 1]
        pheromones_E = maze.pheromones[cur_pos[:, 0] + 1, cur_pos[:, 1] + 2]
        pheromones_W = maze.pheromones[cur_pos[:, 0] + 1, cur_pos[:, 1] + 0]
        pheromones_S = maze.pheromones[cur_pos[:, 0] + 2, cur_pos[:, 1] + 1]

        exit_N = (cur_cell & maze.NORTH) > 0
        exit_E = (cur_cell & maze.EAST) > 0
        exit_W = (cur_cell & maze.WEST) > 0
        exit_S = (cur_cell & maze.SOUTH) > 0

        pheromones_N *= exit_N
        pheromones_E *= exit_E
        pheromones_W *= exit_W
        pheromones_S *= exit_S

        max_NS = np.maximum(pheromones_N, pheromones_S)
        max_EW = np.maximum(pheromones_E, pheromones_W)
        max_pheromone = np.maximum(max_NS, max_EW)

        rand = np.random.uniform(0.0, 1.0, self.num_ants)
        explore = (rand <= EPSILON) | (max_pheromone == 0.0)

        exploring_ants = np.nonzero(  explore  & (~self.is_loaded))[0]
        following_ants = np.nonzero((~explore) & (~self.is_loaded))[0]

        if len(exploring_ants) > 0:
            num_exits = (
                exit_N.astype(np.int8) +
                exit_E.astype(np.int8) +
                exit_W.astype(np.int8) +
                exit_S.astype(np.int8)
            )

            while len(exploring_ants) > 0:
                direction = np.random.randint(0, 4, len(exploring_ants))

                old_pos = self.pos_ants[exploring_ants]
                new_pos = np.copy(old_pos)

                new_pos[:, 0] -= (direction == DIR_N) & exit_N[exploring_ants]
                new_pos[:, 1] += (direction == DIR_E) & exit_E[exploring_ants]
                new_pos[:, 1] -= (direction == DIR_W) & exit_W[exploring_ants]
                new_pos[:, 0] += (direction == DIR_S) & exit_S[exploring_ants]

                valid = (
                    ((new_pos[:, 0] != old_pos[:, 0]) |
                     (new_pos[:, 1] != old_pos[:, 1])) &
                    ((num_exits[exploring_ants] == 1) |
                     (direction + self.direction[exploring_ants] != 3))
                )

                loc_valid = exploring_ants[valid]

                self.path[loc_valid, self.cur_age[loc_valid] + 1, :] = new_pos[valid, :]
                self.direction[loc_valid] = direction[valid]

                exploring_ants = exploring_ants[~valid]

        self.cur_age[~self.is_loaded] += 1

        if len(following_ants) > 0:
            max_is_N = (pheromones_N == max_pheromone)[following_ants]
            max_is_E = (pheromones_E == max_pheromone)[following_ants]
            max_is_W = (pheromones_W == max_pheromone)[following_ants]
            max_is_S = (pheromones_S == max_pheromone)[following_ants]

            self.path[following_ants, self.cur_age[following_ants], :] = (
                self.path[following_ants, self.cur_age[following_ants] - 1, :]
            )

            self.path[following_ants, self.cur_age[following_ants], 0] -= max_is_N
            self.path[following_ants, self.cur_age[following_ants], 1] += max_is_E
            self.path[following_ants, self.cur_age[following_ants], 1] -= max_is_W
            self.path[following_ants, self.cur_age[following_ants], 0] += max_is_S

        # Dying ants mask
        dying_mask = self.cur_age == self.max_age

        self.cur_age[dying_mask] = 0
        self.path[dying_mask, 0, :] = self.pos_nest
        self.direction[dying_mask] = -1

        cur_pos = self.pos_ants

        # Ants in food mask
        in_food_mask = (
            (cur_pos[:, 0] == pos_food[0]) &
            (cur_pos[:, 1] == pos_food[1])
        )

        self.is_loaded[in_food_mask] = True

    def update(self, maze: Maze, pos_food: np.ndarray) -> np.ndarray:
        self.return_nest()
        self.explore(maze, pos_food)

        cur_pos = self.pos_ants
        cur_cell = maze.map[cur_pos[:, 0], cur_pos[:, 1]]

        pheromones_N = maze.pheromones[cur_pos[:, 0] + 0, cur_pos[:, 1] + 1]
        pheromones_E = maze.pheromones[cur_pos[:, 0] + 1, cur_pos[:, 1] + 2]
        pheromones_W = maze.pheromones[cur_pos[:, 0] + 1, cur_pos[:, 1] + 0]
        pheromones_S = maze.pheromones[cur_pos[:, 0] + 2, cur_pos[:, 1] + 1]

        pheromones_N *= (cur_cell & maze.NORTH) > 0
        pheromones_E *= (cur_cell & maze.EAST) > 0
        pheromones_W *= (cur_cell & maze.WEST) > 0
        pheromones_S *= (cur_cell & maze.SOUTH) > 0

        mean_pheromones = pheromones_N + pheromones_E + pheromones_W + pheromones_S
        mean_pheromones = 0.25 * mean_pheromones

        max_NS = np.maximum(pheromones_N, pheromones_S)
        max_EW = np.maximum(pheromones_E, pheromones_W)
        max_pheromones = np.maximum(max_NS, max_EW)

        pheromones = np.copy(maze.pheromones)
        pheromones[cur_pos[:, 0] + 1, cur_pos[:, 1] + 1] = (
            ALPHA * max_pheromones + (1 - ALPHA) * mean_pheromones
        )

        return BETA * pheromones
