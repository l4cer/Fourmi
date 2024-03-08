import numpy as np


class Maze:
    NORTH = 1
    EAST  = 2
    SOUTH = 4
    WEST  = 8

    def __init__(self, width: int, height: int) -> None:
        self.map = np.zeros((width, height), dtype=np.int8)

        is_visited = np.zeros((width, height), dtype=bool)

        # We choose the central cell as the initial cell
        historic = [(width // 2, height // 2)]

        while len(historic) > 0:
            neighbours = []
            directions = []

            cur_pos = historic[-1]
            is_visited[cur_pos] = True

            if 0 < cur_pos[0]:
                if not is_visited[cur_pos[0] - 1, cur_pos[1]]:
                    neighbours.append((cur_pos[0] - 1, cur_pos[1]))
                    directions.append((Maze.NORTH, Maze.SOUTH))

            if cur_pos[0] < width - 1:
                if not is_visited[cur_pos[0] + 1, cur_pos[1]]:
                    neighbours.append((cur_pos[0] + 1, cur_pos[1]))
                    directions.append((Maze.SOUTH, Maze.NORTH))

            if 0 < cur_pos[1]:
                if not is_visited[cur_pos[0], cur_pos[1] - 1]:
                    neighbours.append((cur_pos[0], cur_pos[1] - 1))
                    directions.append((Maze.WEST, Maze.EAST))

            if cur_pos[1] < height - 1:
                if not is_visited[cur_pos[0], cur_pos[1] + 1]:
                    neighbours.append((cur_pos[0], cur_pos[1] + 1))
                    directions.append((Maze.EAST, Maze.WEST))

            if len(neighbours) > 0:
                is_visited[cur_pos] = True

                neighbours = np.array(neighbours)
                directions = np.array(directions)

                chosen = np.random.randint(0, len(neighbours))
                historic.append((neighbours[chosen, 0], neighbours[chosen, 1]))

                self.map[cur_pos] |= directions[chosen][0]
                self.map[tuple(neighbours[chosen])] |= directions[chosen][1]

            else:
                historic.pop()

        # Put borders in the maze
        self.map[ :,  0] &= 15 - Maze.WEST
        self.map[ :, -1] &= 15 - Maze.EAST
        self.map[ 0,  :] &= 15 - Maze.NORTH
        self.map[-1,  :] &= 15 - Maze.SOUTH

        self.pheromones = np.zeros((width + 2, height + 2), dtype=np.float32)
