"""
Construit un labyrinthe à** deux dimensions où chaque case du labyrinthe est décrite par la somme des sorties existantes 
(Nord = 1, Est = 2, Sud = 4, Ouest = 8), chaque case correspondant à une valeur stocké dans un tableau à deux dimensions.la


Creates a two-dimensional maze where each cell in the maze is defined by the sum of existing exits 
(North = 1, East = 2, South = 4, West = 8), with each cell corresponding to a value stored in a two-dimensional array.
"""
import numpy as np
import pygame as pg

NORTH = 1
EAST  = 2
SOUTH = 4
WEST  = 8

casesImg = []

def buildMaze( dimensions, seed ) :
    """
        Construit un labyrinthe de dimensions dimensions en retournant le tableau numpy décrivant ce labyrinthe.
        En entrée :
            dimensions : Tuple contenant deux entiers décrivant la hauteur et la longueur du labyrinthe
            seed       : La graîne aléatoire permettant de générer le labyrinthe. Une même graîne donne le même labyrinthe.
    """
    maze = np.zeros(dimensions,dtype=np.int8)
    isVisited = np.zeros(dimensions,dtype=np.int8)
    historic = []

    # On choisit comme cellule initiale la cellule centrale/Choose central cell as initial cell
    curInd = (dimensions[0]//2,dimensions[1]//2)
    historic.append(curInd)
    while (len(historic) > 0):
        curInd = historic[-1]
        isVisited[curInd] = 1
        # On regarde en premier lieu si il existe au moins une cellule non visitée voisine de la cellule
        # courante :
        #   1. Calcul des voisins de la cellule courante :
        neighbours        = []
        neighboursVisited = []
        direction         = []
        if curInd[1] > 0 and isVisited[curInd[0], curInd[1]-1] == 0 : # Cellule Ouest non visitée/West cell no visited
            neighbours.append((curInd[0], curInd[1]-1)) 
            direction.append((WEST,EAST))
        if curInd[1] < dimensions[1]-1 and isVisited[curInd[0], curInd[1]+1] == 0 : # Cellule Est/East cell
            neighbours.append((curInd[0], curInd[1]+1)) 
            direction.append((EAST,WEST))
        if curInd[0] < dimensions[0]-1 and isVisited[curInd[0]+1, curInd[1]]== 0 : # Cellule Sud/South cell
            neighbours.append((curInd[0]+1, curInd[1])) 
            direction.append((SOUTH,NORTH))
        if curInd[0] > 0 and isVisited[curInd[0]-1, curInd[1]] == 0 : # Cellule Nord/North cell
            neighbours.append((curInd[0]-1, curInd[1])) 
            direction.append((NORTH,SOUTH))
        if len(neighbours) > 0 : # Dans ce cas, au moins une cellule est non visitée/In this case, a cell is non visited
            neighbours = np.array(neighbours)
            direction  = np.array( direction)
            seed = (16807*seed)%2147483647
            chosenDir = seed%len(neighbours)
            dir       = direction[chosenDir]
            historic.append((neighbours[chosenDir,0],neighbours[chosenDir,1]))
            maze[curInd] |= dir[0]
            maze[neighbours[chosenDir,0],neighbours[chosenDir,1]] |= dir[1]
            isVisited[curInd] = 1
        else:
            historic.pop()
    # 
    return maze

def initCases():
    """

    """
    img = pg.image.load("cases.png").convert_alpha()
    for i in range(0,128,8):
        casesImg.append(pg.Surface.subsurface(img, i, 0, 8, 8))
    print(f"Nb cases img : {len(casesImg)}")

def displayMaze( tMaze : np.ndarray ):
    """
    Créer une image du labyrinthe
    """
    mazeImg = pg.Surface((8*tMaze.shape[1],8*tMaze.shape[0]), flags=pg.SRCALPHA)
    for i in range(tMaze.shape[0]):
        for j in range(tMaze.shape[1]):
            mazeImg.blit(casesImg[tMaze[i,j]], (j*8, i*8))

    return mazeImg

if __name__  == "__main__":
    import time
    pg.init()
    initCases()
    t1 = time.time()
    maze = buildMaze( (50,80), 12345)
    t2 = time.time()
    print(f"Temps construction labyrinthe : {t2-t1} secondes")
    resolution = maze.shape[1]*8,maze.shape[0]*8
    print(f"resolution : {resolution}")
    screen = pg.display.set_mode(resolution)

    mazeImg = displayMaze(maze)
    screen.blit(mazeImg, (0, 0))
    pg.display.update()

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
