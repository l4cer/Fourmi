"""
Module managing an ant colony in a labyrinth.
"""
import numpy as np
import labyrinthe
import pheromone
import direction as d
import pygame as pg

UNLOADED, LOADED = False, True

exploration_coefs = 0.

class Colony:
    """
    Représente une colonie de fourmis. Les fourmis ne sont pas individualisées
    par soucis de performance !
    
    Entrées :
        nbAnts : Nombre de fourmis contenus dans la fourmilière
        posInit: Positions initiales des fourmis (position de la fourmilière)
        maxLife: Vie maximale que peut atteindre les fourmis

    Represent an ant colony. Ants are not individualized for performance 
    reasons!

    Inputs :
        nbAnts  : Number of ants in the anthill
        posInit : Initial positions of ants (anthill position)
        maxLife : Maximum life that ants can reach

    """
    def __init__( self, nbAnts, posInit, maxLife ):
        # Graîne aléatoire attribuée à chaque fourmi : doit être unique par
        # fourmi
        self.seeds = np.arange(1,nbAnts+1, dtype=np.int64) 
        # Etat de la fourmi : est chargée ou est non chargée
        self.isLoaded = np.zeros(nbAnts,dtype=np.int8)
        # Calcul de la vie maximale de chaque fourmi :
        #   Màj de la graine aléatoire :
        self.seeds[:] = np.mod(16807*self.seeds[:], 2147483647)
        # Calcul pour chaque fourmi de sa vie maximale = 75% à 100% de la vie
        # maximale des fourmis
        self.maxLife  = maxLife * np.ones(nbAnts, dtype=np.int32)
        self.maxLife -= np.int32(maxLife*(self.seeds/2147483647.))//4
        # L'âge des fourmis : toutes à zéro au début :
        self.age = np.zeros( nbAnts, dtype=np.int64)
        # Historique du chemin parcouru pour chaque fourmi. La position se
        # trouvant à l'age de la fourmi donne sa position actuelle
        self.historicPath = np.zeros( (nbAnts, maxLife+1, 2), dtype=np.int16)
        self.historicPath[:, 0, 0] = posInit[0]
        self.historicPath[:, 0, 1] = posInit[1]
        # Direction dans laquelle se trouve la fourmi actuellement (dépend de
        # la direction d'où elle vient)
        self.directions = d.DIR_NONE*np.ones(nbAnts, dtype=np.int8)
        self.sprites = []
        img = pg.image.load("ants.png").convert_alpha()
        for i in range(0,32,8):
            self.sprites.append(pg.Surface.subsurface(img, i, 0, 8, 8))

    def returnToNest( self, loadedAnts, posNest, foodCounter ):
        """
        Fonction faisant retourner les fourmis chargées de nourriture vers
        leurs nids.

        Entrées :
            loadedAnts : Indices des fourmis étant chargées de nourriture
            posNest    : Position du nid où doit se rendre les fourmis
            foodCounter: La quantité courante de nourriture dans le nid

        Retourne la nouvelle quantité de nourriture

        Function that returns the ants carrying food to their nests.

        Inputs :
            loadedAnts: Indices of ants carrying food
            posNest: Position of the nest where ants should go
            foodCounter: Current quantity of food in the nest

        Returns the new quantity of food
        """
        self.age[loadedAnts] -= 1

        inNestTmp = \
            self.historicPath[loadedAnts,self.age[loadedAnts], :] == posNest
        if inNestTmp.any():
            inNestLoc = np.nonzero(
                np.logical_and(inNestTmp[:,0],inNestTmp[:,1])
                                  )[0]
            if inNestLoc.shape[0] > 0:
                inNest = loadedAnts[inNestLoc]
                self.isLoaded[inNest] = UNLOADED
                self.age[inNest] = 0
                foodCounter += inNestLoc.shape[0]
        return foodCounter

    def explore( self, unloadedAnts, maze, posFood, posNest, pheromones, 
                foodCounter = 0 ):
        """
        Gestion des fourmis non chargées qui explorent le labyrinthe.

        En entrée :
            unloadedAnts : indices des fourmis qui ne sont pas chargées
            maze         : Le labyrinthe dans lequel évoluent les fourmis
            posFood      : Position de la nourriture dans le labyrinthe
            posNest      : Position du nid des fourmis dans le labyrinthe
            pheromones   : La carte des phéromones (qui ont également des 
                           cellules fantômes pour gestion plus facile des 
                           bords)
            foodCounter  : La quantité de nourriture dans le nid

        En sortie : None

        Management of unloaded ants exploring the maze.

        Inputs:
            unloadedAnts: Indices of ants that are not loaded
            maze        : The maze in which ants move
            posFood     : Position of food in the maze
            posNest     : Position of the ants' nest in the maze
            pheromones  : The pheromone map (which also has ghost cells for 
                          easier edge management)
            foodCounter : The quantity of food in the nest

        Outputs: None
        """
        # Mise à jour de la graîne aléatoire (pour le pseudo-random fait main)
        # appliquée à toutes les fourmis non chargées
        self.seeds[unloadedAnts] = np.mod(16807*self.seeds[unloadedAnts], 2147483647)

        # Calcul des sorties possibles pour chaque fourmi dans le labyrinthe:
        oldPosAnts = self.historicPath[range(0,self.seeds.shape[0]),self.age[:], :]
        hasNorthExit = np.bitwise_and(maze[oldPosAnts[:,0],oldPosAnts[:,1]], labyrinthe.NORTH)>0
        hasEastExit  = np.bitwise_and(maze[oldPosAnts[:,0],oldPosAnts[:,1]], labyrinthe.EAST )>0
        hasSouthExit = np.bitwise_and(maze[oldPosAnts[:,0],oldPosAnts[:,1]], labyrinthe.SOUTH)>0
        hasWestExit  = np.bitwise_and(maze[oldPosAnts[:,0],oldPosAnts[:,1]], labyrinthe.WEST )>0

        # Lecture des pheronomes voisins :
        northPos = np.copy(oldPosAnts)
        northPos[:,1] += 1
        northPheromone = pheromones[northPos[:,0],northPos[:,1]]*hasNorthExit

        eastPos = np.copy(oldPosAnts)
        eastPos[:,0] += 1
        eastPos[:,1] += 2
        eastPheromone = pheromones[eastPos[:,0],eastPos[:,1]]*hasEastExit

        southPos = np.copy(oldPosAnts)
        southPos[:,0] += 2
        southPos[:,1] += 1
        southPheromone = pheromones[southPos[:,0],southPos[:,1]]*hasSouthExit

        westPos = np.copy(oldPosAnts)
        westPos[:,0] += 1
        westPheromone = pheromones[westPos[:,0],westPos[:,1]]*hasWestExit

        maxPheromones = np.maximum(northPheromone, eastPheromone )
        maxPheromones = np.maximum(maxPheromones , southPheromone)
        maxPheromones = np.maximum(maxPheromones , westPheromone )

        # Calcul des choix pour toutes les fourmis non chargées de nourriture
        # (pour les autres, on calcule mais pas grave):
        choix = self.seeds[:]/2147483647.

        # Les fourmis explorent le labyrinthe par choix ou si aucun phéromone
        # ne peut les guider :
        indExploringAnts = np.nonzero(
            np.logical_or(
                choix[unloadedAnts] <= exploration_coefs,
                maxPheromones[unloadedAnts]==0.
                         )
                                    ) [0]
        if indExploringAnts.shape[0] > 0:
            indExploringAnts = unloadedAnts[indExploringAnts]
            validMoves = np.zeros(choix.shape[0], np.int8)
            nbExists = hasNorthExit*np.ones(hasNorthExit.shape) + \
                       hasEastExit *np.ones(hasEastExit.shape)  + \
                       hasSouthExit*np.ones(hasSouthExit.shape) + \
                       hasWestExit *np.ones(hasWestExit.shape)
            while np.any(validMoves[indExploringAnts]==0):
                # Calcul des indices des fourmis dont le dernier mouvement 
                # n'était pas valide :
                indAntsToMove = \
                    indExploringAnts[validMoves[indExploringAnts] == 0]
                #print(f"indAntsToMove : {indAntsToMove}")
                self.seeds[:] = np.mod(16807*self.seeds[:], 2147483647)
                # On choisit une direction au hasard :
                dir = np.mod(self.seeds[indAntsToMove],4)
                oldPos = \
                    self.historicPath[indAntsToMove,
                                      self.age[indAntsToMove],:]
                newPos = np.copy(oldPos)
                newPos[:,1] -= np.logical_and(
                    dir==d.DIR_WEST, 
                    hasWestExit[indAntsToMove] ) * \
                                np.ones(newPos.shape[0],dtype=np.int16)
                newPos[:,1] += np.logical_and(
                    dir==d.DIR_EAST, 
                    hasEastExit[indAntsToMove] ) * \
                                np.ones(newPos.shape[0],dtype=np.int16)
                newPos[:,0] -= np.logical_and(
                    dir==d.DIR_NORTH, 
                    hasNorthExit[indAntsToMove] ) * \
                                np.ones(newPos.shape[0],dtype=np.int16)
                newPos[:,0] += np.logical_and(
                    dir==d.DIR_SOUTH, 
                    hasSouthExit[indAntsToMove] ) * \
                    np.ones(newPos.shape[0],dtype=np.int16)
                # Mouvement valide si on n'est pas resté sur place à cause 
                # d'un mur
                validMoves[indAntsToMove] = \
                    np.logical_or(
                        newPos[:,0] != oldPos[:,0],
                        newPos[:,1] != oldPos[:, 1] )
                # et si on n'est pas dans la direction opposée à la direction
                # précédente (et si il existe d'autres sorties)
                validMoves[indAntsToMove] = np.logical_and(
                    validMoves[indAntsToMove],
                    np.logical_or(
                        dir != 3-self.directions[indAntsToMove], 
                        nbExists[indAntsToMove]==1
                                 )                        )
                # Calcul des indices des fourmis dont on vient de valider le
                # mouvement :
                indValidMoves = indAntsToMove[
                    np.nonzero(validMoves[indAntsToMove])[0]
                                             ]
                # Pour ces fourmis, on met à jour leurs positions et leurs
                # directions
                self.historicPath[indValidMoves,
                                  self.age[indValidMoves]+1,:] = \
                    newPos[validMoves[indAntsToMove]==1,:]
                self.directions[indValidMoves] = dir[
                    validMoves[indAntsToMove]==1    ]

        indFollowingAnts = np.nonzero(
            np.logical_and(choix[unloadedAnts] > exploration_coefs,
                           maxPheromones[unloadedAnts]>0.)
                                     )[0]
        if indFollowingAnts.shape[0] > 0:
            indFollowingAnts = unloadedAnts[indFollowingAnts]
            self.historicPath[indFollowingAnts, 
                              self.age[indFollowingAnts]+1,:] = \
                self.historicPath[indFollowingAnts, 
                                  self.age[indFollowingAnts],:]
            maxEast = (eastPheromone[indFollowingAnts] == \
                       maxPheromones[indFollowingAnts])
            self.historicPath[indFollowingAnts, 
                              self.age[indFollowingAnts]+1,1] += \
                maxEast * np.ones(indFollowingAnts.shape[0],dtype=np.int16)
            maxWest = (westPheromone[indFollowingAnts] == \
                       maxPheromones[indFollowingAnts])
            self.historicPath[indFollowingAnts, 
                              self.age[indFollowingAnts]+1,1] -= \
                maxWest * np.ones(indFollowingAnts.shape[0],dtype=np.int16)
            maxNorth = (northPheromone[indFollowingAnts] == \
                        maxPheromones[indFollowingAnts])
            self.historicPath[indFollowingAnts, 
                              self.age[indFollowingAnts]+1,0] -= \
                maxNorth * np.ones(indFollowingAnts.shape[0],dtype=np.int16)
            maxSouth = (southPheromone[indFollowingAnts] == \
                        maxPheromones[indFollowingAnts])
            self.historicPath[indFollowingAnts, 
                              self.age[indFollowingAnts]+1,0] += \
                maxSouth * np.ones(indFollowingAnts.shape[0],dtype=np.int16)

        # On veillit d'une unité l'âge des fourmis non chargées
        if unloadedAnts.shape[0] > 0:
            self.age[unloadedAnts] += 1

        # On fait mourir les fourmis en fin de vie :
        indDyingAnts = np.nonzero(self.age == self.maxLife)[0]
        if indDyingAnts.shape[0] > 0:
            self.age[indDyingAnts] = 0
            self.historicPath[indDyingAnts, 0, 0] = posNest[0]
            self.historicPath[indDyingAnts, 0, 1] = posNest[1]
            self.directions[indDyingAnts] = d.DIR_NONE

        # Pour les fourmis arrivant à la nourriture, on met à jour leurs 
        # états :
        antsAtFoodLoc = np.nonzero(
            np.logical_and(
                self.historicPath[unloadedAnts, 
                                  self.age[unloadedAnts], 0] == posFood[0],
                self.historicPath[unloadedAnts, 
                                  self.age[unloadedAnts], 1] == posFood[1]
                          )       )[0]
        if antsAtFoodLoc.shape[0] > 0:
            antsAtFood = unloadedAnts[antsAtFoodLoc]
            self.isLoaded[antsAtFood] = True

    def advance(self, maze, posFood, posNest, pheromones, foodCounter = 0):
        loadedAnts = np.nonzero(self.isLoaded == True)[0]
        unloadedAnts = np.nonzero(self.isLoaded == False)[0]
        if loadedAnts.shape[0] > 0:
            foodCounter = self.returnToNest(loadedAnts, posNest, foodCounter)
        if unloadedAnts.shape[0] > 0:
            self.explore(unloadedAnts, maze, posFood, posNest, pheromones, 
                         foodCounter)

        oldPosAnts = self.historicPath[range(0,self.seeds.shape[0]),
                                       self.age[:], :]
        hasNorthExit = np.bitwise_and(
            maze[oldPosAnts[:,0],oldPosAnts[:,1]], 
            labyrinthe.NORTH         )>0
        hasEastExit  = np.bitwise_and(
            maze[oldPosAnts[:,0],oldPosAnts[:,1]], 
            labyrinthe.EAST          )>0
        hasSouthExit = np.bitwise_and(
            maze[oldPosAnts[:,0],oldPosAnts[:,1]], 
            labyrinthe.SOUTH         )>0
        hasWestExit  = np.bitwise_and(
            maze[oldPosAnts[:,0],oldPosAnts[:,1]], 
            labyrinthe.WEST          )>0
        # Marquer les phéromones :
        [pheromone.mark(pheromones, self.historicPath[i, self.age[i],:], 
                        [hasNorthExit[i], hasEastExit[i], 
                         hasWestExit[i], hasSouthExit[i]]) \
                            for i in range(self.directions.shape[0])]
        return foodCounter

    def display(self, screen) :
        [screen.blit(self.sprites[self.directions[i]], 
                     (8*self.historicPath[i,self.age[i],1], 
                     8*self.historicPath[i,self.age[i],0])) \
         for i in range(self.directions.shape[0])]

if __name__  == "__main__":
    import time
    pg.init()
    sizeLaby = 25,25
    resolution = sizeLaby[1]*8,sizeLaby[0]*8
    screen = pg.display.set_mode(resolution)
    nbAnts = sizeLaby[0]*sizeLaby[1]//4
    maxLife = 500
    posFood = sizeLaby[0]-1,sizeLaby[1]-1
    posNest = 0,0
    labyrinthe.initCases()
    laby = labyrinthe.buildMaze( sizeLaby, 12345 )
    fourmis = Colony(nbAnts, posNest, maxLife)
    unloadedAnts = np.array(range(nbAnts))
    pherom = pheromone.initPheromone( sizeLaby, posFood )
    pheromone.beta = 0.99
    pheromone.alpha = 0.9
    mazeImg = labyrinthe.displayMaze(laby)
    foodCounter = 0

    snapshopTaken = False
    while True:
        deb = time.time()
        pheromone.displayPheromon( screen, pherom)
        screen.blit(mazeImg, (0, 0))
        fourmis.display(screen)
        pg.display.update()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
        foodCounter = fourmis.advance(laby, posFood, posNest, pherom, 
                                      foodCounter)
        pherom = pheromone.doEvaporation(pherom)
        pherom[posFood[0]+1,posFood[1]+1] = 1.
        end = time.time()
        if foodCounter == 1 and not snapshopTaken:
            pg.image.save(screen, "MyFirstFood.png")
            snapshopTaken = True
        #pg.time.wait(500)
        print(f"FPS : {1./(end-deb):6.2f}, nourriture : {foodCounter:7d}",end='\r')