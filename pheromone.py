"""
"""
import numpy as np
import direction as d
import pygame as pg

alpha = 0.7
beta  = 0.9999

def initPheromone( tDimensions, tFoodPos ):
    pheromon = np.zeros((tDimensions[0]+2,tDimensions[1]+2), dtype=np.double)
    pheromon[tFoodPos[0]+1,tFoodPos[1]+1] = 1.
    return pheromon

def doEvaporation( tPheromon ):
    return beta * tPheromon

def mark( tPheromon, tPosition, hasWESNExists ):
    #print(f"tPosition : {tPosition}")
    assert(tPosition[0] >= 0)
    assert(tPosition[1] >= 0)
    cells = np.array([ tPheromon[tPosition[0]+1,tPosition[1]] if hasWESNExists[d.DIR_WEST ]   else 0.,
                       tPheromon[tPosition[0]+1,tPosition[1]+2] if hasWESNExists[d.DIR_EAST ] else 0.,
                       tPheromon[tPosition[0]+2,tPosition[1]+1] if hasWESNExists[d.DIR_SOUTH] else 0.,
                       tPheromon[tPosition[0],tPosition[1]+1] if hasWESNExists[d.DIR_NORTH]   else 0. ], dtype=np.double )
    pheromones = np.maximum(cells, 0.)
    tPheromon[tPosition[0]+1,tPosition[1]+1] = alpha * np.max(pheromones) + (1-alpha)*0.25*(pheromones.sum())

def getColor( value ):
    val = max(min(value,1),0)
    return [255*(val>1.E-14), 255*val, 128.]

def displayPheromon( screen, tPheromon ) :
    [ [ screen.fill(getColor(tPheromon[i,j]), (8*(j-1),8*(i-1),8,8)) for j in range(1,tPheromon.shape[1]-1)] for i in range(1,tPheromon.shape[0]-1)]