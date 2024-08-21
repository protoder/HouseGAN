import logging
import warnings
import math
import BaseGenetic2 as genetic
import numpy as np
import random
import cv2
from shapely import MultiPolygon, Polygon, LineString, MultiPolygon, Point
from shapely import polygons, prepare
from shapely.geometry import Point, box
from shapely.plotting import plot_polygon
import cv2
from shapely.ops import nearest_points
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from shapely import affinity
from shapely import BufferCapStyle, BufferJoinStyle, buffer, difference
from roads import calculate_distances, dijkstra
import Geometry
import TestHouses

def CalcSz(line):
    return LineString(line).length

def bypass(P0, P1, i, p):
'''

'''
    leftTop = [] # список видимых из P1 точек полигона, при взгляде влево
    rightTop = [] # список видимых из P1 точек полигона, при взгляде вправо
    leftBottom = [] # список видимых из P0 точек полигона, при взгляде влево
    rightBottom = [] # список видимых из P0 точек полигона, при взгляде влево

    for i, coords in enumerate(pCoords):
        str0 = LineString((P0, coords))
        if str0.intersection(p).area == 0:
            if coords[1] < P0[1]: # контроль, слева точка или справа
                leftBottom.append(i)
            else: # точка справа
                rightBottom.append(i)

        str1 = LineString((P1, coords))
        if str1.intersection(p).area == 0:
            if coords[1] < P1[1]:  # контроль, слева точка или справа
                leftTop.append(i)
            else:  # точка справа
                rightTop.append(i)

    # получили список индексов видимых точек

    # теперь находим самые короткие сектора полигона от нижних видимых до верхних видимых
    right = pCoords[max(rightTop) : min(rightBottom)+1]
    rightSz = CalcSz(right)
    left = pCoords[max(leftBottom) : min(leftTop)+1]

    rightWay, rightSz1 = CalcWay(rightTop, P1, i)
    leftWay, leftSz1 = CalcWay(LeftTop, P1, i)

    if right + rightSz1 <= left + leftSz1:
        right.extend(rightWay)

        return right, right + rightSz1
    else:
        left.extend(leftWay)
        return left, left + leftSz1






