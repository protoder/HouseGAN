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

P0 = (510, 83)
P1 = (130, 751)
Field = Polygon( [[0,0], [0, 800], [800, 800], [800, 0]] )
Obj0 = Polygon( [[120, 506], [364, 506], [364, 626], [120, 626]])
Obj1 = box( [309, 329, 579, 452])
Obj2 = box( [290, 200, 617, 280])
Obj3 = box( [215, 130, 286, 197])

import math


def angle_from_point(p1, p2):
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


def is_visible(point, vertex, polygon):
    # Проверка видимости вершины из точки
    for i in range(len(polygon)):
        next_i = (i + 1) % len(polygon)
        if intersect(point, vertex, polygon[i], polygon[next_i]):
            return False
    return True


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def find_extreme_angles(polygon, point):
    visible_vertices = []

    for vertex in polygon:
        if is_visible(point, vertex, polygon):
            visible_vertices.append(vertex)

    angles = [(i, angle_from_point(point, vertex)) for i, vertex in enumerate(visible_vertices)]
    angles.sort(key=lambda x: x[1])

    leftmost_index = angles[0][0]
    rightmost_index = angles[-1][0]

    return visible_vertices[leftmost_index], visible_vertices[rightmost_index]


# Пример использования
polygon = [(1, 1), (5, 1), (5, 5), (1, 5)]
point = (0, 0)

leftmost, rightmost = find_extreme_angles(polygon, point)
print(f"Leftmost visible angle: {leftmost}")
print(f"Rightmost visible angle: {rightmost}")


def bypass(P0, P1, p, i):
    for


def CreateWay(p0, p1, Objects):
    Line = LineString([P0, P1])

    res = []

    for i, p in enumerate(Objects):
        if p.intersect(Line):
            NewP = bypass(P0, P1, p, i+1)
            res.append((P0, NewP))
            P0 = NewP

def CreateWay(p0, p1, [Obj0, Obj1, Obj2, Obj3]):