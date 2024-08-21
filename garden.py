import logging
import warnings
import math
import BaseGenetic2 as genetic
import numpy as np
import random
import cv2
from shapely import MultiPolygon, Polygon, LineString
from shapely import polygons
from shapely.geometry import Point, box
from shapely.plotting import plot_polygon
import cv2
from shapely.ops import nearest_points
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from shapely import affinity
from shapely import BufferCapStyle, BufferJoinStyle, buffer
from roads import calculate_distances, dijkstra

# Превращает текстовое представление цвета в RGB
def get_rgb_from_color_name(color_name):
    rgb = mcolors.hex2color(mcolors.cnames[color_name])
    return tuple(int(c * 255) for c in rgb)

image_height, image_width = 800, 800  # Размеры изображения

# объекты - это то, что находится на площадке. До них можно определить растояние
class TObject:
    CrIndex = 1
    def __init__(self, objType, polygon, image_path = False):
        self.RPos = (2,3)
        if objType != roads:
            if len(polygon) == 2:
                if isinstance(polygon[0], float):
                    self.Polygon = box(polygon[0] - 8, polygon[1] - 8, polygon[0] + 8, polygon[1] + 8)
                else:
                    self.Polygon = box(*polygon[0], *polygon[1])


                self._Coords = np.array(self.Polygon.exterior.coords, dtype=np.int32)
            else:
                if polygon[-1][0] != polygon[0][0] and polygon[-1][1] != polygon[0][1] or len(polygon) == 1:
                    polygon.append(polygon[0])
                self.Polygon = Polygon(polygon)
                self._Coords = polygon

                self.S = self.Polygon.area

            if objType == 0:
                self.Index = 0
            else:
                self.Index = TObject.CrIndex
                TObject.CrIndex += 1

        else:
            self._Coords = np.array(polygon)

        self.ObjType = objType # тип объекта.

        self.Owner = None # реальное значение будет при загрузке списка объектов в объект 

        if self.ObjType >= len(TGardenGenetic.ColorsSet):
            self.Color = (0, 0, 0)
        else:
            self.Color = get_rgb_from_color_name(TGardenGenetic.ColorsSet[self.ObjType])

        self.LineWidth = 2
        self.DrawClosed = True
        self.image_path = image_path
        self.DrawBoundsWithImage = True

    def RoadPos(self, hr, step_x, step_y):
        p0, _ = self.CalcPolygon(hr)
        p = buffer(p0, 20 / min(step_x, step_y))


        coords = np.array(p0.exterior.coords, dtype=np.int32)
        r = (coords[self.RPos[0]] + coords[self.RPos[1]])/2

        point = Point(*r)
        # The points are returned in the same order as the input geometries:
        p1, p2 = nearest_points(p, point)

        res = tuple((np.array(p1.coords) / (step_x, step_y)).astype(np.int32)[0])
        return res

    def CalculateDrawPolygon(self, A):
        A0 = np.array(A)
        A2 = np.empty_like(A0)  # Создаем новый массив такого же размера как A0

        # Обменяем элементы в последнем измерении
        A2[:, 0] = A0[:, 1]  # A2[i,0] = A0[i,1]
        A2[:, 1] =self.Owner.PlanCoord.max() - A0[:, 0]

        k = (image_width - 40)/self.Owner.PlanCoord.max()

        return (A2 * k + 20).astype(np.int32)

    def CalcPolygon(self, hr): # вторым значением возвращает дополднительную ошибку, которая может быть тут посчитана
        return Polygon(self. CalcCoords(hr)), 0

    def CalcLineString(self, hr):
        return LineString(self. CalcCoords(hr))

    def CalcCoords(self, hr):
        return self._Coords

    def Coords(self, gene): # вернет координаты объекта - на основании генома или статично
        return _Coords

    def Mistake(self, gene): # по умолчанию речь про расстояние между этим и другими объектами
        return 0

    def Draw(self, image, hr, img = None):
        coord0 = self.CalcCoords(hr)

        if len(coord0) == 0:
            return image

        coord = self.CalculateDrawPolygon(coord0)
        polygon = Polygon(coord)

        '''if not isinstance(self, TBounds):
            p = polygon.intersection(self.Owner.Plan)

            if not p.is_empty:
                polygon = p
                coord = np.array(polygon.exterior.coords, dtype=np.int32)'''

        if self.image_path:
            # получаем координаты размещения объекта
            minx = int(polygon.bounds[0])
            miny = int(polygon.bounds[1])
            maxx = int(polygon.bounds[2])
            maxy = int(polygon.bounds[3])

            # получаем высоту и ширину изображения
            width = maxx - minx
            height = maxy - miny
            dsize = (width, height)

            # считываем изображение
            img_colored = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            img2 = cv2.resize(img_colored, dsize)

            # получаем место размещения изображения на чертеже
            brows, bcols = image.shape[:2]
            rows, cols, channels = img2.shape
            roi = image[miny:maxy, minx:maxx]

            # размещаем изображение на чертеже
            # Создание маски на основе полигона
            mask = np.zeros_like(roi, dtype=np.uint8)

            DrawBounds = self.DrawBoundsWithImage
            try:
                cv2.fillPoly(mask, [(coord - (minx, miny)).astype(np.int32)], (255, 255, 255))

                ind = mask == 255
                roi[ind] = img2[ind]


            except Exception as e:
                DrawBounds = True

            if DrawBounds:
                cv2.polylines(image, [coord], self.DrawClosed, self.Color, self.LineWidth)

            return image

        else:
            cv2.polylines(image, [coord], self.DrawClosed, self.Color, self.LineWidth)

            if  not (self.ObjType in [boundary, tall_trees, medium_trees, undersized_trees]):

                p = polygon.centroid

                center_x, center_y = int(p.x), int(p.y)

                if not isinstance(self, TRoads):
                    text = str(self.Index)

                    # Цвет текста
                    # Шрифт и размер текста
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    font_thickness = 2

                    # Определение размеров текста для позиционирования в середине прямоугольника
                    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                    text_width, text_height = text_size

                    # Рассчитывание позиции для текста
                    text_x = center_x - (text_width // 2)
                    text_y = center_y + (text_height // 2)

                    # Вставка текста в изображение
                    image_with_text = cv2.putText(image, text, (int(text_x),int(text_y)), font, font_scale, self.Color,
                                                  font_thickness)
                    return image_with_text

        return image

class TRoads(TObject):
    def __init__(self, polygon, step_x, step_y, GeoCoords = True): # polygon - м.б. Point
        p = np.array(polygon) * (step_x, step_y)
        super().__init__(roads, TTree.GraphToGeo(list(p)) if not GeoCoords else list(p))
        #self.DrawClosed = True
        self.image_path = 'roads.jpg'

    def CalcCoords(self, hr):
        coords = super().CalcCoords(hr)
        p = buffer(LineString(coords), 1)
        return np.array(p.exterior.coords)



# представляет фиксированные объекты, которые надо обойти
class TTree(TObject):
    # Из геодезичеких координат в координаты графика
    # 678 + (495 - y) * (3604 - 678) / 495 = > 678 + (3604 - 678) - y* (3604 - 678)/495 -> 3604 - y* (3604 - 678)/495
    #

    Kx = (917 - 496)/71
    Ax = 496
    Ky = (678 - 3604)/495
    Ay = 3604

    A = np.array([Ax, Ay])
    K = np.array([Kx, Ky])

    @staticmethod
    def GraphToGeo(GraphCoords):
        x, y =((GraphCoords - TTree.A)/TTree.K)
        return y, x

    @staticmethod
    def GeoToGraph(GeoCoords):
        return GeoCoords * TTree.K + TTree.A


    def __init__(self, objType, polygon, GeoCoords = False, image_path = 'tree.jpg'): # polygon - м.б. Point
        super().__init__(objType, TTree.GraphToGeo(polygon) if not GeoCoords else polygon, image_path = image_path)
        self.DrawBoundsWithImage = False



#представляет границы объекта
class TBounds(TTree):
            def __init__(self, polygon, image_path = False):
                super().__init__(0, polygon, True, image_path = image_path)



# строения. У каждого из них могут быть правила размещения относительно друг друга.
class TBuilding(TObject):
    def calculate_intersection_area(self, plg1, plg2):

        if not (isinstance(plg1, Polygon) or isinstance(plg1, LineString)) :
            rect1 = np.array(plg1, dtype = np.int32)
            if len(plg1) > 2:
                plg1 = Polygon(rect1)
            else:
                rect1[2:] += rect1[:2]
                plg1 = box(*rect1)

        if not (isinstance(plg2, Polygon) or isinstance(plg2, LineString)) :
            rect2 = np.array(plg2, dtype = np.int32)
            if len(plg2) > 2:
                 plg2 = Polygon(rect2)
            else:
                 rect2[2:] += rect2[:2]

                 plg2 = box(*rect2)

        # Извлечение параметров прямоугольников
        if plg1.area * plg2.area == 0:
            a  = 0

        intersection_area = plg1.intersection(plg2).area

        # Вычисление процента площади пересечения относительно каждого прямоугольника
        percentage_area1 = (intersection_area / (plg1.area+0.000001))
        percentage_area2 = (intersection_area / (plg2.area+0.000001))

        return intersection_area, percentage_area1, percentage_area2

    def Mistake(self, hr): # по умолчанию речь про расстояние между этим и другими объектами
        polygon, err2 = self.CalcPolygon(hr)

        intersection_area, percentage_area1, percentage_area2 = self.calculate_intersection_area(polygon, self.Owner.PlanCoord)
        err = 3*(1 - percentage_area1)

        #isFirst = True
        tp = self.ObjType

        if tp == house:
            rl = polygon.distance(self.Owner.RedLine)

        for obj in self.Owner.Objects:

            if obj == self:
                continue

            objTp = obj.ObjType
            dist = Distances[tp][objTp]

            if dist < 0.01:
                continue

            ls = obj.CalcLineString(hr)
            x = polygon.distance(ls)
            if np.isnan(x):
                err+= 20
            else:
                err += 1/( (x/dist)**32 + 0.001)/1000# функция стремится к 0 при растоянии больше item[1] (limit), и резко возрастает к 1

            if objTp != 0 and not np.isnan(x):
                p, _ = obj.CalcPolygon(hr)
                intersection_area, percentage_area1, percentage_area2 = self.calculate_intersection_area(polygon, p)

                err += 3 * (percentage_area2 + percentage_area1)

            if tp == house:
                if objTp in [cattle_barn, outbuildings, toilet, septik_tank, pit]:
                    d = ls.distance(self.Owner.RedLine)

                    if d < rl:
                        err += rl - d


        return err                                 # если меньше

    def __init__(self, objType, polygon, image_path = False):
        super().__init__(objType, polygon, image_path = image_path) # limits = словарь расстояний до других объектов

    def CalcCoords(self, hr):
        if hr is None:
            return self._Coords

        ind = (self.Index - 1) * self.Owner.ObjectDataSz
        gene = hr[ind:ind + self.Owner.ObjectDataSz].astype(np.int32)
        gene = (gene[0:2]*self.Owner.K)

        coords = self._Coords + gene
        return coords

'''
    Дорога. Представоляет собой объект сложной формы (вместе с въездной площадочкой)
'''
class TRoad(TBuilding):
    def Mistake(self, hr): # по умолчанию речь про расстояние между этим и другими объектами
        '''if not isinstance(polygon, Polygon):
            return 200
        else:
            intersection_area, percentage_area1, percentage_area2 = self.calculate_intersection_area(polygon, self.Owner.PlanCoord)
            err = 3*(1 - percentage_area1)'''

        onlyGarden, _ = self.CalcPolygon(hr, woRoad = True)

        if not isinstance(onlyGarden, Polygon):
            return 200

        # Автоплощадка должна быть в границах поля
        intersection_area, percentage_area1, percentage_area2 = self.calculate_intersection_area(onlyGarden, self.Owner.PlanCoord)
        err = 3 * (1 - percentage_area1)


        polygon, _ = self.CalcPolygon(hr)

        if not isinstance(polygon, Polygon):
            return 200 + err

        # Проверяем на непересекаемость с остальными объектами всей дороги
        tp = self.ObjType
        for obj in self.Owner.Objects:

            if obj == self:
                continue

            objTp = obj.ObjType
            if objTp != 0:
                p, addEr = obj.CalcPolygon(hr)
                intersection_area, percentage_area1, percentage_area2 = self.calculate_intersection_area(polygon, p)

                err += 3 * (percentage_area2 + percentage_area1) + addEr



        return err                                 # если меньше

    # рисуем площадку и дорогу от заданной точки
    def CalcBasePolygon(self, point, roadpos, woRoad = False):
        p = []
        p.append(point)
        p.append( (point[0] - self.Sz[1], point[1]))

        if not woRoad: # рисуем дорогу - просто кидаем ее вниз. Потом обрежем
            p.append( (p[1][0], p[1][1] + roadpos) )
            p.append( (-1000, p[2][1]) )
            p.append( (-1000, p[2][1] + self.Roadwidth) )
            p.append( (p[1][0], p[4][1]) )

        p.append( (p[1][0], point[1] + self.Sz[0]) )

        p.append( (point[0], p[-1][1]) )

        return Polygon(p)

    def CalcPolygon(self, hr, woRoad = False):
        if hr is None:
            gene = (0, 0, 0, 0)
        else:
            ind = (self.Index - 1) * self.Owner.ObjectDataSz
            gene = hr[ind:ind + 4].astype(np.int32)
            gene[0:2] = gene[0:2] * self.Owner.K
            gene[2:] = gene[2:] * self.K2

        p = self.CalcBasePolygon(gene[:2], (gene[3]), woRoad )

        rotated_polygon = affinity.rotate(p, gene[2], origin='center')
        rotated_polygon = p
        if woRoad:
            return rotated_polygon, 0

        res_polygon = rotated_polygon.intersection(self.Owner.Plan)

        dif = rotated_polygon.difference(res_polygon)

        if isinstance(dif, Polygon):
            d = 0 if len(dif.exterior.coords.xy[0]) == 5 and -1000 in dif.exterior.coords.xy[0] else 2
        else:
            d = 2
        return res_polygon, d

    def CalcCoords(self, hr):
        plg, _ = self.CalcPolygon(hr)

        if not (plg.is_empty) and isinstance(plg, Polygon):
            return np.array(plg.exterior.coords)
        else:
            return []


    # sz (x,y)
    def __init__(self, sz = (80, 50), DAngle = 10, roadwidth = 40, image_path = 'autoroads.jpg'):

        self.Sz = sz
        self.Roadwidth = roadwidth

        self.ObjType = road
        self.S = 0

        self.Owner = None

        self.Index = TObject.CrIndex
        TObject.CrIndex += 2

        self.Color = get_rgb_from_color_name(TGardenGenetic.ColorsSet[self.ObjType])
        self.LineWidth = 2 # limits = словарь расстояний до других объектов

        self.K2 = (10/128, (sz[0] - roadwidth)/255)

        self.RPos = (0, 0)

        self.DrawClosed = True

        self.image_path = image_path
        self.DrawBoundsWithImage = True



class TGardenGenetic(genetic.TBaseGenetic):
        ObjectDataSz = 2
        ColorsSet = {0: 'black',
                    1: 'orange',
                    2: 'blue',
                    3: 'yellow',
                    4: 'purple',
                    5: 'green',
                    6: 'green',
                    7: 'green',
                    8: 'magenta',
                    9: 'brown',
                    10: 'teal',
                    11: 'lavender',
                    12: 'lime',
                    13: 'maroon',
                    14: 'indigo',
                    15: 'red',
                    16: 'pink',
                    17: 'black',
                    18: 'black'
                    }
        Colors = list(ColorsSet)

        def __init__(self, objects, plan, theBestListSize=1000,
                     populationSize=4000):  # rooms - список номеров комнат и их предполагаемых площадей в формате двумерного списка

            planObj = TBounds(plan)

            bnd = planObj.Polygon.bounds
            self.PlanH = bnd[2] - bnd[0]
            self.PlanW = bnd[3] - bnd[1]

            self.KX = self.PlanW / 255
            self.KY = self.PlanH / 255

            self.K = [self.KY, self.KX]


            self.Objects = [planObj]
            self.Objects.extend(objects)

            hrSize = 1

            for i in range(len(self.Objects)):
                self.Objects[i].Owner = self

                if isinstance(self.Objects[i], TBuilding):
                    hrSize+= 1

                    if isinstance(self.Objects[i], TRoad):
                        self.Road = self.Objects[i]

                        hrSize += 1


            self.PlanCoord = plan

            self.Plan = Polygon(plan)

            super().__init__(hrSize * self.ObjectDataSz, GenGroupSize=self.ObjectDataSz,
                             TheBestListSize=theBestListSize, PopulationSize=populationSize)

            self.Cells = (200, 200)

        def CreateCells(self, xSz, ySz, hr):
            plan = self.Plan
            objects = self.Objects
            # Создаем пустой массив Res
            res = np.zeros((xSz, ySz), dtype=int)

            # Определяем границы прямоугольника, описывающего Plan
            min_x, min_y, max_x, max_y = plan.bounds

            # Разбиваем прямоугольник на 10 частей по вертикали и горизонтали
            step_x = (max_x - min_x) / xSz
            step_y = (max_y - min_y) / ySz

            # Заполняем массив Res
            for i in range(xSz):
                for j in range(ySz):
                    # Вычисляем координаты текущей ячейки
                    cell_min_x = min_x + i * step_x
                    cell_max_x = min_x + (i + 1) * step_x
                    cell_min_y = min_y + j * step_y
                    cell_max_y = min_y + (j + 1) * step_y

                    # Создаем прямоугольник для текущей ячейки
                    cell_polygon = Polygon([(cell_min_x, cell_min_y),
                                            (cell_max_x, cell_min_y),
                                            (cell_max_x, cell_max_y),
                                            (cell_min_x, cell_max_y)])

                    # Проверяем условия и присваиваем значения в массиве Res
                    if not plan.contains(cell_polygon):
                        res[i, j] = 2
                    else:
                        for obj_polygon in objects[1:]:

                            p, _ = obj_polygon.CalcPolygon(hr)

                            p = buffer(p, 20/min(step_x, step_y))

                            if cell_polygon.intersects(p):
                                res[i, j] = 1
                                break

            for obj_polygon in objects[1:]:
                r = (np.array(obj_polygon.RoadPos(hr, step_x, step_y))/(step_x, step_y)).astype(np.int32)


                res[*r] = 0


            return res, step_x, step_y

        @staticmethod
        def calculate_distance(x1, y1, x2, y2):
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            return distance

        def GetNearest(self, obj, excepts, hr, step_x, step_y):
            excepts.add(obj)

            objPos = obj.RoadPos(hr, step_x, step_y)
            dist = float('infinity')
            nearest = None

            for testedObj in self.Objects[1:]:
                if isinstance(testedObj, TBuilding) and not testedObj in excepts:
                    testPos = testedObj.RoadPos(hr, step_x, step_y)

                    dst = TGardenGenetic.calculate_distance(*objPos, *testPos)

                    if dst < dist:
                        dist = dst
                        nearest = testedObj

            return nearest



        def CalcRoads(self, hr):
            cells, step_x, step_y = self.CreateCells(*self.Cells, hr)
            graph = calculate_distances(cells)
            obj = self.Road

            startPos = obj.RoadPos(hr, step_x, step_y)

            res = []

            ready = set()

            while True:
                obj = self.GetNearest(obj, ready, hr, step_x, step_y)

                if obj is None:
                    break
                #if isinstance(obj, TBuilding) and not obj in ready:

                crPos = obj.RoadPos(hr, step_x, step_y)
                r = dijkstra(graph, startPos, crPos)

                res.extend(r)

                startPos = crPos

            return TRoads(res, step_x, step_y)

        def TestHromosom(self, hr):
            #genes = len(hr)

            err = 0

            hrIndex = 0
            for obj in self.Objects:

                err += obj.Mistake(hr)

            return err

        def CalculatePolygon(self, A):
            A0 = np.array(A)
            A2 = np.empty_like(A0)  # Создаем новый массив такого же размера как A0

            # Обменяем элементы в последнем измерении
            A2[:, 0] = A0[:, 1]  # A2[i,0] = A0[i,1]
            A2[:, 1] = self.PlanCoord.max() - A0[:, 0]

            return (A2 * 5 + 20).astype(np.int32)

        def Draw(self, image, hr=None, drawRoads = False):

            for obj in self.Objects:
                image = obj.Draw(image, hr)

            if drawRoads and hr is not None:
                cv2.imwrite('c:/Res/test_.jpg', image)
                roads = self.CalcRoads(hr)
                roads.Owner = self
                roads.Draw(image, hr)

            return image

        def Visualisation(self, hr, ind, metric):
            list = np.argsort(self.BaseHromosomRatingValues)

            #image_height, image_width = 800, 800  # Размеры изображения
            image = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
            image = self.Draw(image, self.UsefullBestHr, metric < 0.1)


            cv2.imwrite('c:/Res/ga_best_' + str(self.UsefullMetric) + '.jpg', image)

            f = False

            if f:
                print('Проверка best ', str(self.UsefullMetric) + '_' + str(self.UsefullBestHr))
                print(self.TestHromosom(self.UsefullBestHr))

            for i, h in enumerate(self.Hromosoms[list[:self.BestForShowingCnt]]):
                #h = np.full_like(h, 255)
                #image_height, image_width = 800, 800  # Размеры изображения
                image = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
                image = self.Draw(image, h, metric < 1)
                v = self.BaseHromosomRatingValues[list[i]]

                cv2.imwrite('c:/Res/ga_' + str(ind) + '_' + str(i) + '_' + str(v) + '.jpg', image)

                #if v < 0.2:
                #    print('Проверка ', str(ind) + '_' + str(i) + '_' + str(v) + '_' + str(h))
                #    print(self.TestHromosom(h))

individual_housing_construction_objects = ['Граница участка',
                                           'Дом',  # сюда включены усадебные, одно-двухквартирные дома
                                           'Гараж',
                                           'Граница соседнего участка',
                                           'Постройка содержания скота и птицы',
                                           'Хозпостройки',
                                           'Высокорослые деревья (высота выше 20 м)',  # деревья
                                           'Среднерослые деревья (высота от 10 до 20 м)',  # деревья
                                           'Низкорослые деревья (высота до 10 м)',  # деревья
                                           'Окна жилых помещений',
                                           'Стены дома соседнего участка',
                                           'Хозпостройки соседнего участка',
                                           'Надворный туалет',
                                           'Септик',
                                           'Скважина/колодец',
                                           'Красная линия',
                                           'Красная линия 2',
                                           'Дорога',
                                           'Автоплощадка']

print(len(individual_housing_construction_objects))

Distances = [[10 for i in range(len(individual_housing_construction_objects))] for j in
                                range(len(individual_housing_construction_objects))]

boundary = 0
house = 1
garage = 2
neighbouring_plot = 3
cattle_barn = 4
outbuildings = 5
tall_trees = 6
medium_trees = 7
undersized_trees = 8
windows = 9
neighbouring_house = 10
neighbouring_outbuildings = 11
toilet = 12
septik_tank = 13
pit = 14
red_line = 15
red_line2 = 16
road = 17
avtoplace = 18
roads = 19


# Граница участка
Distances[boundary][garage] = 70
Distances[boundary][house] = 70
Distances[boundary][outbuildings] = 70
Distances[boundary][septik_tank] = 70
Distances[boundary][toilet] = 70
Distances[boundary][pit] = 70
Distances[house][red_line] = 50
Distances[house][red_line2] = 30
Distances[garage][red_line] = 30
Distances[house][neighbouring_plot] = 30
Distances[cattle_barn][neighbouring_plot] = 40
Distances[garage][neighbouring_plot] = 10
Distances[outbuildings][neighbouring_plot] = 10
Distances[medium_trees][neighbouring_plot] = 20
Distances[house][neighbouring_house] = 60
Distances[neighbouring_outbuildings][house] = 60
Distances[house][toilet] = 120
Distances[house][septik_tank] = 40
Distances[house][pit] = 250
Distances[septik_tank][pit] = 250

Distances[medium_trees][garage] = 20
Distances[medium_trees][house] = 20
Distances[cattle_barn][medium_trees] = 20
Distances[outbuildings][medium_trees] = 20
Distances[medium_trees][toilet] = 20
Distances[medium_trees][pit] = 20
Distances[medium_trees][septik_tank] = 20


for x in range(len(Distances)):
    for y in range(len(Distances)):
       if Distances[x][y] > 0:
           Distances[y][x] = Distances[x][y]
       elif Distances[y][x] > 0:
           Distances[x][y] = Distances[y][x]

Plan = np.array([[0, 81], [0, 71], [9, 69], [495, 0], [554, 1], [560, 58], [572, 164], [572, 164], [577, 201], [593, 324],
                [594, 331], [589, 332], [11, 380], [2, 217], [2, 217], [3, 111], [0, 81]])

Gor = np.array([[11, 380], [2, 217], [2, 217], [3, 111], [0, 81], [0, 71], [9, 69]])


warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

#Plan = np.array([[210, 0], [210, 259], [94, 259], [94, 100], [0, 100], [0,  0], [210, 0]], dtype = np.int32)

Obj0 = TBuilding(house, [[0,0], [100, 120]], image_path = 'house2.jpg') # [137, 169]])
#Obj1 = TBuilding(garage, [[0,0], [59, 59]])
Obj2 = TBuilding(outbuildings, [[0,0], [30, 60]], image_path = 'bathhouse.jpg') # [60, 90]
Obj3 = TBuilding(septik_tank, [[0,0], [25, 30]], image_path = 'septic.jpg')
Obj4 = TBuilding(pit, [[0,0], [30, 30]], image_path = 'pit.jpg')
Obj3.DrawBoundsWithImage = Obj4.DrawBoundsWithImage = False
Obj5 = TRoad()

'''
1. Дом
2. Гараж
3. Хозпостройка
4. Септик
5. Туалет
6. Дорога и автоплощадка
'''

trees = [TTree(medium_trees, (2033, 1009)),
        TTree(medium_trees, (2572, 2490)),
        TTree(medium_trees, (2345, 2486)),
        TTree(medium_trees, (2092, 2747)),
        TTree(medium_trees, (1493, 2758)),
        TTree(medium_trees, (1815, 3136)),
        TTree(medium_trees, (1489, 2755)),
        TTree(medium_trees, (944,2549)),
        TTree(medium_trees, (1099,2913)),
        TTree(medium_trees, (1819,3138)),
        TTree(medium_trees, (1308,3453)),
        TTree(medium_trees, (960,3587)),
        TTree(medium_trees, (1476,3527)),
        TTree(medium_trees, (1580,3510)),
        TTree(medium_trees, (1682,3500)),
        TTree(medium_trees, (1807.3478)),
        TTree(medium_trees, (1969,3475)),
        TTree(medium_trees, (1973,3473)),
        TTree(medium_trees, (2061,3447)),
        TTree(medium_trees, (2122, 3196))]



Objects = [Obj0, Obj2, Obj3, Obj4, Obj5]

Objects.extend(trees)

n = 6
random.seed(n)
np.random.seed(n)
# Создаем чистое изображение
#image_height, image_width = 800, 800  # Размеры изображения
image = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
Gn = TGardenGenetic(Objects, Plan, theBestListSize=50, populationSize=200)
Gn.Draw(image)
cv2.imwrite(f'c:/Res/Mask.jpg', image)
#Gn = TRoomGenetic(rooms, coord)
Gn.PDeath = 0.75
Gn.StopFlag = 0
Gn.kNovelty = 1
Gn.isStranger = 1.8
Gn.PMutation = 0.05  # probability of the mutation for any individuals. Or, if >=1, the number of individuals that will die in each generation
Gn.PMultiMutation = 0.1  # the probability of an any bit to be mutated, in the case of the mutation has occurred
# self.PDeath = 0.2 # probability of the death. Or, if >=1, the number of individuals that will die in each generation
Gn.PCrossingover = 0.25
Gn.PairingAttempts = 8  # попытки скрещивания. Скрещивание неудачно, если в результате рождается хромосома, которая не проходит проверки.
Gn.MinPMutations = 0.05
Gn.MaxPMutations = 0.75
Gn.RecreatePopulation = False
Gn.StrongMutations = False
Gn.VisualizationStep = 1
Gn.DebugInfo = True
Gn.Weights = [1, 1, 1, 1, 1]
Gn.DWeights = [0.7, 0.7, 0.7, 0.7, 0.7]
Gn.Debug = False
Gn.useNovelty = True
Gn.killDubles = True
Gn.BestForShowingCnt = 10
Gn.RedLine = LineString( ( (0,0), (0, 10000)) )
Gn.Start()

import numpy as np

# Заданный отрезок AB
A = np.array([-10, -10])
B = np.array([3, 3])

# Заданная точка M
M = np.array([-5, -5])

# Векторы AB и AM
AB = B - A
AM = M - A

# Скалярное произведение векторов
dot_product_AB_AM = np.dot(AB, AM)
dot_product_AB_AB = np.dot(AB, AB)

# Проверка условия
if 0 <= dot_product_AB_AM <= dot_product_AB_AB:
    print("Точка M лежит на отрезке AB.")
else:
    print("Точка M не лежит на отрезке AB.")
