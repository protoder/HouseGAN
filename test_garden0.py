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

# Превращает текстовое представление цвета в RGB
def get_rgb_from_color_name(color_name):
    rgb = mcolors.hex2color(mcolors.cnames[color_name])
    return tuple(int(c * 255) for c in rgb)

image_height, image_width = 800, 800  # Размеры изображения

def TestDraw(lst, path = 'c:/Demo/test.jpg'):
    maxY = maxX = 0

    for i, obj in enumerate(lst):
        if isinstance(obj, Polygon):
            obj = np.array(obj.exterior.coords, dtype=np.int32)
            lst[i] = obj

        if obj[:,0].max() > maxY:
            maxY = obj[:,0].max()

        if obj[:,1].max() > maxX:
            maxX = obj[:,1].max()

    image = np.ones((maxX, maxY, 3), dtype=np.uint8)

    for obj in lst:
        cv2.polylines(image, lst, isClosed=True, color=(255, 255, 255), thickness=2)

    cv2.imwrite(path, image)

def relu(x):
    return max(0.0, x)

# объекты - это то, что находится на площадке. До них можно определить растояние
class TObject:
    CrIndex = 1

    @staticmethod
    # вернет ошибку от 0 до mx в зависимости от того, на сколько нарушена дистанция
    def Dist(x, y, mx):
        return mx*relu(y - x)/y

    def InitImage(self, image_path, angle = 0):
        self.image_path = image_path

        if self.image_path:
            # получаем координаты размещения объекта
            minx = int(self.Polygon.bounds[0])
            miny = int(self.Polygon.bounds[1])
            maxx = int(self.Polygon.bounds[2])
            maxy = int(self.Polygon.bounds[3])

            # получаем высоту и ширину изображения
            self.width = maxx - minx
            self.height = maxy - miny
            dsize = (self.width, self.height)

            '''img_colored = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            image = self.CalcImg( hr, img_colored)
            img2 = self.CalculateDrawImg(image, dsize)'''

            # считываем изображение. [h,w,3]
            img_colored = cv2.imread(self.image_path, cv2.IMREAD_COLOR)

            img_colored = Geometry.rotateImg(img_colored, angle)

            self.image = cv2.resize(img_colored, dsize)


            self.image = cv2.polylines(self.image, [self._Coords.astype(np.int32)], self.DrawClosed, self.Color, self.LineWidth)


            #cv2.imwrite('c:/Demo/mask.jpg', self.image)



    def __init__(self, objType, polygon, image_path = False, image_angle = 0):
        self.RPos = (2,3)
        self.AddError = 0
        if objType != roads:
            if len(polygon) == 2:
                if isinstance(polygon[0], float):
                    self.Polygon = box(polygon[0] - 8, polygon[1] - 8, polygon[0] + 8, polygon[1] + 8)
                else:
                    self.Polygon = box(*polygon[0], *polygon[1])

                self._Coords = np.array(self.Polygon.exterior.coords, dtype=np.int32)
            else:
                if polygon[-1][0] != polygon[0][0] and polygon[-1][1] != polygon[0][1] or len(polygon) == 1:
                    polygon = np.append(polygon, [polygon[0]], axis = 0)

                self.Polygon = Polygon(polygon)
                prepare(self.Polygon)
                self._Coords = polygon
                self.Center = self.Polygon.centroid
                prepare(self.Center)
                self.S = self.Polygon.area

            if objType == 0:
                self.Index = 0
            else:
                self.Index = TObject.CrIndex
                TObject.CrIndex += 1

        else:
            self._Coords = np.array(polygon)

        left = self._Coords[0]



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

        self.InitImage(image_path, image_angle)

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

    def CalculateDrawPolygonSize(self, A):
        A0 = np.array(A)
        # A2 = np.empty_like(A0)  # Создаем новый массив такого же размера как A0

        # Обменяем элементы в последнем измерении
        # A2[:, 0] = A0[:, 1]  # A2[i,0] = A0[i,1]
        # A2[:, 1] =self.Owner.PlanCoord.max() - A0[:, 0]
        # A2 = A0

        k = (image_width - 40) / self.Owner.PlanCoord.max()

        return (A0 * k).astype(np.int32)

    def CalculateDrawPolygon(self, A):
        return self.CalculateDrawPolygonSize(A) + 20

    def CalculateDrawImg(self, A0, dsize):
        A2 = np.empty_like(A0)  # Создаем новый массив такого же размера как A0

        # Обменяем элементы в последнем измерении
        A2[:, 0] = A0[:, 1]  # A2[i,0] = A0[i,1]
        A2[:, 1] = self.Owner.PlanCoord.max() - A0[:, 0]

        k = (image_width - 40) / self.Owner.PlanCoord.max()

        img2 = cv2.resize(A2, dsize * k)
        return img2

    def CalcPolygon(self, hr): # вторым значением возвращает дополднительную ошибку, которая может быть тут посчитана
        return Polygon(self. CalcCoords(hr, self._Coords)), 0

    def CalcLineString(self, hr):
        return LineString(self. CalcCoords(hr, self._Coords))

    def CalcCoords(self, hr, coords):
        return coords

    def TransformImgCoords(self, image):
        return image

    def Coords(self, gene): # вернет координаты объекта - на основании генома или статично
        return _Coords

    def Mistake(self, gene): # по умолчанию речь про расстояние между этим и другими объектами
        return 0

    def Draw(self, image, hr, img = None):
        try:
            coord0 = self.CalcCoords(hr, self._Coords)
        except:
            coord0 = self.CalcCoords(hr, None)

        if len(coord0) == 0:
            return image

        if not isinstance(self, TRoad):
            coord = self.CalculateDrawPolygon(coord0)
        else:
            coord = coord0.astype(np.int32)

        polygon = Polygon(coord)

        if isinstance(self, THouse):
            if self.Owner.Road:
                if self.Garage is not None:
                    self.Owner.Road.Garage = coord[self.Garage]
                    self.Owner.Road.House = polygon
                else:
                    self.Garage = None


        '''if not isinstance(self, TBounds):
            p = polygon.intersection(self.Owner.Plan)

            if not p.is_empty:
                polygon = p
                coord = np.array(polygon.exterior.coords, dtype=np.int32)'''

        if self.image_path:

            img2 = self.image
            #coords = np.float32([ (0,0), (0, width), (height,width), (height, 0)])
            #bnd = self.Polygon.bounds
            #coords = np.float32( ((0,0), (0,bnd[3]), (bnd[2], bnd[3]), (bnd[2], 0)))

            if isinstance(self, TRoad):
                #coords= self.CalcCoords(None)
                #coords = self.CalculateDrawPolygon(coords1)
                #coords = np.array(p.exterior.coords)

                #bounds = np.int32(p.bounds)
                #roi = image[bounds[0]:bounds[2], bounds[1]:bounds[3]]
                mask = np.zeros_like(image, dtype=np.uint8)
                DrawBounds = self.DrawBoundsWithImage
                try:
                    cv2.fillPoly(mask, [coord], (255, 255, 255))

                    ind = mask == 255
                    image[ind] = img2[ind]
                except Exception as e:
                    DrawBounds = True
            else:
                coords = np.float32(self.Polygon.bounds)
                sh = self.image.shape
                coords3 = np.float32( ((0,0), (0,sh[1]), (sh[0], sh[1]), (sh[0], 0)))
                coords = np.float32(( (0.0,0.0), (0.0, coords[3]), (coords[2], coords[3]), (coords[2], 0.0)))
                coords1 = self.CalcCoords(hr, coords)
                coords2 = self.CalculateDrawPolygon(coords1)

                ds = coords2.max(0).astype(np.int32)
                #cv2.resize(src, dsize)

                coords2 = coords2.astype(np.float32)
                matrix = cv2.getAffineTransform(coords[:3], coords2[:3])
                #matrix = cv2.getPerspectiveTransform(coords, coords2.astype(np.float32))
                # Применяем преобразование

                im = cv2.warpAffine(img2, matrix, (ds[0],ds[1])) #(img2.shape[0],img2.shape[1]))
                #cv2.imwrite('c:/Demo/mask.jpg', im)
                #cv2.imwrite('c:/Demo/mask1.jpg', img2)

                # получаем координаты размещения объекта
                minx = int(polygon.bounds[0])
                miny = int(polygon.bounds[1])
                maxx = min(int(polygon.bounds[2]), ds[0] )
                maxy = min(int(polygon.bounds[3]), ds[1] )

                # получаем высоту и ширину изображения
                width = maxx - minx
                height = maxy - miny
                dsize = (width, height)
                # получаем место размещения изображения на чертеже
                #brows, bcols = image.shape[:2]
                img2 = im[miny:maxy, minx:maxx]
                #rows, cols, channels = img2.shape
                roi = image[miny:maxy, minx:maxx]
                #cv2.imwrite('c:/Demo/mask3.jpg', img2)

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

                if isinstance(self, THouse) and self.Garage is not None:
                    cv2.polylines(image, [coord[self.Garage]], self.DrawClosed, (0,0,255), 4)
            return image

        else:
            if self.ObjType in [septik_tank, pit]:
                radius = 3  # abs(coord[0,0]-coord[1,0])/2
                if radius < 1:
                    radius = 1
                cv2.circle(image, ((coord[0] + coord[2]) / 2).astype(np.int32), int(radius), self.Color, self.LineWidth)
            else:
                cv2.polylines(image, [coord], self.DrawClosed, self.Color, self.LineWidth)

            if  not (self.ObjType in [boundary, tall_trees, medium_trees, undersized_trees]):

                p = polygon.centroid

                center_x, center_y = int(p.x), int(p.y)

                if not isinstance(self, TRoads):
                    if self.ObjType in [septik_tank, pit]:
                        text = 'C' if self.ObjType == septik_tank else 'K'
                    else:
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
                    if self.ObjType in [septik_tank, pit]:
                        text_y -= text_height

                    # Вставка текста в изображение
                    image_with_text = cv2.putText(image, text, (int(text_x), int(text_y)), font, font_scale, self.Color,
                                                  font_thickness)
                    return image_with_text
        self.ExtDraw(image, coord)
        return image

    def ExtDraw(self, image, coord):
        return image

class TRoads(TObject):
    def __init__(self, polygon, step_x, step_y, GeoCoords = True): # polygon - м.б. Point
        p = np.array(polygon) * (step_x, step_y)
        super().__init__(roads, TTree.GraphToGeo(list(p)) if not GeoCoords else list(p))
        #self.DrawClosed = True
        self.image_path = 'roads.jpg'

    def CalcCoords(self, hr, coords):
        coords = super().CalcCoords(hr, coords)
        try:
            p = buffer(LineString(coords), 1)
        except:
            return []

        return np.array(p.exterior.coords)

    def Draw(self, image, hr, img=None):
        coord0 = self.CalcCoords(hr, self._Coords)

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

            if not (self.ObjType in [boundary, tall_trees, medium_trees, undersized_trees]):

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
                    image_with_text = cv2.putText(image, text, (int(text_x), int(text_y)), font, font_scale, self.Color,
                                                  font_thickness)
                    return image_with_text

        return image


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

                self.Lines = []
                self.Centers = []

                left = self._Coords[0]

                for right in self._Coords[1:]:
                    point = np.array( (left, right))

                    self.Centers.append((point.sum() // 2).astype(np.int32))
                    self.Lines.append((point).astype(np.int32))

                    left = right


    def ExtDraw(self, image, coord):
        for rl in self.Owner.RedLineIndexes:
            cv2.polylines(image, [coord[rl]], False, (0, 0, 255), 4)

        return image

class TBurden(TTree):
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
        #if plg1.area * plg2.area == 0:
        #    a  = 0

        intersection_area = plg1.intersection(plg2).area

        # Вычисление процента площади пересечения относительно каждого прямоугольника
        percentage_area1 = (intersection_area / (plg1.area+0.000001))
        percentage_area2 = (intersection_area / (plg2.area+0.000001))

        return intersection_area, percentage_area1, percentage_area2

    def Mistake(self, hr): # по умолчанию речь про расстояние между этим и другими объектами
        polygon, err2 = self.CalcPolygon(hr)

        coords = np.int32(polygon.exterior.coords)

        '''image = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
        cv2.polylines(image, [coords], False, (0, 0, 0), thickness=2)
        cv2.polylines(image, [self.Owner.PlanCoord], self.DrawClosed, (0, 0, 255), 2)
        cv2.imwrite(f'c:/Demo/debug.jpg', image)'''

        prepare(polygon)

            # контроль выхода за пределы участка
        intersection_area, percentage_area1, percentage_area2 = self.calculate_intersection_area(polygon, self.Owner.Plan)#Coord)
        #err = max(0, 100*(1 - percentage_area1) - 0.01)
        err = 100 * (1 - percentage_area1)
        tp = self.ObjType

        rl = 100000000000
        rlCenters = 100000000000
        crRedline = None

        if percentage_area1 == 0:
            err += 200 * polygon.distance(self.Owner.Plan)
            return err
        else:
            if tp == house:
                ar = np.array(polygon.exterior.coords, dtype=np.int32)
                for line in self.Owner.RedLineAngledPoints:

                    crDist = polygon.distance(line[0])

                    if crDist < rl:
                        rl = crDist
                        crRedline = line[0]

                        center = line[1]
                        rlCenterDist = Point(ar[self.Doors[0]]).distance(center)


                dist_rl = Distances[house][red_line]
                crErr = self.Dist(rl, dist_rl, 100)
                err += crErr #1 / ((rl / dist_rl) ** 80 + 0.00001) / 100 # расстояние до красной линии, не меньше
                err += math.sqrt((rlCenterDist - dist_rl) ** 2)/10 # минимизируем расстояние до центра красной линии

                if self.Owner.Road:
                    if self.Garage is not None:
                        self.Owner.Road.Garage = ar[self.Garage]
                        self.Owner.Road.House = polygon
                    else:
                        self.Garage = None
            elif tp == garage:
                ar = np.array(polygon.exterior.coords, dtype=np.int32)
                for line in self.Owner.RedLineAngledPoints:

                    crDist = polygon.distance(line[0])

                    if crDist < rl:
                        rl = crDist
                        crRedline = line[0]

                        center = line[1]
                        rlCenterDist = Point(ar[self.Doors[0]]).distance(center)

                dist_rl = Distances[garage][red_line]
                #crErr = self.Dist(rl, dist_rl, 100)
                #err += crErr  # 1 / ((rl / dist_rl) ** 80 + 0.00001) / 100 # расстояние до красной линии, не меньше
                err += math.sqrt((rl - dist_rl) ** 2) / 10  # минимизируем расстояние до центра красной линии
                if self.Owner.Road:
                    self.Owner.Road.Garage = ar[self.Garage]
                    self.Owner.Road.House = polygon
            elif tp == outbuildings:
                lastLen = 0
                for line in self.Owner.RedLineAngledPoints:
                    # ищем минимальное расстояние до красной линии и сторону красной линии
                    crDist = polygon.distance(line[0])

                    crLen = line[0]

                    if crDist < rl or (lastLen < 0.3 * crLen and crDist / (rl+ 0.00001) > 0.7):
                        rl = crDist
                        crRedline = line[0]
                        lastLen = crLen

                err+= math.sqrt(self.Owner.Objects[0].S)/(rl + 0.000001)/10


        Water = None

        for obj in self.Owner.Objects:

            if obj == self:
                continue

            objTp = obj.ObjType
            dist = Distances[tp][objTp]

            if dist < 0.01 or (tp in [1,2] and (self.Garage is not None) and objTp == 17 and self.Owner.Road is not None):
                continue

            ls = obj.CalcLineString(hr)
            x = polygon.distance(ls)
            if np.isnan(x):
                err+= 20
            else:
                err += self.Dist(x, dist, 100)#32 функция стремится к 0 при растоянии больше item[1] (limit), и резко возрастает к 1

            if objTp != 0 and not np.isnan(x):
                p, _ = obj.CalcPolygon(hr)
                intersection_area, percentage_area1, percentage_area2 = self.calculate_intersection_area(polygon, p)

                err += 100 * (percentage_area2 + percentage_area1)

            if tp == house:
                if objTp in [cattle_barn, outbuildings, toilet, pit]:
                    d = ls.distance(crRedline)#(self.Owner.RedLines)

                    if d < rl:
                        err += rl - d

                if objTp in [septik_tank, pit]:

                    if Water is None:
                        if self.Water is not None:
                            Water = LineString(ar[self.Water])
                        else:
                            center = np.array(polygon.centroid.coords)[0]
                            Water = LineString((center + (1, 1), center - (1, 1)))

                    d = ls.distance(Water)

                    # септик и колодец на равном расстоянии от мокрых зон
                    err += 0.25*d/(1+math.exp(-((x-Distances[house][pit])*10)))/Distances[septik_tank][pit]

                    d1 = ls.distance(crRedline) * 0.005
                    err += d1

        return err                                 # если меньше

    def __init__(self, objType, polygon, Doors, image_path=False, image_angle=0):
        super().__init__(objType, polygon, image_path=image_path,image_angle=image_angle)  # limits = словарь расстояний до других объектов
        self.Doors = Doors

    def CalcCoords(self, hr, coords):
        if hr is None:
            return coords

        ind = (self.Index - 1) * self.Owner.ObjectDataSz
        gene = hr[ind:ind + self.Owner.ObjectDataSz].astype(np.int32)
        gene[0:2] = (gene[0:2]*self.Owner.K)

        coords = coords + gene[0:2]

        cnt = len(self.Owner.Angles)
        n = int(gene[2] * cnt / 256)

        if self.ObjType==outbuildings:
            p = LineString(coords)

            minDist = 1000000000
            for i, side in enumerate(self.Owner.Points):
                d = side.distance(p)

                if d < minDist:
                    minDist = d
                    ang = self.Owner.LineAngles[i]


        else:
            ang = self.Owner.Angles[n]

        ang += math.pi * int(gene[2] % 4) / 2

        return Geometry.NumpyPolygonRotate(coords, ang * 180/math.pi)

        '''p = Polygon(coords)
        rotated_polygon = affinity.rotate(p, gene[2] * 360/255, origin='centroid')
        return np.array(rotated_polygon.exterior.coords)'''

    def CalcImg(self, hr, Img):
        if hr is None:
            return Img

        height, width = image.shape[:2]

        ind = (self.Index - 1) * self.Owner.ObjectDataSz
        gene = hr[ind:ind + self.Owner.ObjectDataSz].astype(np.int32)
        gene[0:2] = (gene[0:2] * self.Owner.K)

        coords = coords + gene[0:2]

        center = self.Polygon.centroid

        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=gene[2] * 360 / 255, scale=1)

        # rotate the image using cv2.warpAffine
        rotated_image = cv2.warpAffine(src=Img, M=rotate_matrix, dsize=(width, height))

        return rotated_image

    def ProceeImage(self, img, hr, rotateCenter):
        if hr is None:
            return img

        ind = (self.Index - 1) * self.Owner.ObjectDataSz
        gene = hr[ind:ind + self.Owner.ObjectDataSz].astype(np.int32)
        gene[0:2] = (gene[0:2] * self.Owner.K)

        coords = self._Coords + gene[0:2]

        rotate_matrix = cv2.getRotationMatrix2D(center=rotateCenter - coords, angle=gene[2] * 360/255, scale=1)

        # rotate the image using cv2.warpAffine
        rotated_image = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(width, height))

        return rotated_image

class THouse(TBuilding):
    def CalcCoords(self, hr, coords):
        if hr is None:
            return coords

        ind = (self.Index - 1) * self.Owner.ObjectDataSz
        gene = hr[ind:ind + self.Owner.ObjectDataSz].astype(np.int32)
        try:
            gene[0:2] = (gene[0:2] * self.Owner.K)
        except:
            pass

        coords = coords + gene[0:2]

        p = Polygon(coords)

        cnt = len(self.Angles)
        n = int(gene[2] * cnt / 256)
        ang = self.Angles[n]

        self.AddError+= ang[1] # для каждого угла указан штраф за его использование

        rotated_polygon = affinity.rotate(p, ang[0], origin='centroid', use_radians=True)
        return np.array(rotated_polygon.exterior.coords)

    def __init__(self, polygon, Doors, Angles, Garage, image_path=False, image_angle=0, Water = None):
        super().__init__(objType=house, polygon=polygon, Doors=Doors, image_path=image_path,image_angle=image_angle)  # limits = словарь расстояний до других объектов

        self.Garage = Garage
        self._Angles = Angles
        self.Water = Water


'''
    Дорога. Представоляет собой объект сложной формы (вместе с въездной площадочкой)
'''
class TRoad(TBuilding):
    def Mistake(self, hr): # по умолчанию речь про расстояние между этим и другими объектами
        onlyGarden, _ = self.CalcPolygon(hr, woRoad=True)

        if not isinstance(onlyGarden, Polygon):
            return 200
        image = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
        cv2.polylines(image, [np.int32(onlyGarden.exterior.coords)], self.DrawClosed, (0,0,0), 2)
        cv2.polylines(image, [self.Owner.PlanCoord], self.DrawClosed, (0,0,255), 2)
        cv2.imwrite(f'c:/Demo/debug.jpg', image)

        # Автоплощадка должна быть в границах поля
        intersection_area, percentage_area1, percentage_area2 = self.calculate_intersection_area(onlyGarden,
                                                                                                 self.Owner.PlanCoord)
        err = 30 * (1 - percentage_area1)

        polygon, _ = self.CalcPolygon(hr, showLag = True)

        if not isinstance(polygon, Polygon) or polygon.is_empty:
            return 200


        #TestDraw([polygon, self.Owner.PlanCoord])

        redlines = self.Owner.NotRedLines
        for obj in redlines:
            err += 2*polygon.intersection(obj).length

        polygon = polygon.intersection(self.Owner.Plan)
        # Автоплощадка должна быть в границах поля
        if polygon.is_empty or isinstance(polygon, MultiPolygon):
            return err + 100

        # Проверяем на непересекаемость с остальными объектами всей дороги
        tp = self.ObjType
        for obj in self.Owner.Objects:

            if obj == self:
                continue

            objTp = obj.ObjType
            if objTp != 0:
                p, addEr = obj.CalcPolygon(hr)
                try:
                    intersection_area, percentage_area1, percentage_area2 = self.calculate_intersection_area(polygon, p)

                    err += 30 * (percentage_area2 + percentage_area1) + addEr
                except:
                    err+= 100


        return err # если меньше

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

    def CalcPolygon(self, hr, woRoad = False, showLag = False):
        '''
                    Если есть дом с гаражом, надо, чтобы площадку определял он. Для этого сперва обязательно просчитать
                    дом. При расчете дом, если у него есть гараж, сам заполняет соответствующие поля в TRoad: Garage (координаты
                    или None),
                    GaragePosition - где гараж расположен от площадки (0..3 - слева, справа, сверху, снизу)

                '''
        if self.Garage is not None: #Формируем площадку при въезде в гараж, на основании его позиции
            # try:
            start_point, end_point = self.Garage
            # except Exception as e:
            #    start_point = 0
            p0 = np.int32(start_point)
            p1 = np.int32(end_point)

            if self.Owner.DrawMode:
                sz = self.CalculateDrawPolygonSize(self.Sz)[0]
            else:
                sz = self.Sz[0]

            d = p1 - p0 # находим катеты треугольника, гипотенузой которого - въезд в гараж
            GarageSize = np.sqrt((d**2).sum()) # длина въезда

            a = sz/GarageSize * d[1] # один из катетов для расчета боковой стороны дорожки (ее проекция на ось Х)

            # ищем точки на перепендикулярной прямой от угла гаража - слева и справа
            if d[1] == 0: # прямой вертикальный угол
                p2_0 = p0 + [a, 0]
                p2_1 = p0 - [a, 0]
            else:
                k1 = -d[0] / d[1]
                p2_0 = p0 + np.array((a, a * k1))
                p2_1 = p0 - np.array((a, a * k1))

            # проверяем, в какую сторону от дома рисовать площадку
            l0 = self.House.intersection(LineString((p1, p2_0))).length
            l1 = self.House.intersection(LineString((p1, p2_1))).length

            if l1 < l0:
                p2_0 = p2_1
                if d[1] == 0:  # прямой вертикальный угол
                    p2_1 = p1 - [a, 0]
                else:
                    p2_1 = p1 - np.array((a, a * k1))
            else:
                if d[1] == 0:  # прямой вертикальный угол
                    p2_1 = p0 + [a, 0]
                else:
                    p2_1 = p1 + np.array((a, a * k1))

            return Polygon( (p0, p1, p2_1, p2_0, p0) ), 0
            '''

            direction_vector = (end_point - start_point)[::-1]

            length = math.sqrt((direction_vector*direction_vector).sum())

            # Нормализуем направление отрезка
            sin_cos = (direction_vector / length)* (-1, 1)
            add0 = self.Sz[0] * sin_cos

            # Вычисляем координаты  вершин прямоугольника
            vertex1 = start_point + add0
            vertex2 = end_point + add0

            p = (start_point, vertex1, vertex2, end_point)
            if woRoad:
                return Polygon(p), 0

            add1 = self.Roadwidth * sin_cos

            vertex1_1 = start_point + add1
            #vertex1_2 = end_point + add1

            direction_vector2 = direction_vector * 100

            vertex3 = vertex1 - direction_vector2
            vertex4 = vertex1_1 - direction_vector2

            return Polygon( (vertex4, vertex3, vertex2, end_point, start_point, vertex1_1 )), 0'''
        # рисуем ногу


        else:
            if hr is None:
                gene = (0, 0, 0, 0)
            else:
                ind = (self.Index - 1) * self.Owner.ObjectDataSz
                gene = hr[ind:ind + 4].astype(np.int32)
                gene[0:2] = gene[0:2] * self.Owner.K
                gene[2:] = gene[2:] * self.K2

            p = self.CalcBasePolygon(gene[:2], (gene[3]), woRoad )


            rotated_polygon = affinity.rotate(p, gene[2], origin=(p.exterior.coords.xy[0][0], p.exterior.coords.xy[1][0]))
        #rotated_polygon = p
        if woRoad:
            return rotated_polygon, 0

        if not showLag and self.Garage is None:
            res_polygon = rotated_polygon.intersection(self.Owner.Plan)
        else:
            return rotated_polygon, 0
        #dif = rotated_polygon.difference(res_polygon)

        #if isinstance(dif, Polygon):
        #    d = 0 if len(dif.exterior.coords.xy[0]) == 5 and -1000 in dif.exterior.coords.xy[0] else 2
        #else:
        #    d = 2
        return res_polygon, 0

    def CalcCoords(self, hr, coords = None):
        plg, _ = self.CalcPolygon(hr)

        if isinstance(plg, Polygon) and  not (plg.is_empty):
            return np.array(plg.exterior.coords)
        else:
            return []


    # sz (x,y)
    def __init__(self, sz = (70, 50), DAngle = 10, roadwidth = 40, image_path = 'autoroads.jpg'):

        self.Sz = sz
        self.Roadwidth = roadwidth

        self.ObjType = road
        self.S = 0

        self.Owner = None

        self.Index = TObject.CrIndex
        TObject.CrIndex += 2

        self.Color = get_rgb_from_color_name(TGardenGenetic.ColorsSet[self.ObjType])
        self.LineWidth = 2 # limits = словарь расстояний до других объектов

        self.K2 = (360/255, (sz[0] - roadwidth)/255)

        self.RPos = (0, 0)

        self.DrawClosed = True

        self.image_path = image_path
        self.DrawBoundsWithImage = True
        self._Coords = (0,0)

        if image_path:
            img_colored = cv2.imread(image_path, cv2.IMREAD_COLOR)

            self.image = cv2.resize(img_colored, (image_width, image_height))
        else:
            self.image = None

        self.Garage = None
        self.GaragePosition = 0
        self.AddError = 0

def Draw(self, image, hr, img = None):
        coord0 = self.CalcCoords(hr, None)

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

class TGardenGenetic(genetic.TBaseGenetic):
        ObjectDataSz = 3
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
                    11: 'lavendef',
                    12: 'lime',
                    13: 'maroon',
                    14: 'indigo',
                    15: 'red',
                    16: 'pink',
                    17: 'black',
                    18: 'black'
                    }
        Colors = list(ColorsSet)

        def CorrectStep(self, step2, crHr):
            step1 = np.where(step2 + crHr < 0, 0, step2)
            step = np.where(step1 + crHr > 255, 0, step1)

            if (step2 != step).sum() > 0:
                step = step

            return step

        def __init__(self, objects, plan, rlines, theBestListSize=1000,
                     populationSize=4000, name = 'ga_', image = None,
                     Clasters=50, kNovelty=(0.98, 0.01, 0.99)):  # rooms - список номеров комнат и их предполагаемых площадей в формате двумерного списка

            self.DrawMode = False

            if image is not None:
                cv2.polylines(image, [plan + (20,20)], False, (0, 0, 0), 4)

                for rl in rlines:
                    p0 = plan[rl[0]]
                    p1 = plan[rl[-1]]
                    cv2.polylines(image, [np.array((p0+20, p1+20))], False, (0, 0, 255), 4)

                cv2.imwrite('c:/Demo/plan.jpg', image)

                # Создаем описание границ - LineString, угол.
                # Угол до 180 если снизу - слева. Или больше, если сверху справа
            point0 = plan[0]
            points = []
            angles = []

            planObj = TBounds(plan)
            for point1 in plan[1:]:
                d = point1 - point0

                if d[0]==0:
                    if d[1] == 0:
                        continue # точки равны

                    angle = math.pi/2 if d[1] < 0 else 2*math.pi/2
                else:
                    k = d[1]/d[0]
                    k1 = -100/k
                    angle = math.atan(k)

                    ptest0 = point1 + np.array((100, k1))
                    ptest1 = point1 - np.array((100, k1))

                    l0 = planObj.Polygon.intersection(LineString((point1, ptest0))).length
                    l1 = planObj.Polygon.intersection(LineString((p1, ptest1))).length

                    #ln = np.int32((ptest0 + (20, 20), ptest1 + (20, 20)))

                    #cv2.polylines(image, [ln], False, (0, 0, 0), 4)

                    #cv2.imwrite('c:/Demo/plan2.jpg', image)

                    if l1 > l0 or angle < 0:
                        angle+= math.pi

                angles.append(angle)
                points.append(LineString((point0, point1)))

                point0 = point1

            self.LineAngles = angles
            self.Points = points

            self.Name = name
            self.Road = None
            planObj = TBounds(plan)

            center = planObj.Center

            angleLimit = (0.1, 10)
            angles = []
            firstLine = True
            linePoints = False
            rlPoints = []
            p0 = None
            d0 = None

            self.RedLineIndexes = rlines

            redlines = []
            for rl in rlines:
                redlines.append(plan[rl])

            #self.RedLines = []

            '''
                Далее определяем углы красных линий. Если линия состоит из ряда смежных участков, то делаем простую линейную апроксимацию -
                пока разница углов небольшая, просто соединяем вход и выход отрезка. Иначе - создаем новый отрезок
            '''
            rlLines = set()

            for rl in redlines:

                #self.RedLines.append(LineString(rl))

                firstLine = True
                for p1 in rl:

                    newLine = False

                    if p0 is not None: # это не первая точка - иначе просто запоминаем ее

                        if p1[0] == startPoint[0] and p1[1] == startPoint[1]:
                            continue

                        if (tuple(p1), tuple(p0)) not in rlLines: # на всякий проверяем и перевернутый отрезок
                            rlLines.add((tuple(p0), tuple(p1)))

                        # определяем угол от данной точки до начала отрезка
                        d = (p1[1] - startPoint[1]),(p1[0] - startPoint[0])

                        if not firstLine: # если это не первый отрезок
                        # сравниваем с прошлым углом
                            if abs(d[0]) > abs(d[1]):
                               if d0[0] == 0:
                                   newLine = True
                               else:
                                   newLine = abs(math.atan(d[1]/d[0]) - math.atan(d0[1]/d0[0])) > math.pi/3
                            else:
                                if d0[1] == 0:
                                    newLine = True
                                else:
                                    newLine = abs(math.atan(d[0] / d[1]) - math.atan(d0[0] / d0[1])) > math.pi / 3

                        else:
                            firstLine = False
                            d0 = d

                        if newLine: # если новый отрезок
                            #rlPoints.append([startPoint, p0])
                            rlPoints.append([p0, startPoint])
                            startPoint = p0
                            firstLine = True
                    else:
                        startPoint = p1

                    p0 = p1 # прошлая точка

                if not firstLine:
                    rlPoints.append([startPoint, p0])

                    # rlPoints теперь - список прямых, их которых состоит красная линия. По ним надо выверить углы
                    # Сперва запоминаем все наклоны красных линий участка
                Pi2 = math.pi / 2


                for p in np.array(rlPoints):
                    p0 = p[1]-p[0]
                    p1 = (p[1]+p[0])/2 #середина красной линии

                    # ищем точки на перепендикулярной прямой от середины отрезка - слева и справа
                    if p0[1] == 0:
                        ptest0 = p1 + [100, 0]
                        ptest1 = p1 - [100, 0]
                    else:
                        k1 = -p0[0]/p0[1]
                        ptest0 = p1 + np.array((100,100*k1))
                        ptest1 = p1 - np.array((100,100*k1))

                    l0 = planObj.Polygon.intersection(LineString((p1,ptest0))).length
                    l1 = planObj.Polygon.intersection(LineString((p1, ptest1))).length

                    #ln = np.int32((ptest0 + (20, 20), ptest1 + (20, 20)))

                    #cv2.polylines(image, [ln], False, (0, 0, 0), 4)

                    #cv2.imwrite('c:/Demo/plan2.jpg', image)

                    add180 = l1 > l0

                    if p0[0] != 0:
                        t = math.atan(p0[1]/p0[0])
                        if t<0:
                            t+= math.pi
                    else:
                        t = math.pi/2

                    a = t + (math.pi if add180 else 0)

                    angles.append(a)



            self.Angles = angles

            self.RedLineAngledPoints = []

            for p in rlPoints:
                ls = LineString(p)
                prepare(ls)
                ct = ls.centroid
                prepare(ct)

                self.RedLineAngledPoints.append( [ls, ct])


            self.RedLines = list(rlLines) # отрезки, относящиеся к RL

            # Теперь создаем множество отрезков, которые не входят в RL

            lines = []
            p0 = None
            for p1 in plan:
                if p0 is not None:
                        p = (tuple(p0), tuple(p1))
                        notp = (tuple(p1), tuple(p0))

                        if p in rlLines or notp in rlLines:
                            continue

                        lines.append( (p0, p1) )

                p0 = p1

            ls = Geometry.create_point_list(lines)



            self.NotRedLines = [LineString(line) for line in ls]


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
                obj = self.Objects[i]
                obj.Owner = self

                if isinstance(obj, TBuilding):
                    hrSize+= 1

                    if isinstance(obj, TRoad):
                        self.Road = obj

                        hrSize += 1
                    elif isinstance(obj, THouse):
                        # для домов - список допустимых углов на данном участке
                        #frontAngle = math.atan2(obj.Front[0], obj.Front[1])

                        maxRl = 0

                        #angles = obj.Angles
                        #obj.Angles = []
                        for ang in self.Angles:
                            obj.Angles = []
                            for i, an in enumerate(obj._Angles):
                                a = ang - an[0]

                                rotated = affinity.rotate(obj.Polygon, a, origin='centroid', use_radians=True)
                                obj.Angles.append( (ang - an[0], an[1]) )


            self.PlanCoord = plan

            self.Plan = Polygon(plan)

            super().__init__(hrSize * self.ObjectDataSz, GenGroupSize=self.ObjectDataSz,
                             TheBestListSize=theBestListSize, PopulationSize=populationSize,
                             Clasters=Clasters, kNovelty=kNovelty)

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


                res[r[0],r[1]] = 0


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
            self.Tests += 1
            err = 0

            hrIndex = 0
            for obj in self.Objects:

                if obj == self.Road:
                    continue

                obj.AddError = 0
                err += obj.Mistake(hr)
                err+= obj.AddError * 5

            if self.Road:
                self.Road.AddError = 0
                err += self.Road.Mistake(hr)
                err += self.Road.AddError

            return err

        def CalculatePolygon(self, A):
            A0 = np.array(A)
            A2 = np.empty_like(A0)  # Создаем новый массив такого же размера как A0

            # Обменяем элементы в последнем измерении
            A2[:, 0] = A0[:, 1]  # A2[i,0] = A0[i,1]
            A2[:, 1] = self.PlanCoord.max() - A0[:, 0]

            return (A2 * 5 + 20).astype(np.int32)

        def Draw(self, image, hr=None, drawRoads = True):
            self.DrawMode = True
            for obj in self.Objects:
                image = obj.Draw(image, hr)

            if drawRoads and hr is not None:
                cv2.imwrite('c:/Res/test_.jpg', image)
                roads = self.CalcRoads(hr)
                roads.Owner = self
                roads.Draw(image, hr)
                cv2.imwrite(f'Res/test_roads{self.Name}_{self.UsefullMetric}.jpg', image)

            self.DrawMode = False

            return image #Geometry.rotateImg(image, self.GardenAngle, 255)

        def Visualisation(self, hr, ind, metric):
            image = np.full((image_height, image_width, 3), 255, dtype=np.uint8)

            if self.UsefullBestHr is None:
                print('Лучшая хромосома не определена')
                return

            list = np.argsort(self.BaseHromosomRatingValues)

            #image_height, image_width = 800, 800  # Размеры изображения
            image = self.Draw(image, self.UsefullBestHr, True)



            cv2.imwrite(f'Res/{self.Name}best_' + str(self.UsefullMetric) + '.jpg', image)
            np.save(f'ResNpy/{self.Name}best_' + str(self.UsefullMetric) + '.npy', self.UsefullBestHr)
            f = False

            if f:
                print('Проверка best ', str(self.UsefullMetric) + '_' + str(self.UsefullBestHr))
                print(self.TestHromosom(self.UsefullBestHr))

            for i, h in enumerate(self.Hromosoms[list[:self.BestForShowingCnt]]):
                #h = np.full_like(h, 255)
                #image_height, image_width = 800, 800  # Размеры изображения
                image = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
                try:
                    image = self.Draw(image, h, False) #self.UsefullMetric < 0.01)
                except:
                    print('ошибка рисования')

                v = self.BaseHromosomRatingValues[list[i]]

                cv2.imwrite(f'Res/{self.Name}' + str(ind) + '_' + str(i) + '_' + str(v) + '.jpg', image)
                np.save(f'ResNpy/{self.Name}' + str(ind) + '_' + str(i) + '_' + str(v) + '.npy', h)

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
                                           'Автоплощадка',
                                           'Дорожки',
                                           'Ограниченеия']

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
burden = 20



# Граница участка
Distances[boundary][garage] = 30
Distances[boundary][house] = 30#70
Distances[boundary][outbuildings] = 30
Distances[boundary][septik_tank] = 30
Distances[boundary][toilet] = 30
Distances[boundary][pit] = 30
Distances[house][red_line] = 70
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
Distances[house][pit] = 40
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




p0 = (0,0)


#Внимание! Входной формат - (1, n, m)
def ConvertPlans(plans, K = 10):
    mn = np.array((float('infinity'),float('infinity')))
    pl = np.array(plans, dtype = np.float32)
    for obj in pl:
        crMn = obj.min(0)

        if crMn[0]< mn[0]:
            mn[0] = crMn[0]

        if crMn[1]< mn[1]:
            mn[1] = crMn[1]

    res = [((obj - mn) * K) for obj in plans]

    for i in range(len(res)):
        res[i] = np.concatenate([res[i], [res[i][0]]])

    return np.array(res[0], dtype = np.int32)

def ReadFromStorer(House):

    h =  THouse(np.array(House.Points), Doors=House.Doors, Angles = House.Angles,
                    Garage = House.Garage, image_path='Houses/' + House.File, Water = House.Water)
    h.GaragePosition = House.GaragePosition

    return h


warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

Plan0 = np.array([[210, 0], [210, 259], [94, 259], [94, 100], [0, 100], [0,  0], [210, 0]], dtype = np.int32)


from house0 import *
from house2 import *

'''
1. Дом
2. Гараж
3. Хозпостройка
4. Септик
5. Туалет
6. Дорога и автоплощадка
'''

'''trees = [TTree(medium_trees, (2033, 1009)),
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
        TTree(medium_trees, (2122, 3196))]'''



Plans = []

Plan = np.array([[0, 81], [0, 71], [9, 69], [495, 0], [554, 1], [560, 58], [572, 164], [572, 164], [577, 201], [593, 324],
                [594, 331], [589, 332], [11, 380], [2, 217], [2, 217], [3, 111], [0, 81]])

#NotRedLines = LineString(np.array([[0, 81], [0, 71], [9, 69], [495, 0], [554, 1], [560, 58], [572, 164], [572, 164], [577, 201], [593, 324],
#                [594, 331], [589, 332], [11, 380]]))
RedLines = [[1,0,-1,-2,-3,-4,-5]]
Plans.append( (Plan, None, RedLines))


paths = [[[-39.44458, -252.8039], [1143.8928, -252.8039], [1672.8088000000002, 207.98050999999998], [1147.4787000000001, 984.32156], [-193.63702999999987, 984.32156], [-437.9388, 561.17848], [-437.9388, -3.5858709]]]
#paths = [[[-39.44458, -252.8039], [1143.8928, -252.8039], [1672.8088000000002, 207.98050999999998], [1147.4787000000001, 984.32156], [-437.9388, -3.5858709]]]
RedLines = [[1,2]]
'''paths[0][1][0] -= 600
paths[0][2][0] -= 600
paths[0][3][0] -= 600'''

Plan = ConvertPlans(paths, 0.27)
Plans.append( (Plan, None, RedLines))

paths= [[443925.62, 1366365.18],
   [443929.86, 1366396.46],
   [443902.19, 1366405.14],
   [443892.63, 1366367.23],
   [443925.62, 1366365.18]]

paths = Geometry.NumpyPolygonRotate(paths, -90)

Plan = ConvertPlans([paths], 10)
RedLines = [[1,2]]

Plans.append( (Plan, None, RedLines))

paths= [(2171347.34, 467862.48), (2171347.49, 467863.69), (2171342.99, 467896.65), (2171342.24, 467902.1), (2171417.44, 467912.51),
 (2171428.57, 467913.03), (2171438.05, 467910.34), (2171445.25, 467904.09), (2171447.55, 467902.1), (2171448.35, 467898.88),
 (2171430.92, 467895.34), (2171416.17, 467890.09), (2171401.76, 467883.74), (2171380.37, 467869.05), (2171376.29, 467866.81),
 (2171377.64, 467863.16), (2171366.83, 467860.1), (2171347.34, 467862.48)]

burden = [(2171376.2764034085, 467864.54948750924), (2171370.989419742, 467863.5200748104), (2171349.6712042787, 467865.16993670125),
 (2171347.2328627612, 467865.57338742254), (2171346.3869405356, 467871.76929776877), (2171350.518377876, 467871.0901324413),
 (2171350.5192363826, 467871.0900291982), (2171371.0192363826, 467869.52002919826), (2171371.020173339, 467869.5200015025),
 (2171371.021108773, 467869.520061659), (2171371.0220344644, 467869.5202091393), (2171374.1020344645, 467870.1602091392),
 (2171374.1030889936, 467870.1604890527), (2171374.104106752, 467870.1608821829), (2171374.1050756243, 467870.16138385015),
 (2171413.673761099, 467893.4706094863), (2171447.717491795, 467901.4258455251), (2171448.35, 467898.88), (2171430.92, 467895.34),
 (2171416.17, 467890.09), (2171401.76, 467883.74), (2171380.37, 467869.05), (2171376.29, 467866.81), (2171376.974059558, 467864.9605056397),
 (2171376.2764034085, 467864.54948750924)]

RedLines = [[0,1]]
l = len(paths)
p = paths
p.extend(burden)
paths = Geometry.NumpyPolygonRotate(p, -90)

Plan = ConvertPlans([paths], 10)
burden = Plan[l:]
Plan  = Plan[:l]
#RedLines = [[1,2], [2,3]]
RedLines = [[3,4,5]] #ghjnbd часовой

Plans.append( (Plan, burden, RedLines))

paths= [(2171597.15, 467858.06), (2171592.62, 467863.19), (2171610.69, 467907.02), (2171612.82, 467905.73), (2171621.84, 467898.65),
 (2171648.84, 467892.54), (2171639.38, 467843.11), (2171624.55, 467846.88), (2171593.94, 467854.67), (2171597.15, 467858.06)]

burden = [(2171632.6298196013, 467845.20810911816), (2171632.629957257, 467845.2090763983), (2171634.4090357632, 467864.3891418701),
 (2171638.3691213927, 467864.0309431699), (2171636.580087465, 467844.65140805545), (2171636.427850656, 467843.86047896335),
 (2171632.55967246, 467844.84382567945), (2171632.6298196013, 467845.20810911816)]
l = len(paths)
p = paths
p.extend(burden)
paths = Geometry.NumpyPolygonRotate(p, -90)



Plan = ConvertPlans([paths], 10)
burden = Plan[l:]
Plan  = Plan[:l]
#RedLines = [[1,2], [2,3]]
RedLines = [[3,4,5]] #ghjnbd часовой


Plans.append( (Plan, burden, RedLines))

paths= [[2171441.94, 467806.12], [2171445.93, 467805.83], [2171473.81, 467803.82], [2171468.66, 467768.93], [2171457.6, 467751.18],
 [2171453.24, 467736.37], [2171449.65, 467715.72], [2171435.22, 467721.27], [2171417.2, 467728.42], [2171414.08, 467734.24],
 [2171418.22, 467742.82], [2171418.62, 467742.69], [2171431.97, 467769.61], [2171435.53, 467777.62], [2171441.12, 467790.22],
 [2171441.94, 467806.12]]

burden = [(2171597.15, 467858.06), (2171592.62, 467863.19), (2171610.69, 467907.02), (2171612.82, 467905.73), (2171621.84, 467898.65),
 (2171648.84, 467892.54), (2171639.38, 467843.11), (2171624.55, 467846.88), (2171593.94, 467854.67), (2171597.15, 467858.06)]

l = len(paths)
p = paths
p.extend(burden)
paths = Geometry.NumpyPolygonRotate(p, -90)



Plan = ConvertPlans([paths], 10)
burden = Plan[l:]
Plan  = Plan[:l]
#RedLines = [[1,2], [2,3]]
RedLines = [[6,8]] #ghjnbd часовой


Plans.append( (Plan, burden, RedLines))



paths = [(2171441.94, 467806.12), (2171445.93, 467805.83), (2171473.81, 467803.82), (2171468.66, 467768.93), (2171457.6, 467751.18),
 (2171453.24, 467736.37), (2171449.65, 467715.72), (2171435.22, 467721.27), (2171417.2, 467728.42), (2171414.08, 467734.24),
 (2171418.22, 467742.82), (2171418.62, 467742.69), (2171431.97, 467769.61), (2171435.53, 467777.62), (2171441.12, 467790.22),
 (2171441.94, 467806.12)]

burden = [[2171448.097435237, 467792.53930879466], [2171445.619485555, 467799.96316610737], [2171445.619126736, 467799.9640868916],
           [2171445.618678786, 467799.9649677634], [2171445.618146079, 467799.96580012026], [2171445.617533818, 467799.9665758335],
           [2171445.616847982, 467799.9672873275], [2171445.616095269, 467799.9679276538], [2171445.61528303, 467799.96849055914],
           [2171445.614419197, 467799.96897054603], [2171445.6135122064, 467799.96936292714], [2171445.612570916, 467799.96966387035],
           [2171445.6116045183, 467799.96987043676], [2171445.6106224507, 467799.96998060896], [2171445.609634304, 467799.9699933111],
           [2171445.6086497293, 467799.9699084191], [2171445.6076783407, 467799.969726762], [2171441.5731523554, 467799.0067346999],
           [2171441.786918277, 467803.1517080547], [2171448.2334125475, 467804.6881498613], [2171451.05051429, 467796.2468343548],
           [2171451.050860628, 467796.24594144325], [2171451.051290796, 467796.24508575845], [2171451.051800849, 467796.244275149],
           [2171451.0523861074, 467796.24351705034], [2171451.0530412034, 467796.2428184159], [2171451.0537601286, 467796.24218565394],
           [2171451.0545362886, 467796.24162456836], [2171451.0553625636, 467796.24114030565], [2171451.0562313753, 467796.2407373077],
           [2171451.0571347545, 467796.240419271], [2171451.0580644147, 467796.2401891127], [2171451.059011829, 467796.2400489439],
           [2171472.3786486695, 467794.12294214964], [2171471.7964072363, 467790.17839776125], [2171448.097435237, 467792.53930879466]]

l = len(paths)
p = paths
p.extend(burden)
paths = Geometry.NumpyPolygonRotate(p, -90)



Plan = ConvertPlans([paths], 10)
burden = Plan[l:]
Plan  = Plan[:l]
#RedLines = [[1,2], [2,3]]
RedLines = [[6,8]] #ghjnbd часовой


Plans.append( (Plan, burden, RedLines))

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


def GA(PlanIndex, NPY = None, BestForShowingCnt = 0):
    if NPY is not None:
        PlanIndex = int(NPY[0])
        l1 = int(NPY[2])
    else:
        l1 = 0

    l0 = PlanIndex
    start = l1

    Garage = ReadFromStorer(TestHouses.Results[-1])
    Garage.ObjType = garage
    for H in TestHouses.Results[start:-1]:
        Plan, burden, rl = Plans[PlanIndex]
        if True:
        #try:
            Caption = f'{l0}_{l1}'
            print('Цикл ' + Caption)
            Objects = []
            #Plan = np.array([[210, 0], [210, 259], [94, 259], [94, 100], [0, 100], [0,  0], [210, 0]], dtype = np.int32)

            #Obj0 = TBuilding(house, [[0,0], [100, 120]], Doors= ((100, 60)), Front = ( (100, 0), (100, 120)), image_path = 'house2.jpg') # [137, 169]])
            # Внимание! Объект с гаражом должен идти первым!!!
            #Надо штрафовать расстояние от дома до гаража?
            ObjHouse = ReadFromStorer(H)
            #if ObjHouse.Garage is None:
            #    Objects.append(Garage)
            Objects.append(ObjHouse)

            if burden is not None:
                Objects.append(TBurden(burden))
            #Obj1 = TBuilding(garage, [[0,0], [59, 59]])
            Objects.append( TBuilding(outbuildings, [[0, 0], [30, 60]], [[30, 30]], image_path='bathhouse.jpg'))
            #Objects.append( TBuilding(outbuildings, [[0, 0], [30, 60]], [[30, 30]], image_path='bathhouse2.jpg'))# [60, 90]
            Objects.append(TBuilding(septik_tank, [[0, 0], [25, 30]], [[25, 15]]))#, image_path='septic.jpg'))
            Objects.append(TBuilding(pit, [[0, 0], [30, 30]], [[30, 15]]))#, image_path='pit.jpg'))
            Objects[-1].DrawBoundsWithImage = Objects[-2].DrawBoundsWithImage = False

            #if ObjHouse.Garage is not None:
            #    Objects.append(TBuilding(outbuildings, [[0, 0], [30, 60]], [[30, 30]], image_path='bathhouse2.jpg'))  # [60, 90]

            Objects.append(TRoad())


            n = 6
            random.seed(n)
            np.random.seed(n)
            # Создаем чистое изображение
            #image_height, image_width = 800, 800  # Размеры изображения
            image = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
            '''Gn = TGardenGenetic(Objects, Plan, rlines=rl, theBestListSize=100, populationSize=500, name = f'{Caption}___', image = image)


            #Gn.RedLine = TBounds(Gn.RedLines)
            #Gn.RedLine.Owner = self
            #Gn.Draw(image)
            #cv2.imwrite(f'c:/Res/Mask.jpg', image)
            #Gn = TRoomGenetic(rooms, coord)
            Gn.PDeath = 0.75
            Gn.StopFlag = 0
            Gn.kNovelty = 1
            Gn.isStranger = 1.8
            Gn.PMutation = 0.05  # probability of the mutation for any individuals. Or, if >=1, the number of individuals that will die in each generation
            Gn.PMultiMutation = 0.1  # the probability of an any bit to be mutated, in the case of the mutation has occurred
            # self.PDeath = 0.2 # probability of the death. Or, if >=1, the number of individuals that will die in each generation
            Gn.PCrossingover = 0
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
            Gn.BestForShowingCnt = BestForShowingCnt

            Gn.StopFlag = 3
            Gn.GenerationsLimit = 5000
            Gn.MetricLimit = 0.1'''

            Gn = TGardenGenetic(Objects, Plan, rlines = rl, theBestListSize = 100, populationSize = 500,
                                Clasters=50, kNovelty=(0.50, 0.01, 0.99),
                                name = f'{Caption}___', image = image)
            Gn.PDeath = 0.85
            Gn.PMutation = 0.25  # probability of the mutation for any individuals. Or, if >=1, the number of individuals that will die in each generation
            Gn.MutationStopFlag = 0.70
            Gn.PMultiMutation = 0.3
            Gn.PCrossingover = 0.5
            Gn.StopFlag = 4
            Gn.MetricLimit = 0.0000001
            Gn.useNovelty = True
            Gn.useClastering = False
            Gn.avgPairingMask = None
            Gn.GradientEttempts = 1
            Gn.UseGradient = True
            Gn.VisualizationStep = 1
            Gn.BestForShowingCnt = 1
            Gn.DebugInfo = True
            Gn.RecreatePopulation = False
            Gn.StrongMutations = False

            if NPY is None:
                Gn.Start()
            else:
                Gn.Start(f'ResNpy/{NPY}')

            if NPY is not None:
                break
        #except Exception as e:
        #    print(f'ОШИБКА!!! {e}')
        TObject.CrIndex = 1
        #Gn.Start('c://res//0_0___best_19.185848.npy')

        l1+= 1

if __name__ == "__main__":
    GA(0)#, "4_7___18_0_390.90665.npy")