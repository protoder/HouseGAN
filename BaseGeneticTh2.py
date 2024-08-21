"""
    Основной модуль класса генетического алгоритма. Заложен, но не отлажен, механизм многопоточности.
"""
import glob
import math
from multiprocessing import Process, Value, Array
import os
import os.path
import pickle
import random
import threading
import time

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans, KMeans


print('bgn v 13.0')


def Relu(x, add)->float:
    """
        Реализация функции y = Relu(x)

        x - входная переменная (скаляр)
        add - угол наклона

        Вернет y
    """
    if x > 0:
        return x * add
    return 0.0


def Dist(x, limit, mx, add)->float:
    """
       Вернет ошибку от 0 до mx в зависимости от того, на сколько нарушена дистанция (mx при дистанции, равной 0.
       0 - при равной или большей limit

       x - входная переменная (скаляр)
       limit - минимальная допустимая дистанция
       mx - максимальная ошибка
       add - угол наклона

       Возвращает значение ошибки
    """
    return mx * Relu(limit - x, add) / limit

"""
    Многопоточность. В чем идея.
    Часть кода мы выносим в отдельные процедуры. Которые выполняют операции над частью данных.
    Какие именно данные будут обрабатываться, рассчитывается заранее
"""

def CreateFirstPopulation(self, index, BaseHromosomRatingValues, Hromosoms):
    """
        Метод создания исходной популяции. Вызывается в рамках многопроцессорной обработки, создает и просчитывает
        часть всего начального набора хромосом

        index - номер процесса
        BaseHromosomRatingValues - ссылка на объект, хранящий результаты оценки хромосом
        Hromosoms - ссылка на объект, хранящий тела хромосом
    """
    sz = self.GensInThread  # Количество генов для одного процесса (HromosomLen * self.PopulationsInThread)

    start = self.PopulationsInThread * index # исходная позиция для данного потока

    hr_ravel = np.random.randint(0, 256, sz) # генерируем тела хромосом
    hr = hr_ravel.reshape((self.PopulationsInThread, self.HromosomLen))

    # Просчет созданных хромосом
    for i, hromosom in enumerate(hr, start):
        BaseHromosomRatingValues[i] = self.TestHromosom(hromosom)

    # Заполняем хромосомы
    start = sz * index
    Hromosoms[start:start+sz] = hr_ravel


def TestHromosomsTh(self, index, mutates):
    """
        Метод рассчета нерасчитанных хромосом. Предназначен для работы в многопроцессорной системе

        index - номер процесса
        mutates - номера измененных мутациями хромосом (их надо пересчитать, даже если они рассчитаны)
    """
    if len(self.Hromosoms) > self.TheBestListSize:
        # расчет еще не расчитанных хромосом
        start = index * self.PopulationsInThread3 + self.TheBestListSize  # CrBestSize

        for i, hr in enumerate(self.Hromosoms[start: min(len(self.Hromosoms), start + self.PopulationsInThread3)],
                               start):
            try:
                self.BaseHromosomRatingValues[i] = self.TestHromosom(hr)
            except:
                self.BaseHromosomRatingValues[i] = self.TestHromosom(hr)

        start = index * self.BestListsInThread1
        end = start + self.BestListsInThread1

        for n in mutates[mutates < end]:
            if n >= start:
                self.BaseHromosomRatingValues[n] = self.TestHromosom(self.Hromosoms[n])

    # для отобранных хромосом имитация градиентного спуск
    if self.UseGradient:  # and clasters[claster] == 0:
        start = index * self.BestListsInThread
        for i, item in enumerate(self.Hromosoms[start:start + self.BestListsInThread], start):
            raiting = self.BaseHromosomRatingValues[i]
            newRaiting, newHr = self.GradientOptimisation(item, raiting, i)
            if newRaiting < raiting:
                self.Hromosoms[i] = newHr
                self.BaseHromosomRatingValues[i] = newRaiting

    '''Cоздаем массив родителей в мультипоточности - потом они будут использоваться при размножении.
       parentX/Y - индекс первого и второго родителя. Диапозон выбора - TheBestListSize + NoveltyCnt, 
       то есть одна из прошедших отбор хромосом. Размер - нужное количество потомков, то есть размер популяции 
       минус количество прошедших отбор (иными словами, компенсируем потери)
       parentY
    '''
    step = self.GensInThread2# генов сколько надо создать
    start = index * step
    end = start + step

    step2 = self.PopulationsInThread2 # сколько надо создать хромосом
    start2 = index * step2
    end2 = start2 + step2

    bestSz = self.HromosomLen * self.CrBestSize

    flags = np.random.randint(0, 2, step, np.uint8)
    self.pearentIndex[start:end] = flags
    self.pearentX[start2:end2] = np.random.randint(0, self.CrBestSize, step2, np.int32)
    self.pearentY[start2:end2] = np.random.randint(0, self.CrBestSize - 1, step2, np.int32)

    self.pearentY[self.pearentX == self.pearentY] += 1


class TBaseGenetic:
    '''
        Предусмотрено восстановление работы из архива. Но для востановления надо установить переменную
        self.TryLoadOnStart = True

        HromosomLen - размер хромосомы.
        StopFlag: 0 - never stop
                  1 - stop after n populations
                  2 - stop when Metric is More or Less then MetricLimit
                  4 - stop после
                  128 - признак стоп вне зависимости от всех остальных

        GenGroupSize - для удобства, гены можно разбить на группы. Например, относящиеся к одному помещению.

        FixedGroupsLeft - сколько элементов слева не участвуют в рабое алгоритма. Они нужны для хранения информации в хромосоме.


        TheBestListSize - количество элементов Best- списка. В нем хранятся лучшие хромосомы. Все остальные умирают каждое поколнеие
        StartPopulationSize  При запуске сперва создается заданное количество случайных хромосом
        PopulationSize - в данной системе используется популяция фиксированного размера. Должна быть сильно больше TheBestListSize
    '''

    '''
       Поля объекты и методы
           Hromosoms                 - текущий набор хромосом
           HromosomRatingValues      - рейтинг для алгоритма. Он расчитывается на основании пользовательских (полезных) критериев (целевой функции), и критериев
                                       служебных. Например, новизны. 
           baseHromosomRatingValues  - рейтинг целеврой функции. Содержит оценку хромосомыс точки зрения пользователя. Нужен для определения результата, 
                                       а также отслеживания хода работы
                                       
           debug - отладка.                             
                128 - признак запрета параллельной работы. Потоки вызываются последовательно. Позволит проверить логику 
                      потоков, за исключением возможных коллизий
                      
                1 - визуализация каждый шаг      

    '''


    def __init__(self, HromosomLen, GenGroupSize=1, FixedGroupsLeft=0, StopFlag=0, TheBestListSize=100,
                 PopulationSize=10000, Clasters=10, kNovelty=(8, 0.15, 0.75), thCount = 6, debug = 128):

        self.onErrorClastering = True
        self.ClasteringStep = 0
        self.ThCount = thCount - 1 # сколько надо создавать новых потоков
        self.TheBestListSize = TheBestListSize  # the number of storing best hromosoms
        self.HromosomLen = HromosomLen
        self.useNovelty = kNovelty[0] > 0

        self.minKNovelty = kNovelty[1]
        self.MaxKNovelty = kNovelty[2]

        self.Threads = [0]* (thCount-1)
        self.ThRes = [0]*thCount

        # добиваемся того, чтобы размерности были кратны потокам
        add = TheBestListSize % thCount
        if add > 0:
            self.TheBestListSize += (thCount - add)

        add = PopulationSize % thCount

        self.PopulationSize = PopulationSize  # постоянное количество хромосом

        self.FreePopulations = 4
        if add > 0:
            self.PopulationSize += (thCount - add)

        self.PopulationSize2 = PopulationSize * 4
        self.kNovelty = kNovelty[0]

        '''self.CrBestSize = (self.TheBestListSize + self.NoveltyCnt) if self.useNovelty else self.TheBestListSize
        self.PopulationsInThread = self.PopulationSize // thCount

        
        self.BestListsInThread = self.CrBestSize // thCount
        self.Population2 = (self.PopulationSize - self.CrBestSize)
        self.PopulationsInThread2 = (self.PopulationSize - self.CrBestSize) // thCount  # распределение по потокам
        self.GensInThread = HromosomLen * self.PopulationsInThread
        self.GensInThread2 = HromosomLen * self.PopulationsInThread2
        # служебные поля. Служат для подготовки индексов родителей
        self.pearentX = np.zeros( self.Population2, np.int32)
        self.pearentY = np.zeros( self.Population2, np.int32)
        self.pearentIndex = np.zeros(self.Population2 * HromosomLen, np.int32)
        '''#self.Mutates = np.zeros(HromosomLen*PopulationSize, np.int32) # хранилище для индексов мутируемых хромосом. Размер
                                                 # задан с запасом. Используемый размер определяется с помощью PMutation
        #self.MutationsValues = np.zeros(HromosomLen*PopulationSize, np.int8) #

        self.Debug = debug
        self.GenCount = HromosomLen * self.PopulationSize
               # участков хромосом

        self.avgPairingMask = [True] * HromosomLen
        # K = PopulationSize / TheBestListSize
        self.Clasters = Clasters
        self.BestInClaster = TheBestListSize // Clasters

        # The list of the best result for any time



        self.StopFlag = StopFlag
        self.InverseMetric = True  # метрика убывает, т.е. оптимизация на убывание
        self.Metric = 100000000000000000 if self.InverseMetric else 0  # исходное значение метрики - обновляется каждый раз при улучшении
        self.MetricLimit = 1  # При достижении этого зхначения - остановка при StopFlag == 2

        self.UsefullMetric = 100000000000000000 if self.InverseMetric else 0  # исходное значение пользовательской метрики - обновляется каждый раз при улучшении
        self.UsefullBestHr = None  # чтобы не потерять решение, лучшую хромосомы мы запоминаем

        self.FirstStep = True
        self.GenerationsLimit = 100  # Стоп при достижени этого лимита при StopFlag == 1

        self.Generation = 0

        self.PMutation = 0.2  # вероятность мутации хромосомы
        self.PMultiMutation = 0.1  # вероятность того, что мутация затронет не один ген, а несколько
        self.PCrossingover = 0.5  # вероятность кросинговера. Размножение может идти за счет обмена отдельными генами (случайно выбирается, акой
        # от кого из родителей). Либо кросинговером - лева часть хромосомы принадлежит одному родителю,
        # правая второму. Место перегиба - случайное
        self.PairingAttempts = 8  # Сейчас е используется. попытки скрещивания. Скрещивание неудачно, если в результате рождается хромосома, которая не проходит проверки.
        self.MinPMutations = 0.1  # вероятность мутаций по пиле меняется со временем от минимума к максимому
        self.MaxPMutations = 0.8
        self.RecreatePopulation = False  # пока не используется
        self.StrongMutations = False  # пока не используется
        self.PShuffle = 0.1  # вероятность ( от всех мутаций) мутации перемешиванием. То есть сохраняем позицию помещений или построек, но меняем саму постройку
        # эффективно в планировке участков
        self.GenGroupSize = GenGroupSize  # размер одного гена
        # self.HromosomLen = HromosomLen  # В хромосому добавляем два флага. 1 - признак измененности. 2 - ссылка на доп. данные

        self.FixedGroupsLeft = FixedGroupsLeft  # часть хромосомы. не участвующая в алгоритме

        self.ReportStep = 1  # поколения, после которых выводим отчет

        # self.HrList = {}

        self.StoredPath = 'copy/copy.dat'  # механизм сохранения не отлажен и наверняка устарел
        self.StorePeriod = 10000 # не сохраняем по умолчанию

        self.TryLoadOnStart = False  # При старте проверка сохраненной версии
        self.VisualizationStep = 100  # Как часто вызывать визуализацию - метод отображения хода проесса

        self.RecreatePopulation = True  # Не используется.0 - пересоздание половины хромосом

        # Механизм новизны. Идея в том, что добавляем штраф за неоригинальность. Это не только защищает алгоритм от вырождения ( условия в популяции достаточно инцестуальные),
        # но и позволяет решать частично проблему застревания в седловых точках
        # Применяю KNN
        self.stranges = np.empty((0, HromosomLen))

        self.isStranger = 2  # на сколько дистанция у хромосомы до ближайших соседей должна быть больше средней, чтобы ее сохранять как "странную"
        self.Neighbors = 5  #

        self.finalMetric = 0  # лучшая метрика с учетом штрафов ( новизна и то, что придумаем в будущем)
        self.avgMetric = 0  # среднеее значение метрики по популяции. Для отладки

        self.DebugInfo = False

        self.Weights = [1]  # не используется
        self.DWeights = [0]

        self.PDeath = 0.75

        self.isStranger = 1.8

        self.PMutation = 0.05  # probability of the mutation for any individuals. Or, if >=1, the number of individuals that will die in each generation
        self.PMultiMutation = 0.1  # the probability of an any bit to be mutated, in the case of the mutation has occurred
        # self.PDeath = 0.2 # probability of the death. Or, if >=1, the number of individuals that will die in each generation
        self.PCrossingover = 0
        self.PairingAttempts = 8  # попытки скрещивания. Скрещивание неудачно, если в результате рождается хромосома, которая не проходит проверки.
        self.MinPMutations = 0.05
        self.MaxPMutations = 0.75
        self.MutStepLimit = 10  # сколько должно быть пустых циклов для увеличения вероятности мутаций
        self.RecreatePopulation = False
        self.StrongMutations = False
        self.VisualizationStep = 1
        self.DebugInfo = True
        self.useNovelty = True

        self.killDubles = True
        self.BestForShowingCnt = 10

        self.GenerationsLimit = 5000
        self.MetricLimit = 0.1
        self.useClastering = True
        self.UseGradient = True
        self.GradientEttempts = 1
        self.GradientStep = 1
        self.MutationStopFlag = self.PMutation + 0.05


    def Threading(self, Fun, array0, array1, args = []):
        #global thSelf
        #thSelf = self
        th = [0]*self.ThCount
        if self.Debug & 128 == 0:
            for thIndex in range(self.ThCount):
                arguments = (self, thIndex, array0, array1, *args)

                #th = threading.Thread( target = Fun, args = arguments)
                th[thIndex] = Process(target = Fun, args = arguments)
                th[thIndex].start()
                #self.Threads[thIndex] = th[thIndex]

            Fun(self, thIndex+ 1, array0, array1, *args)

            for t in th[:thIndex+ 1]:
                t.join()

            a = 0
        else:
            for thIndex in range(self.ThCount+1):
                arguments = (self, thIndex, array0, array1, *args)

                Fun(*arguments)

    def ForEachGene(self, Fn, hr):
        for i in range(self.FixedGroupsLeft, len(hr), self.GenGroupSize):
            Fn[hr[i:i + self.GenGroupSize]]

    def CalcNovelty(self):
        # Инициализируем модель k-NN
        '''if len(self.stranges) > 0:
            hallHromosoms = np.concatenate([self.Hromosoms, self.stranges])
        else:'''

        hallHromosoms = self.Hromosoms

        #hallHromosoms = self.Hromosoms
        knn = NearestNeighbors(n_neighbors=min(self.Neighbors, len(hallHromosoms) - 1))

        knn.fit(hallHromosoms)

        # Находим k ближайших соседей для каждого вектора
        distances, indices = knn.kneighbors(hallHromosoms)

        sortIndexes = np.argsort(distances.sum(1))
        sortIndexes = sortIndexes[sortIndexes < len(self.Hromosoms)][-self.NoveltyCnt:]

        #self.stranges = np.concatenate([self.Hromosoms[sortIndexes], self.stranges])

        #self.stranges = np.unique(self.stranges, return_index=False, axis=0)

        return sortIndexes

    # При сохранении метод может передать дополнительный список значений для сохранения
    # Как их интерпретировать, определить надо в LoadFromStorer
    def AddToStored(self):
        return ()

    def LoadFromStorer(self, StoredList):
        pass

    def Save(self):
        StoreList = [
            # self.TheBestList,
            # self.TheBestValues,  # The list of the best results
            self.Hromosoms,
            self.HromosomRatingValues,
            self.Metric,
            self.FirstStep,
            self.Generation]

        AddList = self.AddToStored()
        StoreList.extend(AddList)

        with open(fr'{self.StoredPath}', 'wb') as FileVar:
            pickle.dump(StoreList, FileVar)

    def Load(self):
        try:
            with open(self.StoredPath, 'rb') as FileVar:
                StoreList = pickle.load(FileVar)

            # self.TheBestList = StoreList[0]
            # self.TheBestValues = StoreList[1]
            self.Hromosoms = StoreList[0]
            self.HromosomRatingValues = StoreList[1]
            self.Metric = StoreList[2]
            self.FirstStep = StoreList[3]
            self.Generation = StoreList[4]
            self.LoadFromStorer(StoreList[5:])
            return True
        except:
            return False

    def CorrectStep(self, step2, crHr):
        return step2

    def GradientOptimisation(self, Hr, CrRaiting, i):
        Ettempts = self.GetGradientEttempts(i)
        raiting = CrRaiting

        needOk = False

        if Ettempts < 0:
            Ettempts = -Ettempts
            needOk = True


        crHr = Hr.copy()
        OK = True
        while OK:  # цикл, пока удается снижаься
            OK = False
            for i in range(Ettempts):  # количество шагов подбора направлений градиента
                step2 = np.random.randint(-1, 2,
                                          self.HromosomLen) * self.GradientStep  # случайный шаг. По любому измерения +-1

                step = self.CorrectStep(step2, crHr)

                newHr = crHr + step
                np.ma.clip(newHr, 0, 255, out=newHr)
                newRaiting = self.TestHromosom(newHr)

                if newRaiting > raiting:
                    step = step * -1

                    step = self.CorrectStep(step, crHr)

                    newHr = crHr + step
                    np.ma.clip(newHr, 0, 255, out=newHr)
                    newRaiting = self.TestHromosom(newHr)

                if newRaiting < raiting:  # если результат лучше прошлого
                    crHr = newHr

                    step = self.CorrectStep(step, crHr)

                    newHr = crHr + step
                    np.ma.clip(newHr, 0, 255, out=newHr)
                    d = raiting - newRaiting
                    raiting = newRaiting

                    # дальше в цикле идем туда же до тех пор, пока приращение.
                    newRaiting = self.TestHromosom(newHr)

                    d1 = raiting - newRaiting
                    while d1 > 0:
                        raiting = newRaiting
                        crHr = newHr

                        # if d > d1: # при этом если приращение меньше, то выход
                        #    break

                        step = self.CorrectStep(step, crHr)
                        newHr = crHr + step
                        np.ma.clip(newHr, 0, 255, out=newHr)
                        d = d1
                        newRaiting = self.TestHromosom(newHr)
                        d1 = raiting - newRaiting

                    if not needOk:
                        OK = True

                    break

        return raiting, crHr

    def Stop(self):
        Res = self.Metric == 0
        if self.StopFlag & 1 == 1:
            Res = self.Generation >= self.GenerationsLimit
        if self.StopFlag & 2 == 4:
            Res = Res or self.PMutation >= self.MutationStopFlag

        return Res or (self.StopFlag & 128 == 128)  # never stop

    def ExtReport(self): # виртуальный метод - можно что-то добавить в хвост типовому сообщению
        return None

    # Выводит сообщение. Последний параметр выводит сообщение для конкретной хромосомы (для отладки)
    # Выводит без сдвига строки ( если не указана хромосома)
    def Report(self, g, m, PMut, redraw=False, Hr=None, time = 0, ext = ''):
        if redraw:
            print('')

        if Hr is not None:
            print(
                f'Поколение {g:5} тестов {self.Tests:5}: {m:5}   PMut = {PMut:2}/{self.kNovelty}  Hr {Hr} {time:5} {ext}')
        else:
            print(f'Поколение {g:5} тестов {self.Tests:5}: {m:5}  PMut = {PMut:2}/{self.kNovelty} {time:5} {ext}',
                  end='\r')

    # виртуальный метод - отработка визуализации процесса - вывод в файл или видео
    def Visualisation(self, hr, ind, metric, argRating):
        pass

    # Пока не использую.
    def Clastering(self, hr, argRating):
        kmeans = KMeans(init="k-means++", n_clusters=self.Clasters, n_init=10)#algorithm{“lloyd”, “elkan
        kmeans.fit(hr)

        inClaster = self.TheBestListSize // self.Clasters

        if self.TheBestListSize % self.Clasters:
            inClaster+= 1

        '''
                    теперь строим новый рейтинг:
        '''
        # для каждого элемента рейтинга находим кластер/ То есть кластеры в порядке argRating
        clasterIndexes = kmeans.labels_[argRating]

        limit = self.TheBestListSize
        clasterIndexes0 = clasterIndexes[:limit]
        #unique, inverse, count = np.unique(clasterIndexes0, return_inverse=True, return_counts=True)
        unique, count = np.unique(clasterIndexes0, return_inverse=False, return_counts=True)
          # позиции в индексе рейтинга, которые надо удалить

        if len(count) < self.Clasters:
            #надо добавить пропущенные позиции
            lst = np.zeros(self.Clasters)
            lst[unique] = count
            count = lst

            unique = np.arange(self.Clasters)

        more = count > inClaster
        mustDel = unique[more] # номера кластеров, из которых надо убрать элементы
        delCount = count[more] - inClaster # сколько надо удалить

        posDelIndex = np.in1d(clasterIndexes0, unique[count > inClaster])
        posDel = np.nonzero(posDelIndex)[0] # позиции в рейтинге которые надо удалить
        #posDelClasters = inverse[posDelIndex] # соответствующие им кластеры

        insIndexes = count < inClaster
        mustInsert = unique[insIndexes] # номера кластеров, в которые надо добавить элементы

        insertCount = inClaster - count
        insertFrom = np.array(np.nonzero(np.in1d(clasterIndexes, mustInsert)[limit:])[0], dtype = np.int32) + limit #индексы позиций, из которых берем недостающее

        if len(delCount) > 0:
            delIndex = len(posDel) - 1
            insIndex = 0
            cnt = delCount[0]
            delCountIndex = 0
            for inRaiting, claster in zip(argRating[insertFrom], clasterIndexes[insertFrom] ):
                count_ = insertCount[claster]
                if count_ > 0:

                    insertCount[claster]-= 1

                    try:
                        pos = posDel[delIndex]
                        delIndex-= 1
                    except:
                        a = 0

                    argRating[pos] = inRaiting

                    cnt-= 1

                    if cnt == 0:
                        delCountIndex+= 1

                        if delCountIndex >= len(delCount):
                            break

                        cnt = delCount[delCountIndex]

        return argRating

    def GradientStrategy(self, Generation, Metric, newMetric):
        pass


    # Главный метод запуска алгоритма
    def Start(self, fileName=None, ):

        # метод создает первоначальную популяцию
        time0 = time.time()
        # Если передан файл хромосомы, мы ее просчитываем - для возможности отладки
        # Надо бы это делать вне виртуального TestHromosom. Но TestHromosom - самый часто вызываемый метод. И усложнять
        # его не хочется по соображением ооптимизации по времени
        self.Tests = 0
        if fileName is not None: # отладочный просчет хромосомы из файла
            hr = np.load(fileName)
            self.DbgHr = hr
            while True:
                self.BestForShowingCnt = 1
                res = self.TestHromosom(self.DbgHr)
                self.bestHromosom = self.DbgHr
                self.BaseHromosomRatingValues = [res]
                self.Hromosoms = np.array([self.DbgHr])
                self.argRating = np.array([0])
                self.Visualisation(self.Hromosoms, 0, res, self.argRating)
                print(res)
            return

        # создаем первоначальную популяцию
        #self.Hromosoms = np.empty( (self.PopulationSize, self.HromosomLen), dtype = np.uint8) # для другого типа генов надо создать новый файл
                               # BaseGenetic. Криво, но это вопрос быстродействия.
        self.BaseHromosomRatingValues = np.empty(self.PopulationSize, dtype = np.float32)

        time0 = time.time()
        BaseHromosomRatingValues = Array('f', len(self.BaseHromosomRatingValues), lock = False)
        Hromosoms = Array('i', self.GenCount, lock=False)
        self.Threading(CreateFirstPopulation, BaseHromosomRatingValues, Hromosoms) # многопоточное создание популяции и ее просчет

        self.Hromosoms = np.array(Hromosoms, dtype=np.uint8).reshape((self.PopulationSize, self.HromosomLen))
        self.BaseHromosomRatingValues = np.array(BaseHromosomRatingValues, dtype=np.float32)
        r = self.TestHromosom(self.Hromosoms[0])

        '''# первоначальный отбор - сортируем и оставляем лучших
        argRating = np.argsort(self.BaseHromosomRatingValues)
        self.Hromosoms = self.Hromosoms[argRating]

        if self.useClastering:
            argRating = self.Clastering(argRating)
            self.Hromosoms = self.Hromosoms[argRating]

        self.BaseHromosomRatingValues = self.BaseHromosomRatingValues[argRating]

        # запоминаем лучшие. Инициализация статистики
        self.Metric = self.BaseHromosomRatingValues[0]
        self.bestHromosom = self.Hromosoms[0]'''

        hear = 0  # счетчик поколений без изменений
        vasReplacing = False
        lastVisMetric = -1 # метрика предыдущей визуализации. Если метка не поменялась, мы не производим новой визуализации

        time1 = time.time()
        print(time1 - time0)

        mutates = np.array([])
        # основной цикл алгоритма
        while not self.Stop():


            # расчет хромосом, если надо - градиентный спуск

            self.BaseHromosomRatingValues = np.resize(self.BaseHromosomRatingValues, len(self.Hromosoms))
            #self.Threading(TestHromosomsTh, [mutates])
            TestHromosomsTh(self, 0, mutates)

            argRating = np.argsort(self.BaseHromosomRatingValues)

            if self.useClastering:

                argRating = self.Clastering(self.Hromosoms[argRating], argRating)[0:self.CrBestSize]

                if self.onErrorClastering or self.ClasteringStep > 0:
                    self.useClastering = False
            else:
                argRating = argRating[0:self.CrBestSize]
            # Сортировка и новизна, если надо
            if self.useNovelty:

                try:
                    argRating[self.TheBestListSize:] = self.CalcNovelty()
                except:
                    a = 0
            else:
                limit = self.TheBestListSize
                #argRating = np.argsort(self.BaseHromosomRatingValues)[0:self.CrBestSize]

            self.BaseHromosomRatingValues = self.BaseHromosomRatingValues[argRating]

            # анализ результата и вывод сообщений
            crMetric = self.BaseHromosomRatingValues[0]
            if crMetric < self.Metric:
                self.Metric = crMetric
                self.bestHromosom = self.Hromosoms[argRating[0]]

                hear = 0
                newMetric = True
            else:
                hear += 1  # счетчик поколений без изменений
                newMetric = False

            # можно поиграть со стратегией расчета градиента в зависимости от метрики или генерации
            if self.UseGradient:
                self.GradientStrategy(self.Generation, self.Metric, newMetric)

            if self.ClasteringStep > 0 and self.Generation % self.ClasteringStep == 0:
                self.useClastering = True

                # надо ли выводить отчет
            if self.Generation % self.ReportStep == 0 or newMetric:
                self.Report(self.Generation, self.Metric, self.PMutation, newMetric, time = time.time() - time0,
                            ext = self.ExtReport())

                if self.Generation % self.VisualizationStep == 0 and lastVisMetric != self.Metric or self.Debug % 2:  # self.VisualizationStep - 1:

                    self.Visualisation(self.Hromosoms, self.Generation, self.Metric, argRating)

                    lastVisMetric = self.Metric

            parentX = self.Hromosoms[argRating[self.pearentX]].ravel()
            parentY = self.Hromosoms[argRating[self.pearentY]].ravel()
            hromosoms1 = np.where(self.pearentIndex, parentX, parentY)


            if self.PopulationSize2 > self.PopulationSize:
            #self.Hromosoms = np.concatenate
                hromosoms1 = hromosoms1.reshape(self.Population2, self.HromosomLen)

                d = int((len(hromosoms1) - self.Population2) / self.FreePopulations) + 1
                newSize = len(hromosoms1)

                while newSize > self.PopulationWOBestSize:
                    argParentX = np.arange(0, newSize, dtype = np.int32)
                    argParentY = argParentX.copy()

                    newSize = max(newSize - d, self.PopulationWOBestSize)

                    np.random.shuffle(argParentX)
                    np.random.shuffle(argParentY)

                    argParentX, argParentY = argParentX[:newSize], argParentY[:newSize]

                    hromosoms1 = np.where(np.random.randint(0, 2, newSize * self.HromosomLen, np.uint8),
                             hromosoms1[argParentX].ravel(), hromosoms1[argParentY].ravel())

                    hromosoms1 = hromosoms1.reshape(newSize, self.HromosomLen)

            hromosoms1 = np.append(self.Hromosoms[argRating].ravel(), hromosoms1)

            mutates = np.random.randint(self.TheBestListSize * self.HromosomLen, self.GenCount, int(self.GenCount * self.PMutation), np.int32)

            H, indexes = np.unique(self.Hromosoms, return_index=True, axis=0)
            if len(H) != len(self.Hromosoms):
                neg = np.arange(0, len(self.Hromosoms))
                neg[indexes] = -1
                neg = neg[neg != -1]
                neg = neg * self.HromosomLen + np.random.randint(0, self.HromosomLen, len(neg))
                mutates = np.append(mutates, neg)
            MutateValues = np.random.randint(0, 256, len(mutates))

            try:
                hromosoms1[mutates] = MutateValues
            except:
                a = 0
            self.Hromosoms = hromosoms1.reshape( (self.PopulationSize, self.HromosomLen))


            # Действия, если давно не было изменений (пробуем увеличить вероятность мутаций)
            if hear % self.MutStepLimit == self.MutStepLimit - 1:  # and not vasReplacing:
                if self.PMutation < self.MaxPMutations:
                    self.PMutation += 0.02
                else:
                    self.PMutation = self.MinPMutations

                if self.onErrorClastering:

                    self.useClastering = True

                '''if self.kNovelty < self.MaxKNovelty:
                    self.kNovelty += self.minKNovelty
                else:
                    self.kNovelty = self.minKNovelty'''

                hear = 0
                vasReplacing = True

            self.Generation += 1  # Счетчик поколений

            # аварийное сохранение
            if self.StorePeriod == 1 or self.Generation % self.StorePeriod == self.StorePeriod - 1:
                self.Save()  # сохраняем список хромосом, BestList и BestValues

        print('')
        # self.Report(self.Generation, self.Metric, self.UsefullMetric, self.avgMetric, self.PMutation, True,
        #            self.ExtReport())
        self.Visualisation(self.Hromosoms, self.Generation, self.Metric, argRating)

    def GetGradientEttempts(self, i):
        return self.GradientEttempts
    @property
    def kNovelty(self):
        return self._kNovelty

    # сеттер для свойства full_name
    @kNovelty.setter
    def kNovelty(self, new):
        self._kNovelty = new

        self.NoveltyCnt = int(self.TheBestListSize * new)

        thCount = self.ThCount+1
        add = self.NoveltyCnt % (thCount)
        if add > 0:
            self.NoveltyCnt += (thCount - add)

        self.CrBestSize = (self.TheBestListSize + self.NoveltyCnt) if self.useNovelty else self.TheBestListSize
        self.PopulationsInThread = self.PopulationSize // thCount

        self.BestListsInThread = self.CrBestSize // thCount
        self.BestListsInThread1 = self.TheBestListSize // thCount
        self.PopulationWOBestSize = (self.PopulationSize - self.CrBestSize)
        self.Population2 = (self.PopulationSize2 - self.CrBestSize)
        self.PopulationsInThread2 = (self.PopulationSize2 - self.CrBestSize) // thCount  # распределение по потокам
        self.PopulationsInThread3 = (self.PopulationSize - self.TheBestListSize) // thCount
        self.GensInThread = self.HromosomLen * self.PopulationsInThread
        self.GensInThread2 = self.HromosomLen * self.PopulationsInThread2
        # служебные поля. Служат для подготовки индексов родителей
        self.pearentX = np.zeros(self.Population2, np.int32)
        self.pearentY = np.zeros(self.Population2, np.int32)
        self.pearentIndex = np.zeros(self.Population2 * self.HromosomLen, np.int32)

