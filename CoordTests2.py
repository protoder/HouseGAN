import numpy as np
from shapely.geometry import Polygon, LineString


def generate_dataset(N):
    dataset = []
    pos = 0
    neg = 0

    while pos + neg < N:
        # Генерируем случайные координаты для первого четырехугольника
        poly1_coords = np.random.randint(0, 1000, size=(2, 4, 2))
        # Создаем объект Polygon для первого четырехугольника
        poly1 = Polygon(poly1_coords[0])
        poly2 = Polygon(poly1_coords[1])

        # Проверяем пересечение двух четырехугольников
        if poly1.intersects(poly2):
            OK = pos < N//2

            if OK:
                pos+= 1
                intersect = 1
        else:
            OK = neg < N//2

            if OK:
                neg+= 1
                intersect = 0

        # Находим кратчайшее расстояние между четырехугольниками
        if OK:
            #distance = poly1.centroid.distance(poly2.centroid) * 0.001

            #poly1_coords = poly1_coords / 1000
        # Сохраняем координаты и метки в массив
            data = np.concatenate([poly1_coords.flatten(), np.array(poly1.centroid.coords).flatten(), np.array(poly2.centroid.coords).flatten()])/1000
            dataset.append(data)

    return np.array(dataset)


# Генерируем датасет из 1000 элементов
'''dataset = generate_dataset(10000)
np.save('test2.npy', dataset)
dataset = generate_dataset(50000)
np.save('dataset2.npy', dataset)'''

dataset = np.load('dataset2.npy')
test = np.load('test2.npy')

# Пример первых 5 элементов датасета
#print(dataset[:5])

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Определение нейросети
class Autovar(nn.Module):
    def __init__(self):
        super(Autovar, self).__init__()
        self.fc1 = nn.Linear(8, 128)
        self.do1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 128)
        self.do2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(64, 32)
        self.do3 = nn.Dropout(0.4)
        self.fc4 = nn.Linear(32, 16)
        self.do4 = nn.Dropout(0.4)
        self.fc5 = nn.Linear(16, 8)
        self.do5 = nn.Dropout(0.4)
        self.fc6 = nn.Linear(8, 4)
        self.do6 = nn.Dropout(0.4)
        self.fc7 = nn.Linear(4, 8)
        self.do7 = nn.Dropout(0.4)
        self.fc8 = nn.Linear(8, 16)
        self.do8 = nn.Dropout(0.4)
        self.fc9 = nn.Linear(16, 32)
        self.do9 = nn.Dropout(0.4)
        self.fc10 = nn.Linear(32, 64)
        self.do10 = nn.Dropout(0.4)
        self.fc11 = nn.Linear(64, 128)
        self.do11 = nn.Dropout(0.4)
        self.fc12 = nn.Linear(128, 2)
        self.do12 = nn.Dropout(0.4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x= torch.relu(self.fc2(x))
        '''x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = torch.relu(self.fc9(x))
        x = torch.relu(self.fc10(x))
        x = torch.relu(self.fc11(x))'''
        x = self.fc12(x)
        return x


# Пример обучения нейросети
def train_model(dataset, batch_size, epochs, lr):
    # Создание DataLoader для загрузки данных в виде батчей
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Инициализация модели
    model = Autovar()

    # Определение критерия (бинарная кросс-энтропия)
    criterion = nn.MSELoss()

    # Определение оптимизатора (SGD)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    last_error = 0

    results = []
    # Обучение модели
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            # Получение входных данных и меток из батча
            inputs, labels = batch[0], batch[1]

            # Обнуление градиентов
            optimizer.zero_grad()

            # Прямой проход
            outputs = model(inputs.float())

            # Вычисление потерь
            loss = criterion(outputs, labels)

            # Обратный проход и оптимизация
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        res = np.array(model(torch.tensor(test[:,:8]).float()).detach().cpu())
        error = np.sqrt(((res - test[:,16:18])**2).sum())/len(test)

        d = last_error-error
        last_error = error

        results.append( (epoch_loss / len(dataloader), error) )

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader)}, Test: {error}  D= {d}")

        if epoch % 10 == 9:
            np.save('res2', np.array(results))


# Пример использования
# Преобразуем numpy датасет в TensorDataset
tensor_dataset = TensorDataset(torch.Tensor(dataset[:, :8]), torch.Tensor(dataset[:, 16:18]))

# Параметры обучения
batch_size = 32
epochs = 10000
learning_rate = 0.001

# Обучение модели
train_model(tensor_dataset, batch_size, epochs, learning_rate)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Определение нейросети регрессии
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Пример обучения нейросети регрессии
def train_regression_model(dataset, batch_size, epochs, lr):
    # Создание DataLoader для загрузки данных в виде батчей
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Инициализация модели
    model = RegressionModel()

    # Определение критерия (средняя квадратичная ошибка)
    criterion = nn.MSELoss()

    # Определение оптимизатора (SGD)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Обучение модели
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            # Получение входных данных и меток из батча
            inputs, labels = batch[:, :8], batch[:, 9].unsqueeze(1)

            # Обнуление градиентов
            optimizer.zero_grad()

            # Прямой проход
            outputs = model(inputs.float())

            # Вычисление потерь
            loss = criterion(outputs, labels.float())

            # Обратный проход и оптимизация
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader)}")


# Пример использования
# Преобразуем numpy датасет в TensorDataset
tensor_dataset = TensorDataset(torch.Tensor(dataset[:, :8]), torch.Tensor(dataset[:, 9]))

# Параметры обучения
batch_size = 32
epochs = 10
learning_rate = 0.01

# Обучение модели регрессии
train_regression_model(tensor_dataset, batch_size, epochs, learning_rate)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Определение нейросети Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Определение нейросети Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Определение автокодировщика
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Пример обучения автокодировщика
def train_autoencoder(dataset, batch_size, epochs, lr):
    # Создание DataLoader для загрузки данных в виде батчей
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Инициализация модели Encoder и Decoder
    encoder = Encoder()
    decoder = Decoder()
    autoencoder = Autoencoder(encoder, decoder)

    # Определение критерия (средняя квадратичная ошибка)
    criterion = nn.MSELoss()

    # Определение оптимизатора (SGD)
    optimizer = optim.SGD(autoencoder.parameters(), lr=lr)

    # Обучение автокодировщика
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            # Получение входных данных и меток из батча
            inputs = batch[:, :8]

            # Обнуление градиентов
            optimizer.zero_grad()

            # Прямой проход
            outputs = autoencoder(inputs.float())

            # Вычисление потерь
            loss = criterion(outputs, inputs.float())

            # Обратный проход и оптимизация
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader)}")


# Пример использования
# Преобразуем numpy датасет в TensorDataset
tensor_dataset = TensorDataset(torch.Tensor(dataset[:, :8]))

# Параметры обучения
batch_size = 32
epochs = 10
learning_rate = 0.01

# Обучение автокодировщика
train_autoencoder(tensor_dataset, batch_size, epochs, learning_rate)

