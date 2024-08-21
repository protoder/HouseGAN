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
            distance = poly1.distance(poly2) * 0.001

            poly1_coords = poly1_coords / 1000
        # Сохраняем координаты и метки в массив
            data = np.concatenate((poly1_coords.flatten(), [intersect, distance]))
            dataset.append(data)

    return np.array(dataset)


# Генерируем датасет из 1000 элементов
#dataset = generate_dataset(50000)
#dataset = np.save('dataset.npy', dataset)

#dataset = generate_dataset(10000)
#np.save('test.npy', dataset)
dataset = np.load('dataset.npy')
test = np.load('test.npy')

# Пример первых 5 элементов датасета
#print(dataset[:5])

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Определение нейросети
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(16, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


# Пример обучения нейросети
def train_model(dataset, batch_size, epochs, lr):
    # Создание DataLoader для загрузки данных в виде батчей
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Инициализация модели
    model = Classifier()

    # Определение критерия (бинарная кросс-энтропия)
    criterion = nn.BCELoss()

    # Определение оптимизатора (SGD)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Обучение модели
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            # Получение входных данных и меток из батча
            inputs, labels = batch[0], batch[1][:, 0]

            # Обнуление градиентов
            optimizer.zero_grad()

            # Прямой проход
            outputs = model(inputs.float())

            # Вычисление потерь
            loss = criterion(outputs, labels.unsqueeze(1).float())

            # Обратный проход и оптимизация
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        res = np.array(model(torch.tensor(test[:,:16]).float()).detach().cpu())>0.5
        error = (res == test[:,16:17]).sum()/len(test)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader)}, Test: {error}")


# Пример использования
# Преобразуем numpy датасет в TensorDataset
tensor_dataset = TensorDataset(torch.Tensor(dataset[:, :16]), torch.Tensor(dataset[:, 16:]))

# Параметры обучения
batch_size = 32
epochs = 10000
learning_rate = 0.01

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

