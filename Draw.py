import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных из файла
results = np.load('res2.npy')#[100:]

# Создание графиков
fig, axs = plt.subplots(4, 1, figsize=(10, 15))

# Заголовок всей фигуры
fig.suptitle('Графики трех различных кривых из сохраненных результатов')

# Названия графиков
titles = ['Sum Error', 'Discriminator Loss', 'Gradient Penalty']

# Построение каждого графика
text = ("Gen", "Discr Hall", "Penalty", "Discr Only")
for i in range(2):
    r = results[:, i]

    axs[i].plot(r)
    axs[i].set_title(text[i])
    axs[i].set_xlabel('Iteration')
    axs[i].set_ylabel(text[i])

# Автоматическое расположение графиков, чтобы заголовок не перекрывался
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Сохранение графиков в файл
plt.savefig('result_graphs.png')

# Показать графики
plt.show()
