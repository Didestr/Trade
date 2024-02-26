import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

data = pd.read_csv("EURRUB.csv", sep=';') # Импортируем данные

# Рисуем график
plt.style.use(style='ggplot')
plt.figure(figsize=(16,10))
plt.plot(data['open'])
plt.xlabel('Дни')
plt.ylabel('Цена')
plt.legend([f'Курс евро'])
plt.show()

# Нормализуем данные и создаем обучающую выборку
scaler = MinMaxScaler()
data['scaled_open'] = scaler.fit_transform(np.expand_dims(data['open'].values, axis=1))
data['scaled_close'] = scaler.fit_transform(np.expand_dims(data['close'].values, axis=1))

train_dataset = data.sample(frac=0.8,random_state=0)
test_dataset = data.drop(train_dataset.index)

x_train = train_dataset['scaled_open'].tolist()
y_train = train_dataset['close'].tolist()

x_test = test_dataset['scaled_open'].tolist()
y_test = test_dataset['close'].tolist()

# Создаем модель нейронной сети
model = keras.Sequential([
    keras.layers.Flatten(input_shape=[1]),
    keras.layers.Dense(4, activation='linear'),
    keras.layers.Dense(1, activation='linear')
  ])

keras.optimizers.Adam(
    learning_rate=0.03)

model.compile(loss='mse',
              optimizer='adam')

# Обучаем нейросеть
model.fit(x_train, y_train,
            batch_size=2,
            epochs=30)

# Тестируем нейросеть
model.evaluate(x_test, y_test, batch_size=1)

# прогнозируем курс евро на обученной модели со смещением одного дня вперед
predicted_all = model.predict(data['scaled_close'], batch_size=1)
predicted_all = np.insert(predicted_all, 0, predicted_all[0])

#рисуем график прогноза
plt.style.use(style='ggplot')
plt.figure(figsize=(16,10))
plt.title(label=f'Последняя предсказанная цена {float(predicted_all[-1])}')
plt.plot(data['open'])
plt.plot(predicted_all)
plt.xlabel('Дни')
plt.ylabel('Цена')
plt.legend([f'Курс евро',
            'Предсказанный нейросетью курс евро'])
plt.show()