import tensorflow as tf
import numpy as np

numeros = np.array([1,2,3,4,5], dtype=float)
multiplo_de_dos = np.array([2,4,6,8,10], dtype=float)

entrada = tf.keras.layers.Dense(units=3, input_shape=[1])
neurona1 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([entrada, neurona1, salida])

modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Comenzando entrenamiento...")
historial = modelo.fit(numeros, multiplo_de_dos, epochs=1000, verbose=False)
print("Modelo entrenado!")

print("Hagamos una prediccion de multiplo de dos!")
resultado = modelo.predict([100.0])
print("El resultado es " +str(resultado))

print("Variables internas del modelo")
print(salida.get_weights())