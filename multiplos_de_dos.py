import tensorflow as tf
import numpy as np

numeros = np.array([1,2,3,4,5,6,7,8,9,10], dtype=int)
multiplo_de_dos = np.array([2,4,6,8,10,12,14,16,18,20], dtype=int)

capa_entrada = tf.keras.layers.Dense(units=3, input_shape=[1])
capa_oculta = tf.keras.layers.Dense(units=3)
capa_salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([capa_entrada, capa_oculta, capa_oculta, capa_salida])

modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Comenzando entrenamiento...")
historial = modelo.fit(numeros, multiplo_de_dos, epochs=300, verbose=False)
print("Modelo entrenado!")

print("Hagamos una prediccion de multiplo de dos!")
resultado = np.around(modelo.predict([100, 200, 111, 1325]), decimals=1).astype(int)
print("El resultado es " + str(resultado))

print("Variables internas del modelo")
print(capa_entrada.get_weights())