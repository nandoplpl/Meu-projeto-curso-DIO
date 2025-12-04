# Meu-projeto-curso-DIO
machine Learning

- treinamento para reconhecer gatos e cachorros


Descrição Detalhada

- Este projeto realiza um treinamento atraves do colab para identificar a imagem, se é uma cachorro ou um gato.


- Este repositório foi criado como parte de um desafio técnico do curso DIO, com o objetivo de demonstrar minha habilidade com machine learning.

---

Tecnologias Utilizadas

- Python
- GitHub  
- Colab  

---

# Estrutura:

Instalar e importar bibliotecas:

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras import layers, models



Carregar o dataset Cachorros e gatos:
dataset, info = tfds.load("cats_vs_dogs", with_info=True, as_supervised=True)
train = dataset["train"]



Preparar as imagens (aqui ele vai ajustar as imagens):
IMG_SIZE = 160
BATCH = 32

def preparar(img, label):
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    return img, label

train = train.map(preparar).batch(BATCH)



Transfer Learning (aqui vai carregar aquivos pre- treinados):
base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False




Criar o modelo final( configuraçoes padroes):
modelo = models.Sequential( [ base, layers.GlobalAveragePooling2D(), layers.Dense(1, activation='sigmoid')])
modelo.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


Treinamentos ( aqui é a quantidade de epocas que ele ira trinar):
modelo.fit(train, epochs=3)


Aqui é um teste com uma imagem(Carregue uma imagem no Colab para fazer este teste):

from PIL import Image

img = Image.open("/content/coloque_o_nome_da_imagem_aqui.jpg").resize((IMG_SIZE, IMG_SIZE))
img = np.array(img) / 255.0
img = np.expand_dims(img, axis=0)

pred = modelo.predict(img)[0]

print("Cachorro 🐶" if pred > 0.5 else "Gato 🐱")
