import discord
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import pandas as pd
import tflearn
import tensorflow
import json
import random
import pickle
from sklearn.linear_model import LinearRegression

llave = "NzQ5NjU0NDU1NzM3Mzg0OTkx.X0vIHg.As6-YpoaRP5ckRmcXLDY-xczmf8"


with open("contenido.json", encoding='utf-8') as archivo:
    datos = json.load(archivo)
############Modelo Prediccion Dietas
df = pd.read_csv('berry_diet.csv')
sexo = df['Sexo'].values
objetivo = df['Objetivo'].values
altura = df['Altura'].values
peso = df['Peso'].values
edad = df['edad'].values
calc = df['calculos'].values
actividad = df['actividad_fisica'].values
cena = df['Cenas_tarde'].values
azucar = df['Azucar'].values
refresco = df['Refresco'].values
sal = df['Sal'].values
menu = df['Menu asignado'].values
X = np.array([sexo, objetivo, altura, peso, edad, calc, actividad, cena, azucar, refresco, sal]).T
Y = np.array(menu)
reg = LinearRegression()
reg = reg.fit(X,Y)

palabras = []
tags = []
auxX = []
auxY = []
for contenido in datos["contenido"]:
    for patrones in contenido["patrones"]:
        auxPalabra = nltk.word_tokenize(patrones)
        palabras.extend(auxPalabra)
        auxX.append(auxPalabra)
        auxY.append(contenido["tag"])
        if contenido["tag"] not in tags:
            tags.append(contenido["tag"])

palabras = [stemmer.stem(w.lower()) for w in palabras if w != "?"]
palabras = sorted(list(set(palabras)))
tags = sorted(tags)

entrenamiento = []
salida = []
salidaVacia = [0 for _ in range(len(tags))]

for x, documento in enumerate(auxX):
    cubeta = []
    auxPalabra = [stemmer.stem(w.lower()) for w in documento]
    for w in palabras:
        if w in auxPalabra:
            cubeta.append(1)
        else:
            cubeta.append(0)
    filaSalida = salidaVacia[:]
    filaSalida[tags.index((auxY[x]))] = 1
    entrenamiento.append(cubeta)
    salida.append(filaSalida)

entrenamiento = np.array(entrenamiento)
salida = np.array(salida)

tensorflow.reset_default_graph()

red = tflearn.input_data(shape=[None, len(entrenamiento[0])])
red = tflearn.fully_connected(red, 10)
red = tflearn.fully_connected(red, 10)
red = tflearn.fully_connected(red, len(salida[0]), activation = "softmax")
red = tflearn.regression(red)

modelo = tflearn.DNN(red)
modelo.fit(entrenamiento, salida, n_epoch= 1000, batch_size=10, show_metric = True)
modelo.save("modelo.tflearn")

def mainBot():
    while True:
            entrada = input("Tu: ")
            cubeta = [0 for _ in range(len(palabras))]
            entradaProcesada = nltk.word_tokenize(entrada)
            entradaProcesada = [stemmer.stem(palabra.lower()) for palabra in entradaProcesada]
            for palabraIndividual in entradaProcesada:
                for i, palabra in enumerate(palabras):
                    if palabra == palabraIndividual:
                        cubeta[i] = 1
            resultados = modelo.predict([np.array(cubeta)])
            resultadosIndices = np.argmax(resultados)
            tag = tags[resultadosIndices]

            for tagAux in datos["contenido"]:
                if tagAux["tag"] == tag:
                    respuesta = tagAux["respuestas"]
            if(entrada !="dieta"):
                print("Berry: ",random.choice(respuesta))
            else:
                print("Te pediremos algunos datps para darte la mejor dieta. Te pedimos que ingreses el número de la opcion deseada en cada caso.")
                sexo = int(input("¿Cuál es tu sexo?\n   1: Masculino\n   2: Femenino\n"))
                objetivo = int(input("¿Cuál es tu objetivo?\n   1: Perder peso\n   2: Ganar peso\n   3: Tener buenos habitos\n"))
                altura = int(input("Ingresa tu estatura en centimetros, es decir, si mides 1.76m ingresa unicamente el numero 176\n"))
                peso = int(input("Ingresa tu peso en kilogramos, ejemplo 80.4\n"))
                edad = int(input("¿Cuál es tu edad?\n"))
                calc = altura - 100 - peso
                actividad = int(input("¿Cuál es tu nivel de actividad física?\n   1: Alta(Me ejercito al menos 5 veces por semana)\n   2: Mediana(Me ejercito al menos 3 veces por semana)\n  "
                                      " 3: Baja(Me ejercito 1 vez por semana)\n   4: Nula(No hago ejercicio)\n"))
                cena = int(input("¿Sueles cenar despues de las 9pm?\n   1: Si\n   2: No\n"))
                azucar = int(input("¿Tienes dificultad para dejar de comer alimentos con azúcar?\n   1: Si\n   2: No\n"))
                refresco = int(input("¿Tomas mucho refresco?\n   1: Si\n   2: No\n"))
                sal = int(input("¿Puedes comer sal?\n   1: Si\n   2:No\n"))
                prediccion = round((reg.predict(
                    [[sexo, objetivo, altura, peso, edad, calc, actividad, cena, azucar, refresco, sal]])).item(0))
                print("La mejor dieta para ti es la dieta ", prediccion)



mainBot()