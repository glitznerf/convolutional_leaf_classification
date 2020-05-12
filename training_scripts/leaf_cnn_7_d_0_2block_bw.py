#Testing architecture and parameters

#IMPORTS
import os
i# Architekturen entwerfen und testen

# Importe
import os
import time
import random
from distutils.version import StrictVersion

import sklearn
from sklearn.model_selection import train_test_split
assert StrictVersion(sklearn.__version__ ) >= StrictVersion('0.18.1')

# Import des Tensorflow-Backend Framework
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
assert StrictVersion(tf.__version__) >= StrictVersion('1.1.0')

# Importe aus dem Keras Framework
import keras
assert StrictVersion(keras.__version__) >= StrictVersion('2.0.0')
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.utils.np_utils import to_categorical
from keras.backend.tensorflow_backend import set_session

# Zusaetzliche Importe fuer die Verarbeitung von Daten und Zahlen
import matplotlib.pylab as plt
import matplotlib.pyplot as pyplot
import numpy as np
import matplotlib
import pandas as pd
assert StrictVersion(pd.__version__) >= StrictVersion('0.19.0')
import skimage.data
import skimage.transform
from skimage.color import rgb2gray


# Protokollierung
def write_log(dropout,epochs,current_timestamp,layers_array,eval_metrics,base_pixels):
    # Speichern von Einstellungen und Ergebniswerten in einer Textdatei für die Evaluation und Dokumentation
    with open('../backup/settings_doc_2.txt', 'a') as file:
        file.write('Leaf recognition model leaf_cnn_7_d_0_2block_bw fitting of ' + str(current_timestamp) + ':') # Zeitpunkt des Speicherns
        file.write('\nDropout: ' + str(dropout) + ', epochs: ' + str(epochs) + ', test_accuracy: ' + str(eval_metrics[-1])) # Kurzinfos
        file.write('\nLayers: ' + str(layers_array) + ', Base pixels:' + str(base_pixels))  # Kurzinfos
        file.write('\nMetrics loss, acc, val_loss, val_acc, test_loss, test_accuracy:\n') # Ausfuehrliche Ergebnisse
        for m in eval_metrics: 
            file.write(str(m) + '\n') 
        file.write('\n')

        
# DATEN
# Bilder laden
def load_data(data_dir, base_pixels):
    # Alle Unterordner des Dateiverzeichnisses werden gesammelt, da sie je eine Kategorie Bilder beinhalten
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Deklarieren der Platzhalter labels und images (Kategorien und Bilder) als arrays (Felder)
    type=".jpg"
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(type)]
        # Die Daten der Unterordner werden iterativ in den Variablen Labels (Kategorien) und Images (Bilder) gespeichert
        for f in file_names:
            image = skimage.data.imread(f)
            image = rgb2gray(image) #Makes the image a black and white image 
            images.append(image)
            labels.append(int(d))
    images_bp = [skimage.transform.resize(image, (base_pixels, base_pixels)) for image in images] # Veraendern der Bildgröße nach den geforderten Dimensionen
    print('Vorverarbeitung fertig!')
    return images_bp, labels
    
    
# Datensatz laden
def create_dataset(data_dir, base_pixels):
    ROOT_PATH = "./"
    original_dir = os.path.join(ROOT_PATH, data_dir)
    images, labels = load_data(original_dir, base_pixels)

    y = np.array(labels)
    X = np.array(images)
    X = np.expand_dims(X, axis=3)	# Nur noetig wenn schwarz-weiss aufgrund fehlender Farbdimension
    num_categories = 10
    y = to_categorical(y, num_categories)

    # Teilen des Datensatzes in Trainingsset und Testset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40, stratify=y)
    #print(X_train.shape, X_test.shape)
    print('Datensatz geschaffen!')
    return X_train, X_test, y_train, y_test


# ARCHITEKTUR
def build_architecture(dropout, base_pixels):
    # Initialisieren des Modells
    model = Sequential()
    
    # Aufbau der Architektur
    # Erster Block CL
    model.add(Conv2D(base_pixels, 3, activation='relu', padding='same', input_shape=(base_pixels, base_pixels, 1)))
    model.add(Conv2D(base_pixels, 3, activation='relu', padding='same'))
    model.add(Conv2D(base_pixels, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Zweiter Block CL
    model.add(Conv2D(base_pixels, 3, activation='relu', padding='same'))
    model.add(Conv2D(base_pixels, 3, activation='relu', padding='same'))
    model.add(Conv2D(base_pixels, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Dimensionsreduzierung und dichte Schichten
    model.add(Flatten())
    model.add(Dense(30, activation='relu'))

    # Schicht mit Aktivierungsfunktion fuer die zehn Kategorien
    model.add(Dense(10, activation='softmax'))
    model.summary()
    
    # Speichern der Schichtenordnung für die Dokumentation
    # Schichten: Conv=0, MaxP=1, Dropout=2, Flatten=3, Dense=4
    layers_array = [0,0,0,1,0,0,0,1,3,4,4]
    
    # Definieren des Optimierungsverfahrens
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model, layers_array


# TRAINING
# Konfigurieren des Tensorflow-Backends
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # Dynamische Veraenderung des verwendeten GPU-Speichers
sess = tf.Session(config=config)
set_session(sess) # Uebernehmen der Tensorflow Einstellungen von Keras


# Trainieren des Modells: Berechnung geeigneter Parameter fuer die verwendete Architektur
def train_model(X_train, X_test, y_train, y_test, dropout, epochs, base_pixels): 
    model,layers_array = build_architecture(dropout, base_pixels)
    BATCH_size = 12 # Vorher: 10, Variabel je nach Rechenleistung der GPU
    
    # Wenn die Ergebnisse lange Zeit schlecht bleiben, kann abgebrochen werden
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=1)
    
    #trained = model.fit(X_train, y_train, epochs=epochs, batch_size=BATCH_size, validation_split=0.1, callbacks=[early_stopping_callback]) # Mit fruehzeitigem Stoppen
    trained = model.fit(X_train, y_train, epochs=epochs, batch_size=BATCH_size, validation_split=0.1) # Ohne fruehzeitiges Stoppen
    print('Training abgeschlossen!')

    # Speichern des Modells inklusive der Architektur und aller berehneter Parameter im hdf5 Format mit aktuellem Zeitstempel
    current_timestamp = int(time.time())
    model.save('../models/leaf_cnn_' + str(current_timestamp) + '.hdf5')


    # EVALUATION
    # Evaluation des Modells und der Paramter
    train_loss, train_accuracy = model.evaluate(X_train, y_train, batch_size=BATCH_size)
    print(train_loss)
    print(train_accuracy)

    test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=BATCH_size)
    print(test_loss)
    print(test_accuracy)
    
    # Zwischenspeichern der Evaluationsergebnisse
    eval_metrics = ['loss','acc','val_loss','val_acc']
    a=0
    for m in eval_metrics:
        m = trained.history[m]
        eval_metrics[a] = m
        a = a + 1
    eval_metrics.append(test_loss)
    eval_metrics.append(test_accuracy)
    
    # Schreiben des Dokumentationseintrags mit Weitergabe der Evaluationsergebnisse
    write_log(dropout,epochs,current_timestamp,layers_array,eval_metrics, base_pixels)


# Deklaration/Initialisierung der Variablen
data_dir = '../scan_clean'
base_pixels = 64
epochs = 200
dropout = 0

# Initialisieren der Platzhalter und starten der Funktionen
X_train, X_test, y_train, y_test = create_dataset(data_dir, base_pixels)
train_model(X_train, X_test, y_train, y_test, dropout, epochs, base_pixels)