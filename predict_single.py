# Klassifizieren eines einzelnen Bilds auf Basis eines zuvor trainierten Modells

# Importe
import keras
from keras.models import load_model
from keras.models import Model
import numpy as np
import skimage.data
import skimage.transform
#from skimage.color import rgb2gray # Zu verwenden, wenn ein schwarz-weiss Modell und Farbbilder vorliegen

# Variablen initialisieren und deklarieren
img_cat_pred = 0
a = 0
v = 0

# Gespeichertes Modell mit Architektur und trainierten Gewichten und Tendenzen aus Datei laden
model = load_model('models/leaf_cnn_1540935811.hdf5')
model_bp = 64 # Angeben der in dem trainierten Modell verwendeten Dimensionen der Bilder

# Definieren des Bildes
img_cat = 3 # Kategorie des Bildes (wenn bekannt)
file = 'SCAN0141-Bearbeitet' # Dateiname des Bildes
filename = 'scan_clean/' + str(img_cat) + '/' + file + '.jpg'
#filename = 'variation01.jpg' # Alternatives Bild im Ordner des Skripts
print(filename)

# Oeffnen des Bildes
img = skimage.data.imread(filename)

# Anpassen der Dimensionen des Bildes wenn noetig
if img.shape != (model_bp,model_bp,3):
    img = skimage.transform.resize(img, (model_bp, model_bp))
print(img.shape)

# Umwandeln des Bildes in ein Numpy Feld
img = [img]
img = np.array(img)
print(img.shape)

#img = np.reshape(img,[1,320,240,3])
#img = np.expand_dims(img, axis=0)# Noetig fuer schwarz-weiss Modell
#print(img.shape)

# Kategorie ermitteln und angeben
categories = model.predict(img)
print(categories)

# Maximale Aktivierung der Kategorie bestimmen
for a in range(len(categories[0])):
    if categories[0][a]>v:
        print(str(categories[0][a]) + '>' + str(v))
        img_cat_pred = a
        v = categories[0][a]

# Abgleichen der bestimmten Kategorie mit der richtigen Kategorie
if img_cat == img_cat_pred:
    print('The predicted category ' + str(img_cat_pred) + ' is correct.')
else: 
    print('The predicted category ' + str(img_cat_pred) + ' does not match the actual category ' + str(img_cat) + '.')