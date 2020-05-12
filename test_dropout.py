# Testen von Dropout Raten von 0 bis 90 Prozent in 10 Prozent Schritten mit 80 Epochen
# Quelldatei ist leaf_cnn_07.py
# Das Protokoll wird gespeichert in backup/settings_doc_1.txt 

import os

#Initialisieren und Deklarieren der Variablen
dropout_10 = 0
dropout = 0
length = 0

with open('leaf_cnn_7.py', 'r') as file: # Zaehlen der Zeilen in der Quelldatei
	for line in file:
		length = length + 1

for dropout_10 in range(10): # Schleife mit Intervall von 0 bis 9
    dropout = dropout_10 * 0.1  # Initialisieren der Dropout Rate von Integer zu float Dezimalzahl
    with open('leaf_cnn_7.py', 'r') as file: # Neuoeffnen der Quelldatei
        with open('leaf_cnn_7_tmp.py', 'w') as file_2: # Schreiben einer temporaeren Zieldatei 
            line_v = 0
            for line in file: 
                if line_v < length-5:
                    file_2.write(line)
                elif line_v < length-4: # Austauschen der fuer die Dropout Rate verantworlichen Zeilen
                    file_2.write('dropout = ' + str(dropout) + '\n\n# Initialisieren der Platzhalter und Starten des Codes \nX_train, X_test, y_train, y_test = create_dataset(data_dir, base_pixels) \ntrain_model(X_train, X_test, y_train, y_test, dropout, epochs, base_pixels)')
                else:
                    pass
                line_v = line_v + 1
    os.system('python leaf_cnn_7_tmp.py') # Ausfuehren der temporaeren Zieldatei