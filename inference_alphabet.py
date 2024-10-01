############## OPENCV ##############
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Funzione per preprocessare l'immagine per la predizione
def preprocess_image(img_path, target_size=(64, 64)):            ############################## target_size=(200, 200) se si usa il modello addestrato a 64px
    img = image.load_img(img_path, target_size=target_size)  # Ridimensiona l'immagine per il modello
    img_array = image.img_to_array(img)  # Converti l'immagine in array NumPy
    img_array = np.expand_dims(img_array, axis=0)  # Aggiungi una dimensione per il batch size
    img_array = img_array / 255.0  # Normalizza l'immagine
    return img_array

# Funzione per fare previsioni
def predict_image_class(img_path, model, target_size=(64, 64)):  ############################## target_size=(200, 200) se si usa il modello addestrato a 64px
    img_array = preprocess_image(img_path, target_size=target_size)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class

# Funzione per scrivere la classe predetta sull'immagine originale usando OpenCV
def draw_prediction_on_image(original_img_path, prediction_text, font_scale=1.5, thickness=2):
    # Carica l'immagine originale usando OpenCV
    img = cv2.imread(original_img_path)
    # Specifica la posizione del testo, font, dimensione del font, colore e spessore
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, 50)  # posizione del testo
    color = (0, 0, 255)  # colore del testo in BGR (qui rosso)
    
    # Scrivi il testo sull'immagine
    cv2.putText(img, prediction_text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return img

# Carica il modello salvato
model = load_model(os.path.join('alphabet_mute_model_adhoc_64px_worked/architecture_weights', 'best_model.keras'))

class_indices = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']
class_labels = {k: v for k, v in enumerate(class_indices)} 

img_path = 'alphabet_immagini_da_utilizzare_come_test_manuale_MAI_VISTE_NEL_TRAINING/A_test.jpg'
predicted_class = predict_image_class(img_path, model)
predicted_label = class_labels[predicted_class[0]]
print(f"Predicted class: {predicted_label}")

# Scrivi la classe predetta sull'immagine originale usando OpenCV
annotated_img = draw_prediction_on_image(img_path, predicted_label)

# Mostra l'immagine con la predizione usando OpenCV
cv2.imshow('Predicted Image', annotated_img)
cv2.waitKey(0)  # Premi un tasto per chiudere la finestra
cv2.destroyAllWindows()



################### PIL #####################

# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import os
# from PIL import Image, ImageDraw, ImageFont

# # Funzione per preprocessare l'immagine per la predizione
# def preprocess_image(img_path, target_size=(64, 64)):
#     img = image.load_img(img_path, target_size=target_size)  # Ridimensiona l'immagine per il modello
#     img_array = image.img_to_array(img)  # Converti l'immagine in array NumPy
#     img_array = np.expand_dims(img_array, axis=0)  # Aggiungi una dimensione per il batch size
#     img_array = img_array / 255.0  # Normalizza l'immagine
#     return img_array

# # Funzione per fare previsioni
# def predict_image_class(img_path, model, target_size=(64, 64)):
#     img_array = preprocess_image(img_path, target_size=target_size)
#     prediction = model.predict(img_array)
#     predicted_class = np.argmax(prediction, axis=1)
#     return predicted_class

# # Funzione per scrivere la classe predetta sull'immagine originale
# def draw_prediction_on_image(original_img_path, prediction_text, font_size=20):
#     img = Image.open(original_img_path)  # Carica l'immagine originale
#     draw = ImageDraw.Draw(img)
#     font = ImageFont.load_default()
#     try:
#         font = ImageFont.truetype("arial.ttf", font_size)
#     except IOError:
#         # Se non hai un font TTF specifico, usa il font predefinito con un font size
#         font = ImageFont.load_default()
#     # Aggiungi il testo all'immagine, in alto a sinistra
#     draw.text((10, 10), prediction_text, font=font, fill=(255, 0, 0))
#     return img

# # Carica il modello salvato
# model = load_model(os.path.join('alphabet_mute_resnet50_worked/architecture_weights', 'best_model.keras'))

# # Ottenere la mappatura delle classi dal generatore
# class_indices = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']
# class_labels = {k: v for k, v in enumerate(class_indices)} 

# # Esempio di utilizzo
# img_path = 'asl_alphabet_test/L_test.jpg'
# predicted_class = predict_image_class(img_path, model)
# predicted_label = class_labels[predicted_class[0]]
# print(f"Predicted class: {predicted_label}")
# # Scrivi la classe predetta sull'immagine originale
# annotated_img = draw_prediction_on_image(img_path, predicted_label)

# # Mostra l'immagine con la predizione
# annotated_img.show()

