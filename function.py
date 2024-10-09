import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import urllib
from PIL import Image

# Descărcarea modelului MobileNetV2 pre-antrenat
model = tf.keras.applications.MobileNetV2(weights='imagenet')


# Funcție pentru încărcarea și prelucrarea imaginii
def load_and_process_image(image_path):
    image = Image.open(image_path)
    image = np.array(image)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

# Funcție pentru recunoașterea obiectelor în imagine și returnarea tuturor predicțiilor
def predict_objects(image_path):
    # Preprocesarea imaginii
    processed_image = load_and_process_image(image_path)

    # Realizarea previziunilor cu ajutorul modelului
    predictions = model.predict(np.expand_dims(processed_image, axis=0))

    # Decodificarea și returnarea tuturor predicțiilor
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]

    return decoded_predictions, processed_image  # Returnați lista de predicții și imaginea procesată


# Funcție pentru afișarea imaginii cu obiectele recunoscute și denumirile acestora
def display_image_with_objects(image, predictions):
    # Afișați imaginea
    plt.imshow(image)

    # Adăugați numele obiectelor recunoscute în afara imaginii
    for i, prediction in enumerate(predictions):
        plt.text(image.shape[1] + 10, 30 * i + 20, f"{prediction[1]} ({prediction[2]:.2f})", color='black', fontsize=10,
                 fontweight='bold')

    plt.axis('off')
    plt.show()

