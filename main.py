# This is a sample Python script.
from matplotlib import pyplot as plt

from function import predict_objects, display_image_with_objects

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Testarea funcționalității aplicației cu o imagine din directorul local
if __name__ == "__main__":
    # Calea către imaginea din directorul local
    image_path = "assets/images/img5.jpg"  # Calea către imaginea din sistemul de fișiere local1

    # Apelarea funcției de recunoaștere a obiectelor și obținerea rezultatelor
    predictions, processed_image = predict_objects(image_path)

    # Afișarea imaginii cu obiectele recunoscute și numele lor deasupra imaginii
    display_image_with_objects(processed_image, predictions)


