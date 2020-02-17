from helpers import Helpers
from PIL import Image
from mlxtend.image import extract_face_landmarks
import os
import pandas as pd
import sys

helpers = Helpers()

input_directory = sys.argv[1]
output_file = sys.argv[1]

paths = helpers.devolverArchivos("/home/julio/PycharmProjects/ethnicity/" + str(input_directory))
print(paths)
if not os.path.isdir("/home/julio/PycharmProjects/ethnicity/" + str(input_directory) + "/cropped/"):
    direc_salida = helpers.output_directory(
        "/home/julio/PycharmProjects/ethnicity/" + str(input_directory) + "/cropped/")
else:
    direc_salida = "/home/julio/PycharmProjects/ethnicity/" + str(input_directory) + "/cropped/"
if not os.path.isdir("/home/julio/PycharmProjects/ethnicity/" + str(input_directory) + "/rotatted/"):
    direc_roteted = helpers.output_directory(
        "/home/julio/PycharmProjects/ethnicity/" + str(input_directory) + "/rotatted/")
else:
    direc_roteted = "/home/julio/PycharmProjects/ethnicity/" + str(input_directory) + "/rotatted/"
datos = []

for i in paths:

    etiqueta = []
    coordenadas = []

    print(i)
    salida = str(i).split("/")
    image = helpers.load_image(i)
    detected_faces = helpers.detect_faces(image)
    print("salida", salida[6])
    if detected_faces is not None:
        for n, face_rect in enumerate(detected_faces):
            puntos = []
            face = Image.fromarray(image).crop(face_rect)

            # plt.subplot(1, len(detected_faces), n+1)
            # plt.axis('off')
            # plt.imshow(face)
            face.save(direc_salida + str(salida[6]) + ".jpg")
            image_croped = helpers.load_image(direc_salida + str(salida[6]) + ".jpg")
            image_rotated = helpers.rotation_detection_dlib(image_croped)
            if image_rotated is not None:
                face_aux = Image.fromarray(image_rotated)
            if image_rotated is None:
                print("vacio")
                puntos.append("NA")
            else:
                face_aux.save(direc_roteted + str(salida[6] + ".jpg"))
                landmarks = extract_face_landmarks(image_rotated)
                if landmarks is not None:
                    for i in range(len(landmarks)):

                        # print("etiqueta:", i)
                        for j in range(1, len(landmarks[i])):
                            # print("coordenadas", j)

                            puntos.append(i)
                            print("etiqueta:", i)
                            puntos.append(landmarks[i])
                            print("coordenadas", landmarks[i])
                            print("puntos", puntos)

        # salida=str(i).split("/")
        salida_aux = salida[6].split(".")
        # print(salida_aux[0])
        image_id = salida_aux[0]
        image_tipo = image_id.split("_")
        image_tipo_aux = image_tipo[0]
        # print(image_tipo_aux)
        # datos.append(helpers.create_dataframe(salida_aux,image_id,puntos))
        # print("datos",datos)
        datos_aux = str(image_tipo_aux) + "," + str(salida_aux[0]) + "," + str(puntos)
        datos.append(datos_aux)

    else:
        print("NO SE DETECTA CARA")
df = pd.DataFrame(datos)
df.to_csv("/home/julio/PycharmProjects/ethnicity/" + str(output_file) + ".csv", sep=';')
print("datos", df)
