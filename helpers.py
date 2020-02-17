import dlib
# import Image
from PIL import Image
import os
from skimage import io
#import matplotlib.pyplot as plt
import numpy as np
import imageio
import matplotlib.pyplot as plt
from mlxtend.image import extract_face_landmarks
import cv2


class Helpers(object):
    def detect_faces(self,image):
        # Create a face detector
        face_detector = dlib.get_frontal_face_detector()

        # Run detector and get bounding boxes of the faces on image.
        detected_faces = face_detector(image, 1)
        face_frames = [(x.left(), x.top(),
                        x.right(), x.bottom()) for x in detected_faces]

        return face_frames


    def devolverArchivos(self, carpeta):
        paths = []
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(dir_path)
        for archivo in os.listdir(carpeta):

        # print(os.path.join(carpeta,archivo))
            paths.append(os.path.join(carpeta, archivo))

            if os.path.isdir(os.path.join(carpeta, archivo)):
                self.devolverArchivos(os.path.join(carpeta, archivo))
                #print(paths)
        return paths


    def get_points(self, img):

        #img = imageio.imread(image_path)
        landmarks = extract_face_landmarks(img)
        print(landmarks.shape)
        print('\n\nPrimeros 20:\n', landmarks[:20])
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(img)
        ax = fig.add_subplot(1, 3, 2)
        ax.scatter(landmarks[:, 0], -landmarks[:, 1], alpha=0.8)
        ax = fig.add_subplot(1, 3, 3)
        img2 = img.copy()
        for p in landmarks:
            img2[p[1] - 3:p[1] + 3, p[0] - 3:p[0] + 3, :] = (255, 255, 255)
        ax.imshow(img2)
        plt.show()

    def rotation_detection_dlib(self,img):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('/home/julio/PycharmProjects/ethnicity/shape_predictor_5_face_landmarks.dat')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        if len(rects) > 0:
            for rect in rects:
                x = rect.left()
                y = rect.top()
                w = rect.right()
                h = rect.bottom()
                shape = predictor(gray, rect)
                shape = self.shape_to_normal(shape)
                nose, left_eye, right_eye = self.get_eyes_nose_dlib(shape)
                center_of_forehead = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
                center_pred = (int((x + w) / 2), int((y + y) / 2))
                length_line1 = self.distance(center_of_forehead, nose)
                length_line2 = self.distance(center_pred, nose)
                length_line3 = self.distance(center_pred, center_of_forehead)
                cos_a = self.cosine_formula(length_line1, length_line2, length_line3)
                angle = np.arccos(cos_a)
                rotated_point = self.rotate_point(nose, center_of_forehead, angle)
                rotated_point = (int(rotated_point[0]), int(rotated_point[1]))
                if self.is_between(nose, center_of_forehead, center_pred, rotated_point):
                    angle = np.degrees(-angle)
                else:
                    angle = np.degrees(angle)

                # if mode:
                #    img = rotate_opencv(img, nose, angle)
                # else:
                img = Image.fromarray(img)
                img = np.array(img.rotate(angle))
            # if show:
            #show_img(img)
            return img
        # else:
        # return img


    def shape_to_normal(self, shape):
        shape_normal = []
        for i in range(0, 5):
            shape_normal.append((i, (shape.part(i).x, shape.part(i).y)))
        return shape_normal


    def get_eyes_nose_dlib(self, shape):
        nose = shape[4][1]
        left_eye_x = int(shape[3][1][0] + shape[2][1][0]) // 2
        left_eye_y = int(shape[3][1][1] + shape[2][1][1]) // 2
        right_eyes_x = int(shape[1][1][0] + shape[0][1][0]) // 2
        right_eyes_y = int(shape[1][1][1] + shape[0][1][1]) // 2
        return nose, (left_eye_x, left_eye_y), (right_eyes_x, right_eyes_y)


    def distance(self,a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


    def rotate_point(self, origin, point, angle):
        ox, oy = origin
        px, py = point

        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy


    def is_between(self, point1, point2, point3, extra_point):
        c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (point2[1] - point1[1]) * (extra_point[0] - point1[0])
        c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (point3[1] - point2[1]) * (extra_point[0] - point2[0])
        c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (point1[1] - point3[1]) * (extra_point[0] - point3[0])
        if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
            return True
        else:
            return False


    def save_img(self,path, img):
        cv2.imwrite(path, img)

    def load_image(self,img_path):
        image = io.imread(img_path)
        return image
    def output_directory(self,output_path):
        os.mkdir(output_path)
        return output_path

    def cosine_formula(self,length_line1, length_line2, length_line3):
        cos_a = -(length_line3 ** 2 - length_line2 ** 2 - length_line1 ** 2) / (2 * length_line2 * length_line1)
        return cos_a

    def create_dataframe(self,getnico,picture_id,puntos):
        puntos_aux=[]
        puntos_aux.append(puntos)
        datos={
            "Grupo_etnico",getnico,
            "Foto_id",picture_id,
            "Coordenadas",puntos_aux,

            }
        return datos
