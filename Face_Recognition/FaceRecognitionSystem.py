import cv2
import dlib
from simpleface_reg import SimpleFacerec
from mtcnn.mtcnn import MTCNN
import numpy as np
import face_recognition
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.layers import Conv2D, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model

class FaceRecognitionSystem:
    def __init__(self):
        self.mtccnDetector = MTCNN()
        self.dlib_detector = dlib.get_frontal_face_detector()

        modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
        configFile = "models/deploy.prototxt.txt"
        self.caffe_net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

        self.cascade_cls = cv2.CascadeClassifier('models/haarcascade_frontalface2.xml')

        self.sfr = SimpleFacerec()
        self.sfr.load_encoding_images("./image/")

        self.spoofing_model = self._load_mobilenetv2_spoof_model()

    @staticmethod
    def _load_mobilenetv2_spoof_model():
        # img_width, img_height = 224, 224
        # pretrain_net = mobilenet_v2.MobileNetV2(input_shape=(img_width, img_height, 3), include_top=False, weights='imagenet')
        # x = pretrain_net.output
        # x = Conv2D(32, (3, 3), activation='relu')(x)
        # x = Dropout(rate=0.2, name='extra_dropout1')(x)
        # x = GlobalAveragePooling2D()(x)
        # x = Dense(1, activation='sigmoid', name='classifier')(x)
        # model = Model(inputs=pretrain_net.input, outputs=x, name='mobilenetv2_spoof')
        # model.load_weights('mobilenetv2-best.hdf5')
        # Load Anti-Spoofing Model graph
        json_file = open('models/MobileNetV2.json','r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load antispoofing model weights 
        model.load_weights('model/finalyearproject_antispoofing_model_98-0.942647.h5')
        return model

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            fin = self.process_frame(frame)
            cv2.imshow("Frame", fin)
            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        frame_1 = frame.copy()
        small_frame = cv2.resize(frame_1, (0, 0), fx=self.sfr.frame_resizing, fy=self.sfr.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_1 = self.mtccnDetector.detect_faces(rgb_small_frame)

        for result in face_1:
            x, y, w, h = result['box']
            x1, y1 = x + w, y + h
            
            try:
                drop_frame = cv2.resize(rgb_small_frame[x:x+w, y:y+h], (224, 224))
                batch_drop_frame = np.expand_dims(drop_frame, axis=0)
                prediction = self.spoofing_model.predict(batch_drop_frame)
                prob = prediction[-1][-1]
                
                if prob >= 0.7:
                    spoof_text = "Real"
                    cv2.putText(frame_1, spoof_text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                else:
                    spoof_text = "Spoof"
                    cv2.putText(frame_1, spoof_text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                
            except cv2.error as e:
                spoof_text = "Spoof"
                cv2.putText(frame_1, spoof_text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)


            # Scale back up face locations since the frame we detected in was scaled
            x *= int(1/self.sfr.frame_resizing)
            y *= int(1/self.sfr.frame_resizing)
            x1 *= int(1/self.sfr.frame_resizing)
            y1 *= int(1/self.sfr.frame_resizing)

            # Extract the face encoding
            face_encoding = face_recognition.face_encodings(frame_1, [(y, x1, y1, x)])[0]

            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.sfr.known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.sfr.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.sfr.known_face_names[best_match_index]

            # Draw rectangle around the face and put a label with the name
            cv2.rectangle(frame_1, (x, y), (x1, y1), (0, 0, 255), 2)
            cv2.putText(frame_1, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame_1, 'mtcnn', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        
        
        frame_2 = frame.copy()
        small_frame = cv2.resize(frame_2, (0, 0), fx=self.sfr.frame_resizing, fy=self.sfr.frame_resizing)
        gray_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        face_2 = self.dlib_detector(gray_small_frame, 1)

        for result in face_2:
            x, y = result.left(), result.top()
            x1, y1 = result.right(), result.bottom()

            # Scale back up face locations
            x, y, x1, y1 = [int(v / self.sfr.frame_resizing) for v in [x, y, x1, y1]]

            face_encoding = face_recognition.face_encodings(frame_2, [(y, x1, y1, x)])[0]
            matches = face_recognition.compare_faces(self.sfr.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.sfr.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.sfr.known_face_names[best_match_index]

            cv2.rectangle(frame_2, (x, y), (x1, y1), (0, 0, 255), 2)
            cv2.putText(frame_2, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame_2, 'dlib', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        
        frame_3 = frame.copy()
        h, w = frame_3.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame_3, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
        self.caffe_net.setInput(blob)
        face_3 = self.caffe_net.forward()

        for i in range(face_3.shape[2]):
            confidence = face_3[0, 0, i, 2]
            if confidence > 0.5:
                box = face_3[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")

                face_encoding = face_recognition.face_encodings(frame_3, [(y, x1, y1, x)])[0]
                matches = face_recognition.compare_faces(self.sfr.known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(self.sfr.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.sfr.known_face_names[best_match_index]

                cv2.rectangle(frame_3, (x, y), (x1, y1), (0, 0, 255), 2)
                cv2.putText(frame_3, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame_3, 'dnn', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        
        
        frame_4 = frame.copy()
        gray_frame = cv2.cvtColor(frame_4, cv2.COLOR_BGR2GRAY)
        face_4 = self.cascade_cls.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in face_4:
            x1, y1 = x + w, y + h

            face_encoding = face_recognition.face_encodings(frame_4, [(y, x1, y1, x)])[0]
            matches = face_recognition.compare_faces(self.sfr.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.sfr.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.sfr.known_face_names[best_match_index]

            cv2.rectangle(frame_4, (x, y), (x1, y1), (0, 0, 255), 2)
            cv2.putText(frame_4, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame_4, 'haar', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            
        
        h1 = cv2.hconcat([frame_1, frame_2])
        h2 = cv2.hconcat([frame_3, frame_4])
        fin = cv2.vconcat([h1, h2])
        return fin

""" __Usage:
face_system = FaceRecognitionSystem()
face_system.run()
"""