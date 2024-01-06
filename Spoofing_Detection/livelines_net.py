import cv2
import os
import numpy as np
from tensorflow.keras.models import model_from_json
from facenet_pytorch import MTCNN
import time
import torch



root_dir = os.getcwd()
# Load Face Detection Model
# face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
# Load Anti-Spoofing Model graph
json_file = open('model/mobilenet_face_antispoofing_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load antispoofing model weights 
model.load_weights('model/finalyearproject_antispoofing_model_98-0.942647.h5')
print("Model loaded from disk")

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
prev_frame_time = 0
new_frame_time = 0

mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

while cap.isOpened():
    isSuccess, frame = cap.read()
    if isSuccess:
        # Detect faces using MTCNN
        boxes, _, points_list = mtcnn.detect(frame, landmarks=True)
        if boxes is not None:
            for box in boxes:
                bbox = list(map(int, box.tolist()))
                x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                
                # Ensure coordinates are within the frame to avoid empty slices
                x1, y1 = max(0, x-5), max(0, y-5)
                x2, y2 = min(frame.shape[1], x+w+5), min(frame.shape[0], y+h+5)
                
                # Extract the face ROI
                face = frame[y1:y2, x1:x2]
                
                # Only proceed if the face ROI is not empty
                if face.size > 0:
                    # Resize the face to the expected input size of the model
                    resized_face = cv2.resize(face, (160, 160))
                    resized_face = resized_face.astype("float") / 255.0
                    resized_face = np.expand_dims(resized_face, axis=0)
                    
                    # Predict real or spoof
                    preds = model.predict(resized_face)[0]
                    print(preds)
                    if preds >  0.5:
                        label = f'spoof: {preds}'
                        cv2.putText(frame, label, (x,y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                        cv2.rectangle(frame, (x, y), (x+w,y+h),
                            (0, 0, 255), 2)
                    else:
                        label = f'real: {preds}'
                        cv2.putText(frame, label, (x,y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                        cv2.rectangle(frame, (x, y), (x+w,y+h),
                        (0, 255, 0), 2)

                    # label = 'spoof' if preds > 0.5 else 'real'
                    # color = (0, 0, 255) if label == 'spoof' else (0, 255, 0)
                    # cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                else:
                    print("Face ROI is empty, skipping to the next frame.")

        # Display the frame
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

# Release the video capture and close windows
cap.release()        
cv2.destroyAllWindows()

# # Release the video capture and close windows
# cap.release()        
# cv2.destroyAllWindows()

#     new_frame_time = time.time()
#     fps = 1/(new_frame_time-prev_frame_time)
#     prev_frame_time = new_frame_time
#     fps = str(int(fps))
#     cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
#     cv2.imshow('Face Detection', frame)
#     if cv2.waitKey(1)&0xFF == 27:
#         break
# cap.release()
# cv2.destroyAllWindows()