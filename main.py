import cv2
import numpy as np
from tensorflow.keras.models import load_model

# connection to the video camera
cap = cv2.VideoCapture(0)

# gesture classes
classes = ["01_palm", "02_I", "03_Fist", "04_Fist_Moved", "05_Thumb", "06_Index", "07_Ok", "08_Palm_moved", "09_C",
           "10_Down"]

# loading trained model
model = load_model("D:\\HandGestureRecognition\\trained_model\\my_model.keras")


# preprocessing of frame
def preprocessing_frame(frame):
    # converting frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # resizzing the frame to feed to our CNN model
    resized_frame = cv2.resize(gray_frame, (64, 64))

    # normalizing the frame
    normalized_frame = resized_frame / 255.0

    # adding channel to the frame (64,64,1) for grayscale img
    expanded_frame = np.expand_dims(normalized_frame, axis=-1)
    # changing shape from (64,64,1) to (1,64,64,1) as our model expects this input shape
    final_frame = np.expand_dims(expanded_frame, axis=0)
    return final_frame


while True:
    ret, frame = cap.read()  # reading  frame by frame
    if not ret:  # ret is a boolean value which tells us if the frame is captured properly or not
        break

    # preprocessing the frame
    processed_frame = preprocessing_frame(frame)

    # prediction
    prediction = model.predict(processed_frame)
    gesture_index = np.argmax(prediction)
    gesture = classes[gesture_index]
    cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
