import numpy as np
import cv2
import tensorflow as tf

model = tf.keras.models.load_model('cnn.keras')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
emotions=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = face_cascade.detectMultiScale(rgb, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Crop face
        face = frame[y:y+h, x:x+w]

        # Preprocess for classifier
        face_resized = cv2.resize(face, (48,48)) / 255.0  # Adjust to your model's input size
        face_resized = np.expand_dims(face_resized, axis=0)

        # Predict emotion
        prediction = model.predict(face_resized)
        emotion_label = np.argmax(prediction)  # Replace with mapping if you have label names

        # Draw box + emotion label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Emotion: {emotions[emotion_label]}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Face Detection + Emotion Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
