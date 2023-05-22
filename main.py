import sys
import cv2 as cv
import numpy as np
from keras.models import model_from_json
from keras.utils import img_to_array
from utils import nightfilter

IMAGE_SIZE = (48, 48)

# Load the pre-trained face detection classifier
face_cascade = cv.CascadeClassifier('./models/haarcascade_frontalface_default.xml')

def faceDetection(model, emotions, nightmode = False):

    # Load the video stream
    stream = cv.VideoCapture(0)

    # Loop through each frame of the video
    while stream.isOpened():

        success, frame = stream.read()

        if not success:
            break

        # Convert the frame to grayscale
        frame_grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if (nightmode):
            frame_grayscale = nightfilter(frame_grayscale)
            
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(frame_grayscale, scaleFactor=1.3, minNeighbors=5)

        # Loop through each face and draw a rectangle around it
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
      
            detected_face = frame_grayscale[int(y):int(y+h), int(x):int(x+w)]
            detected_face = cv.equalizeHist(detected_face)
            detected_face = cv.resize(detected_face, IMAGE_SIZE)
            
            image_data = img_to_array(detected_face)
            image_data = np.expand_dims(image_data, axis = 0)

            image_data /= 255

            predictions = model.predict(image_data)

            max_index = np.argmax(predictions[0])
            
            emotion = emotions[max_index]

            cv.putText(frame, emotion, (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        if (nightmode):
            b,g,r = cv.split(frame)

            b = nightfilter(b)
            g = nightfilter(g) 
            r = nightfilter(r)

            frame = cv.merge((b,g,r))

        # Display the resulting frame
        cv.imshow('Video Stream', frame)

        # Exit if the 'q' key is pressed or the window is closed
        if cv.waitKey(1) & 0xFF == ord('q') or cv.getWindowProperty('Video Stream', cv.WND_PROP_VISIBLE) < 1:
            break

    # Release the video stream and close all windows
    stream.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    if "--model" not in sys.argv:
        print("Please specify a model.")
        print("Usage: python main.py --model <[dex|vgg]>")
        exit()
        
    model_name = sys.argv[sys.argv.index("--model") + 1]

    if model_name not in ["dex", "vgg"]:
        print("Model not found.")
        print("Please choose between 'dex' and 'vgg'.")
        exit()

    print("Loading model %s..." % model_name)

    model_path = ""
    model_weights_path = ""
    model = None
    emotions = None

    if model_name == "dex":
        model_path = "models/deXpression.json"
        model_weights_path = "models/deXpression_weights.h5"
        emotions = ('neutral', 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise')
    else :
        model_path = "models/vgg.json"
        model_weights_path = "models/vgg_weights.h5"
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

    model = model_from_json(open(model_path, "r").read())
    model.load_weights(model_weights_path)

    print("Model loaded.")

    faceDetection(model, emotions)