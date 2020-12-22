import tensorflow.keras
from PIL import Image, ImageOps
import cv2
import numpy as np


model = tensorflow.keras.models.load_model('model/keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


vid = cv2.VideoCapture(0) 

while True: 
    ret, raw_frame = vid.read()
    # OpenCV uses the blue-green-red color system and everyone else
    # use the red-green-blue system, so we need to reorder the channels
    frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
    # OpenCV does not mirror the image, but teachable machine does.
    frame = frame[:, ::-1]
  
    # Convert to PIL-style image so we can reshape it correctly
    image = Image.fromarray(frame)

    # Resize the image to a 224x224 with the same strategy as in TM2:
    # Resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # Turn the image back into a numpy array and normalise it
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127) - 1

    # Load the image into the array
    data = normalized_image_array.reshape(1, 224, 224, 3)

    # Predict which label the image should have
    prediction = model.predict(data)
    image_class = np.argmax(prediction)
    print(f"Class: {image_class}")
    
    # Display the video
    cv2.imshow('Video', raw_frame) 
      
    # Close the window if `q` is pressed.
    if (cv2.waitKey(1) & 0xFF) == ord('q'): 
        break


# After the loop release the capture object 
vid.release() 

# Close the video window
cv2.destroyAllWindows() 
