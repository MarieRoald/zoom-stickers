import tensorflow.keras
from PIL import Image, ImageOps
import cv2
import numpy as np


def overlay_sticker(background, transparent_sticker):
    sticker = transparent_sticker[:, :, :3]
    alpha = transparent_sticker[:, :, 3:].astype('float32')/255 
    background_alpha = 1 - alpha

    return np.uint8(background*background_alpha + sticker*alpha)

model = tensorflow.keras.models.load_model('model/keras_model.h5')


vid = cv2.VideoCapture(0) 
# -1 at the end to load alpha channel
stickers = [
    None, # The background class has no sticker,
    cv2.imread("resources/heart.png", -1),
    cv2.imread("resources/thumbs_up.png", -1),
    cv2.imread("resources/question.png", -1),
]

while True: 
    ret, raw_frame = vid.read()
    # OpenCV doesn't mirror when we display the image,
    # which can make the video stream look weird.
    raw_frame = raw_frame[:, ::-1]
    
    # OpenCV uses the blue-green-red color system and everyone else
    # use the red-green-blue system, so we need to reorder the channels
    frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
    # OpenCV does not mirror the image, but teachable machine does.
    
    # Convert to PIL-style image so we can reshape it correctly
    image = Image.fromarray(frame)

    # Resize the image to a 224x224 with the same strategy as in TM2:
    # Resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # Turn the image back into a numpy array
    image_array = np.asarray(image)

    # Finalise the image array so it can be used by Keras
    data = (image_array.reshape(1, 224, 224, 3) / 127) - 1

    # Predict which label the image should have
    prediction = model.predict(data)
    image_class = np.argmax(prediction)

    # Add the sticker
    print(image_class)
    if stickers[image_class] is not None:
        raw_frame = overlay_sticker(raw_frame, stickers[image_class])
    
    # Display the video
    cv2.imshow('Video', raw_frame) 
      
    # Close the window if `q` is pressed.
    if (cv2.waitKey(1) & 0xFF) == ord('q'): 
        break


# After the loop release the capture object 
vid.release() 

# Close the video window
cv2.destroyAllWindows() 
