import cv2


vid = cv2.VideoCapture(0) 

while True: 
    ret, frame = vid.read() 
    cv2.imshow('Video', frame) 
    print(frame.shape)
      
    # Close the window if `q` is pressed.
    if (cv2.waitKey(1) & 0xFF) == ord('q'): 
        break

# After the loop release the capture object 
vid.release() 

# Close the video window
cv2.destroyAllWindows() 