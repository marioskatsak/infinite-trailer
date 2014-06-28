import numpy as np
import cv2

cap = cv2.VideoCapture('./transformers-4.mp4')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    number_of_zeros = np.count_nonzero(gray)
    if number_of_zeros == 0:
        gray[0:100] = 255

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
