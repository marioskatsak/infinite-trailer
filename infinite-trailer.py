import numpy as np
import cv2

cap = cv2.VideoCapture('transformers-4.mp4')
seen_black = False
start_black = []
end_black = []
frame_number = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    number_of_zeros = np.count_nonzero(gray > 10)
    if number_of_zeros == 0 and not seen_black:
        gray[0:100,0:100] = 255
        seen_black = True
        start_black.append(frame_number)
    elif number_of_zeros > 0 and seen_black:
        end_black.append(frame_number)
        seen_black = False

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_number += 1

print start_black
print end_black	
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
