import numpy as np
import cv2

cap = cv2.VideoCapture('transformers-4.mp4')
seen_black = False
start_black = []
end_black = []
frame_number = 0
current_clip_number = 0

video_width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
dimensions = (video_width, video_height)

def get_video_writer(clip_number):
    fourcc = cv2.cv.CV_FOURCC('F', 'M', 'P', '4')
    out = cv2.VideoWriter('clip%d.avi' % clip_number,
                          fourcc,
                          30.0,
                          dimensions,
                          1)
    return out

writer = get_video_writer(current_clip_number)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        # End of the video
        break

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    number_of_zeros = np.count_nonzero(gray > 10)
    if number_of_zeros == 0 and not seen_black:
        gray[0:100,0:100] = 255
        seen_black = True
        start_black.append(frame_number)
        writer.release()
        print('found black section %d' % current_clip_number)
    elif number_of_zeros > 0 and seen_black:
        end_black.append(frame_number)
        seen_black = False
        current_clip_number += 1
        writer = get_video_writer(current_clip_number)

    if number_of_zeros > 0:
        writer.write(frame)

    # Display the resulting frame
    # cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_number += 1

print start_black
print end_black
# When everything done, release the capture
cap.release()
writer.release()
cv2.destroyAllWindows()
