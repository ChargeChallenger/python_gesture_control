import cv2

cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8

camera = cv2.VideoCapture(0)
#camera.set(10, 200)

while(True):
    ret, frame = camera.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    if ret is True:
        cv2.imshow('camera', frame)
        if cv2.waitKey(1) & 0xFF == 27: 
            break

camera.release()
cv2.destroyAllWindows()
