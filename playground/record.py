import cv2
from cameras import Cameras

cams = Cameras()
cams.captureRGB(0)

writer= cv2.VideoWriter('basicvideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (640,480))


while True:
    img = cams.getRGBFrame(0)

    writer.write(img)

    cv2.imshow('frame', img)

    if cv2.waitKey(1) & 0xFF == 27:
        break


writer.release()
cv2.destroyAllWindows()