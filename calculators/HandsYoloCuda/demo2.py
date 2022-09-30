import os
import os.path as osp
import sys

import cv2
import calculators.HandsYoloCuda.handdet as handdet

from cameras import Cameras
import time


def detect_demo(imgs):
    for img in imgs:
        im = cv2.imread(img)
        res = handdet.detect(im)
        rim = handdet.visualize(im, res)
        cv2.imwrite(f"saved{osp.basename(img)}", rim)
    print("Done!")


def crop_demo(img):
    im = cv2.imread(img)
    res = handdet.detect(im)
    crops = handdet.crop(im, res)
    for (i, crop) in enumerate(crops):
        cv2.imwrite(f"crop_{i}.png", crop)

if __name__ == '__main__':
    cams = Cameras()
    cams.captureRGB(0)

    f_count = 0
    t1 = time.time()
    while True:
        if f_count>30:
            t1=time.time()
            f_count=0
        f_count+=1

        img = cams.getRGBFrame(0)
        res = handdet.detect(img)
        img = handdet.visualize(img, res)

        fps = f_count / (time.time() - t1)
        cv2.putText(img, "FPS: %.2f" % (fps), (int(20), int(40)), 0, 5e-3 * 200, (0, 255, 0), 3)

        cv2.imshow('img',img)
        if cv2.waitKey(5) & 0xFF == 27:
            break

