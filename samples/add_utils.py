import math
import cv2
import numpy as np


def draw_angled_rec(x0, y0, image, width, height, color, angle):

    _angle = angle * math.pi / 180.0
    b = math.cos(_angle) * 0.5
    a = math.sin(_angle) * 0.5
    pt0 = (int(x0 - a * height - b * width),
           int(y0 + b * height - a * width))
    pt1 = (int(x0 + a * height - b * width),
           int(y0 - b * height - a * width))
    pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
    pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))
    points = np.array([[pt0, pt1, pt2, pt3]], dtype=np.int32)

    image = cv2.fillPoly(image, points, color)
    return image

    #cv2.line(img, pt0, pt1, (255, 255, 255), 3)
    #cv2.line(img, pt1, pt2, (255, 255, 255), 3)
    #cv2.line(img, pt2, pt3, (255, 255, 255), 3)
    #cv2.line(img, pt3, pt0, (255, 255, 255), 3)
