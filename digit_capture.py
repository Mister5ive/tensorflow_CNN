# coding:utf-8
import cv2 as cv
import numpy as np
import sys
import datetime
import CNN_Model_Restore as predict

width = 512
height = 512
img = 0
palette = 0
point = (-1,-1)
min_x = sys.maxsize
min_y = sys.maxsize
max_x = -1
max_y = -1
line_thickness1 = 3
line_thickness2 = 20
drawing = False


def img_normal(raw):
    # normalize pixels to 0 and 1.
    # digit is 1, background is 0
    norm = np.array(raw,np.float32)
    norm /= 255.0
    return norm

def palette_init(width,height,channels,type,border = False):
    palette = np.zeros((width, height, channels), type)
    if border == True:
        cv.line(palette, (0,int(height / 2)), (width - 1, int(height / 2)), (255, 255, 255), 1)
        cv.line(palette, (int(width / 2),0), (int(width / 2),height -1 ), (255, 255, 255), 1)
    return palette

def draw_digit(event,x,y,flags,param):
    global palette
    global drawing
    global point
    global min_x
    global min_y
    global max_x
    global max_y

    if event == cv.EVENT_LBUTTONDOWN:
        point = (x,y)
        if x > max_x:
            max_x = x
        elif x < min_x:
            min_x = x

        if y > max_y:
            max_y = y
        elif y < min_y:
            min_y = y

        drawing = True
    if event == cv.EVENT_LBUTTONUP:
        point = (0, 0)
        drawing = False
    if event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_LBUTTONDOWN:
        if drawing == True:

            if x > max_x:
                max_x = x
            elif x < min_x:
                min_x = x

            if y > max_y:
                max_y = y
            elif y < min_y:
                min_y = y

            cv.line(palette,point,(x,y),(255,255,255),line_thickness1)
            cv.line(img, point, (x, y), (255, 255, 255), line_thickness2)
            point = (x, y)

def detect():
    row_min = min_y - 2 * line_thickness2
    row_max = max_y + 2 * line_thickness2

    col_min = min_x - 2 * line_thickness2
    col_max = max_x + 2 * line_thickness2
    roi = img[row_min if row_min >= 0 else 0:row_max if row_max < height else height - 1,
          col_min if col_min >= 0 else 0:col_max if col_max <= width else width - 1]

    # process
    # roi = cv.bitwise_not(roi)
    #cv.imshow('roi', roi)
    img_capture = cv.resize(roi, (28, 28))
    #cv.imshow('img_capture', img_capture)

    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    img_capture = cv.dilate(img_capture, kernel)
    #cv.imshow('dilate', img_capture)

    img_norm = img_normal(img_capture)

    img_newshape = np.reshape(img_norm, [-1, 28 * 28])
    start = datetime.datetime.now()
    predict.detect(img_newshape)
    end = datetime.datetime.now()
    print('cost time:%d s' % (end - start).seconds)


if __name__ == '__main__':
    cv.namedWindow('Input')
    cv.setMouseCallback('Input',draw_digit)
    palette = palette_init(width, height, 1, np.uint8,border=True)
    img = palette_init(width, height, 1, np.uint8)

    while(1):
        cv.imshow('Input',palette)

        c = cv.waitKey(33)&0xFF

        if c == ord('q') :
            cv.destroyAllWindows()
            break;
        elif c== ord('r'):
            palette = palette_init(512, 512, 1, np.uint8, border=True)
            img = palette_init(512, 512, 1, np.uint8)
            min_x = sys.maxsize
            min_y = sys.maxsize
            max_x = -1
            max_y = -1
        elif c == ord('f'):
            detect()












