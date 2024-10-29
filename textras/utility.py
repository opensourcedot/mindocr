import cv2
import numpy as np

def add_padding(image, padding_size, padding_color=(0, 0, 0)):
    if isinstance(padding_size, int):
        top, bottom, left, right = padding_size, padding_size, padding_size, padding_size
    else:
        top, bottom, left, right = padding_size

    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return padded_image

def sort_words_by_poly(words, polys):
    '''
    Sort word-boxes by polygon position 
    '''
    from functools import cmp_to_key
    def compare(x, y):
        dist1 = y[1][3][1] - x[1][0][1]
        dist2 = x[1][3][1] - y[1][0][1]
        if abs(dist1 - dist2) < x[1][3][1] - x[1][0][1] or abs(dist1 - dist2) < y[1][3][1] - y[1][0][1]:
            if x[1][0][0] < y[1][0][0]:
                return -1
            elif x[1][0][0] == y[1][0][0]:
                return 0
            else:
                return 1
        else:
            if x[1][0][1] < y[1][0][1]:
                return -1
            elif x[1][0][1] == y[1][0][1]:
                return 0
            else:
                return 1
    tmp = sorted(zip(words, polys), key=cmp_to_key(compare))
    return [item[0][0] for item in tmp]