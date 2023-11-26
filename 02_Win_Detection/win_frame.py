import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib
matplotlib.use('TKAgg')
import argparse


def resize(img, height=1024, width=1024):
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def ConTour(image_path, line_width = 80):
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)
    contours, hierarchy = cv2.findContours(edged,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("Number of Contours found = " + str(len(contours)))

    # Draw all contours
    outlined_img = cv2.drawContours(image, contours, -1, (0, 0, 0), line_width)

    return outlined_img

def Find_Biggest_ConTour(image):

    img = np.uint8(image)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    edged = cv2.Canny(blurred, 10, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(edged, kernel, iterations=1)
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contour = sorted(contours, key=cv2.contourArea, reverse=True)

    h, w = image.shape[:2]
    canvas = np.zeros((h, w))

    # draw the contours on a copy of the original image
    img_pl_con = cv2.drawContours(canvas, sorted_contour, 0, (255, 255, 255), thickness = -1)
    img_pl_con = cv2.drawContours(img_pl_con, sorted_contour, 0, (255, 255, 255), thickness= 40)

    return img_pl_con

def Hide_Biggest_ConTour(image):

    img = np.uint8(image)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    edged = cv2.Canny(blurred, 10, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(edged, kernel, iterations=1)
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours), "Contour objects.")
    sorted_contour = sorted(contours, key=cv2.contourArea, reverse=True)

    # h, w = image.shape[:2]
    # canvas = np.zeros((h, w))
    # draw the contours on a copy of the original image
    img_pl_con = cv2.drawContours(image, sorted_contour, 0, (0, 0, 0), thickness = -1)

    return img_pl_con

class BoundingBox(object):
    """
    A 2D bounding box
    """
    def __init__(self, points):
        if len(points) == 0:
            raise ValueError("Can't compute bounding box of empty list")
        self.minx, self.miny = float("inf"), float("inf")
        self.maxx, self.maxy = float("-inf"), float("-inf")
        for x, y in points:
            # Set min coords
            if x < self.minx:
                self.minx = x
            if y < self.miny:
                self.miny = y
            # Set max coords
            if x > self.maxx:
                self.maxx = x
            elif y > self.maxy:
                self.maxy = y
    @property
    def width(self):
        return self.maxx - self.minx
    @property
    def height(self):
        return self.maxy - self.miny
    def __repr__(self):
        return "BoundingBox({}, {}, {}, {})".format(
            self.minx, self.maxx, self.miny, self.maxy)

def bounding_box_numpy(points):
    """
    Find min/max from an N-collection of coordinate pairs, shape = (N, 2), using
    numpy's min/max along the collection-axis
    """
    max = np.max(points)
    min = np.min(points)
    mid = np.median(points)
    return min, mid, max







if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Description of your program.')

    parser.add_argument('--sem_path', type=str, default='img/front.png', help='Path to the semantic image')
    parser.add_argument('--line_path', type=str, default='img/front-0.74.png', help='Path to the line image')
    parser.add_argument('--process_mode', type=str, default='cube', choices=['cube', 'broken_win'],
                        help='Mode of processing (choose from cube or other_mode)')

    args = parser.parse_args()

    """
    read the local sem and line files
    """
    sem_path = args.sem_path
    line_path = args.line_path
    process_mode = args.process_mode

    # sem_path = 'img/front.png'
    # line_path = 'img/front-0.74.png'
    # process_mode = 'cube'
    #process_mode = 'broken_win'

    # if process_mode == 'pano':
    #    sem_path = 'img/04_left.png'

    if process_mode == 'cube':

        # #mode = 'regular'
        # mode = 'window' #window_on_the_door
        # door_pos = 'upper'
        # biggest_door = 'yes'

        sem_img = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)
        line_img = cv2.imread(line_path, cv2.IMREAD_GRAYSCALE)

        h,w = sem_img.shape[:2]
        base_img = np.zeros((h,w))
        line_img = resize(line_img, height=h, width=w)
        door_win = base_img.copy()

        #get window
        base_img[sem_img == 230] = 255
        base_img[sem_img == 64] = 255
        # if mode == 'window_on_the_door':
        #     door_win[sem_img == 157] = 255
        #     door_win = Find_Biggest_ConTour(door_win)
        #     base_img = base_img + door_win

        base_img = cv2.rectangle(base_img, (0, 0), (w, h), (0,0,0), 80)
        cv2.imwrite('01_wind.png', base_img)
        outlined_base = ConTour('01_wind.png', line_width = 35)

        if np.max(outlined_base) == 0:
            outlined_base = ConTour('01_wind.png', line_width = 15)

        base_img = cv2.cvtColor(outlined_base, cv2.COLOR_BGR2GRAY)

        #get frame
        base_img[line_img == 38] = 0
        cv2.imwrite('01_frame.png', base_img)

        # line_img = cv2.rectangle(line_img, (0, 0), (w, h), 38, 30)
        # win_constraint = Find_Biggest_ConTour(line_img)
        #
        # base_img[win_constraint == 255] = 0
        # cv2.imwrite('01_win.png', base_img)
        #
        # win_constraint = 255 - win_constraint
        # win_constraint = cv2.rectangle(win_constraint, (0, 0), (w, h), 0, 80)
        # cv2.imwrite('01_win_constraint.png', win_constraint)



    if process_mode == 'broken_win':

       sem_img = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)
       sem_img[sem_img == 230] = 255
       sem_img[sem_img != 255] = 0

       #get the min and max values for x, y coordinates
       indices = np.where(sem_img == 255)
       min_x, mid_x, max_x = bounding_box_numpy(indices[1])
       min_y, mid_y, max_y = bounding_box_numpy(indices[0])

       #draw new bounding box
       contours = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
       sem = cv2.fillPoly(sem_img, pts=[contours], color=(255, 255, 255))
       cv2.imwrite("filled_Polygon.png", sem)


