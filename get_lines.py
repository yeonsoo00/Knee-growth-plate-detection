import cv2
import numpy as np

def getBordered(image, width=1):
    bg = np.zeros(image.shape)
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest = 0
    bigcontour = None
    for contour in contours:
        area = cv2.contourArea(contour) 
        if area > biggest:
            biggest = area
            bigcontour = contour
    return cv2.drawContours(bg, [bigcontour], 0, (255, 255, 255), width).astype(bool) 

def getRangePoints(mask, type):
    # Get range points
    # type == bottom : the case of most bottom 
    # type == top : the case of most top
    roi_idx_x, roi_idx_y = np.where(mask==1)
    unique_y = np.unique(roi_idx_y)
    minmax_y = [np.min(unique_y), np.max(unique_y)]
    points_list = []

    for y in minmax_y:
        x_list = []
        overlapped_y_idx = np.where(np.array(roi_idx_y) ==y)[0]
        x_list = [roi_idx_x[i] for i in overlapped_y_idx]
        points_list.append([x_list, y])
    rangepoints = []
    for xy in points_list:
        if type == 'bottom':
            pair = (max(xy[0]), xy[1])
        else:
            pair = (min(xy[0]), xy[1])
        rangepoints.append(pair)

    return rangepoints

def draw_line(mask, type, thickness=1):
    """
    mask : closed binary mask of ROI object (0, 1)
    outline : binary image (0, 1)
    return : connected most bottom line or top line
    """
    outline = (getBordered(np.array(mask*255, np.uint8))).astype(np.uint8)
    point_a, point_b = getRangePoints(mask, type)

    broken_outline = np.copy(outline)
    broken_outline[point_a] = 0
    broken_outline[point_b] = 0

    contours, _ = cv2.findContours(broken_outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areaArray = []
    for c in contours:
        area = cv2.contourArea(c)
        areaArray.append(area)

    min_contour = min(contours, key=cv2.contourArea)
    min_line_img = np.zeros(broken_outline.shape)
    cv2.drawContours(min_line_img, [min_contour], -1, 255, -1)

    max_contour = min(contours, key=cv2.contourArea)
    max_line_img = np.zeros(broken_outline.shape)
    cv2.drawContours(max_line_img, [max_contour], -1, 255, -1)

    if np.sum(min_line_img, axis=1)[0] > 0:
        line_img = max_line_img
    else:
        line_img = min_line_img

    thickened_line = cv2.dilate(line_img, np.ones((thickness, thickness), np.uint8))
    
    return thickened_line