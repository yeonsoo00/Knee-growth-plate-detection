import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from scipy.ndimage import gaussian_filter

def draw_line(mask):
    # Get the most bottom line
    roi_idx_x, roi_idx_y = np.where(mask==255)
    unique_y = np.unique(roi_idx_y)
    points_list = []
    for y in unique_y:
        x_list = []
        overlapped_y_idx = np.where(np.array(roi_idx_y) ==y)[0]
        x_list = [roi_idx_x[i] for i in overlapped_y_idx]
        points_list.append([x_list, y])
    bottom_line_idx = []
    for xy in points_list:
        pair = [max(xy[0]), xy[1]]
        bottom_line_idx.append(pair)

    bottom_line = np.zeros(mask.shape)
    for i, j in bottom_line_idx:
        bottom_line[i,j] = 1

    return bottom_line

class mineral:
    """
        color chennal : 2
        line : 09D0CML
    """
    def __init__(self, path2img):
        self.img = np.array(plt.imread(path2img))
        self.img_section = self.img[int(self.img.shape[0]/3):int(self.img.shape[0]/2), :]

    def get_line09(self):
        blurred = cv2.GaussianBlur(self.img_section, (7,7), 0)

        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

        magnitude = cv2.magnitude(sobelx, sobely)
        magnitude = cv2.convertScaleAbs(magnitude)

        threshold = 200  
        edges = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)[1]
        kernel1 = np.ones((7,7), np.uint8)
        kernel2 = np.ones((2,2), np.uint8)
        dilated_image = cv2.dilate(edges, kernel1, iterations=7)
        closed_img = cv2.erode(dilated_image, kernel2, iterations=7)

        # Close the object
        idx_img=sorted(np.where(closed_img==255)[1])
        left_idx = idx_img[0]
        right_idx = idx_img[-1]
        closed_img[0, left_idx:right_idx] = 255

        # Find mask
        contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        largest_contour_mask = np.zeros_like(closed_img)
        cv2.drawContours(largest_contour_mask, [largest_contour], -1, 255, -1)

        # Gaussian filter
        mask09_=gaussian_filter(largest_contour_mask, sigma=70, mode="wrap")
        mask09 = np.where(mask09_>255*0.5, 255, 0)

        line09 = draw_line(mask09)
        return line09

    def plot_img(self, path2save, dilation=True):
        line09 = mineral.get_line09()
        
        if dilation:
            line09 = cv2.dilate(line09, kernel=np.ones((7,7), np.uint8))

        overlapped_img = np.zeros((self.img_section.shape[0], self.img_section.shape[1], 3), dtype=np.uint8)
        overlapped_img[:,:,0] = line09*255 # Red
        overlapped_img[:,:,1] = np.zeros(self.img_section.shape) # Green
        overlapped_img[:,:,2] = np.zeros(self.img_section.shape) # Blue

        plt.imshow(self.img_section, 'gray', alpha=0.7)
        plt.imshow(overlapped_img, alpha=0.7)
        plt.title('Detected Mineral lines')
        plt.savefig(path2save)
        plt.close()