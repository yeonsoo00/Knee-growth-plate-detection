import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter

def roi_ap(img):
    # input : image np array
    # output : roi image np array

    intensity = np.sum(img, axis=1)
    peaks, _ = find_peaks(intensity, height=0, distance=2000)
    ordered_peaks = sorted(peaks, key=lambda peak: intensity[peak], reverse=True)[:2]
    x = range(len(intensity))


    selected_peak = min(ordered_peaks[0], ordered_peaks[1])
    upper_bound = int(x[selected_peak]*0.8) 
    lower_bound = x[ordered_peaks[1]]+1500 
    img_section = img[upper_bound:lower_bound, :]

    return img_section

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

class ap:
    """
    AP img : color channel : 5
    Lines : 16D0BAP, 08D0CAP
    """
    def __init__(self, path2img):
        self.img = np.array(plt.imread(path2img))
        self.img_section = roi_ap(self.img)

    def get_mask16(self):
        blurred = cv2.GaussianBlur(self.img_section, (7,7), 0)
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

        magnitude = cv2.magnitude(sobelx, sobely)
        magnitude = cv2.convertScaleAbs(magnitude)

        # Threshold the image to get edges
        threshold = 200  # You can adjust this threshold value
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

        # Find contours
        contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Select the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        largest_contour_mask = np.zeros_like(closed_img)
        cv2.drawContours(largest_contour_mask, [largest_contour], -1, 255, -1)

        # Erode & Gaussian filter
        kernel=np.ones((5,5))
        eroded_largest_contour_mask = cv2.erode(largest_contour_mask, kernel, iterations=7)
        mask16_=gaussian_filter(eroded_largest_contour_mask, sigma=70, mode="wrap")
        mask16 = np.where(mask16_>255*0.5, 255, 0)

        return mask16
    def get_mask08(self):
        gray_filtered = cv2.inRange(self.img_section, 15, 60)
        contours, _ = cv2.findContours(gray_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        largest_contour_mask2 = np.zeros_like(gray_filtered)
        cv2.drawContours(largest_contour_mask2, [largest_contour], -1, 255, -1)
        kernel = np.ones((100,100), np.uint8)
        closed_largest_contour_mask2 = cv2.morphologyEx(largest_contour_mask2, cv2.MORPH_CLOSE, kernel)

        # Gaussian filter
        mask08_=gaussian_filter(closed_largest_contour_mask2, sigma=150, mode="wrap") #########
        mask08 = np.where(mask08_>255*0.5, 255, 0)
        
        return mask08
    
    def plot_img(self, path2save, dilation=True):
        mask16 = ap.get_mask16()
        mask08 = ap.get_mask08()
        line16 = draw_line(mask16)
        line08 = draw_line(mask08)

        if dilation:
            line16 = cv2.dilate(line16, kernel=np.ones((7,7), np.uint8))
            line08 = cv2.dilate(line08, kernel=np.ones((7,7), np.uint8))

        overlapped_img = np.zeros((self.img_section.shape[0], self.img_section.shape[1], 3), dtype=np.uint8)
        overlapped_img[:,:,0] = line16*255 # Red
        overlapped_img[:,:,1] = line08*255 # Green
        overlapped_img[:,:,2] = np.zeros(self.img_section.shape) # Blue

        plt.imshow(self.img_section, 'gray', alpha=0.7)
        plt.imshow(overlapped_img, alpha=0.7)
        plt.title('Detected AP lines')
        plt.savefig(path2save)
        plt.close()