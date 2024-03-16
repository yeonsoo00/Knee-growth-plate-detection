import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter

def roi_trap(img):
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

class trap():
    """
    Color channel : 3
    Lines : 15TRPLL and 07TRPUL
    """
    def __init__(self, path2img):
        self.img = np.array(plt.imread(path2img))
        self.img_section = roi_trap(self.img)
    
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

    def get_masks(self):
        blurred = cv2.GaussianBlur(self.img_section, (7,7), 0)

        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

        magnitude = cv2.magnitude(sobelx, sobely)
        magnitude = cv2.convertScaleAbs(magnitude)

        threshold = 200
        edges = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)[1]

        kernel1 = np.ones((5,5), np.uint8)
        kernel2 = np.ones((2,2), np.uint8)
        dilated_image = cv2.dilate(edges, kernel1, iterations=7)
        closed_img = cv2.erode(dilated_image, kernel2, iterations=7)

        # Find contours
        contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areaArray = []
        for c in contours:
            area = cv2.contourArea(c)
            areaArray.append(area)

        sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

        # Select the 1st and 2nd largest contour
        largest_contour_1 = max(contours, key=cv2.contourArea)
        largest_contour_2 = sorteddata[1][1]

        # Create a mask for the contours
        largest_contour_mask = np.zeros_like(closed_img)
        cv2.drawContours(largest_contour_mask, [largest_contour_1], -1, 255, -1)
        largest_contour_mask2 = np.zeros_like(closed_img)
        cv2.drawContours(largest_contour_mask2, [largest_contour_2], -1, 255, -1)
        
        # Gaussian filter
        mask15_=gaussian_filter(largest_contour_mask, sigma=70, mode="wrap")
        mask15 = np.where(mask15_>255*0.5, 255, 0)
        mask07_=gaussian_filter(largest_contour_mask2, sigma=70, mode="wrap")
        mask07 = np.where(mask07_>255*0.5, 255, 0)

        return mask15, mask07
    
    def get_line1507(self):
        mask15, mask07 = trap.get_masks()

        # Apply gaussian filter
        kernel = np.ones((80,80), np.uint8)
        closed_mask07= cv2.morphologyEx(mask07, cv2.MORPH_CLOSE, kernel)
        closed_mask15= cv2.morphologyEx(mask15, cv2.MORPH_CLOSE, kernel)

        line07 = trap.draw_line(closed_mask07)
        line15 = trap.draw_line(closed_mask15)

        
        
        return line07, line15
    
    def plot_img(self,  path2save, dilation=True):
        line07, line15 = trap.get_line()

        if dilation:
            line07=cv2.dilate(line07, kernel=np.ones((7,7), np.uint8))
            line15=cv2.dilate(line15, kernel=np.ones((7,7), np.uint8))

        overlapped_img = np.zeros((self.img_section.shape[0], self.img_section.shape[1], 3), dtype=np.uint8)
        overlapped_img[:,:,0] = line07*255 # Red
        overlapped_img[:,:,1] = line15*255 # Green
        overlapped_img[:,:,2] = np.zeros(self.img_section.shape) # Blue

        plt.imshow(self.img_section, 'gray', alpha=0.7)
        plt.imshow(overlapped_img, alpha=0.7)
        plt.title('Detected TRAP lines')
        plt.savefig(path2save)
        plt.close()
