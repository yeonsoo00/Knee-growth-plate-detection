import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from ac import *
from mineral import *
from trap import *
from ap import *

if __name__ == '__main__':

    path2ac = '/home/yec23006/projects/research/KneeGrowthPlate/Knee_GrowthPlate/Images/CCC_K05_hK_FL1_s1_1_shift3.jpg'
    ac.plot_img(path2img=path2ac, path2save=)
    
    path2mineral = '/home/yec23006/projects/research/KneeGrowthPlate/Knee_GrowthPlate/Images/CCC_K05_hK_FL1_s1_2_shift3.jpg'
    mineral.plot_img(path2img=path2mineral, path2save=)

    path2trap = '/home/yec23006/projects/research/KneeGrowthPlate/Knee_GrowthPlate/Images/CCC_K05_hK_FL1_s1_3_shift3.jpg'
    trap.plot_img(path2img=path2trap, path2save=)

    path2ap = '/home/yec23006/projects/research/KneeGrowthPlate/Knee_GrowthPlate/Images/CCC_K05_hK_FL1_s1_5_shift3.jpg'
    ap.plot_img(path2img=path2ap, path2save=)