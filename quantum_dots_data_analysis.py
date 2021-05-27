import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import skimage.filters
import time
import scipy.stats as stats
import pandas as pd
from sklearn import preprocessing
import math
import scipy.optimize
from scipy import interpolate

save_figures = True
timestamps = False

# Load the raw, background and autofluorescent data images
raw_280_p1 = np.asarray(Image.open('data\\280 QD p1.tif'), dtype=np.int16)
raw_280_p2 = np.asarray(Image.open('data\\280 QD p2.tif'), dtype=np.int16)

raw_365_p1 = np.asarray(Image.open('data\\365 QD p1.tif'), dtype=np.int16)
raw_365_p2 = np.asarray(Image.open('data\\365 QD p2.tif'), dtype=np.int16)

background_280_p1 = np.asarray(Image.open('data\\280 bg p1.tif'), dtype=np.int16)
background_280_p2 = np.asarray(Image.open('data\\280 bg p2.tif'), dtype=np.int16)

background_365_p1 = np.asarray(Image.open('data\\365 bg p1.tif'), dtype=np.int16)
background_365_p2 = np.asarray(Image.open('data\\365 bg p2.tif'), dtype=np.int16)

auto_280_p1 = np.asarray(Image.open('data\\280 auto p1.tif'), dtype=np.int16)
auto_280_p2 = np.asarray(Image.open('data\\280 auto p2.tif'), dtype=np.int16)

auto_365_p1 = np.asarray(Image.open('data\\365 auto p1.tif'), dtype=np.int16)
auto_365_p2 = np.asarray(Image.open('data\\365 auto p2.tif'), dtype=np.int16)

auto_280_bg_p1 = np.asarray(Image.open('data\\280 auto bg p1.tif'), dtype=np.int16)
auto_280_bg_p2 = np.asarray(Image.open('data\\280 auto bg p2.tif'), dtype=np.int16)

auto_365_bg_p1 = np.asarray(Image.open('data\\365 auto bg p1.tif'), dtype=np.int16)
auto_365_bg_p2 = np.asarray(Image.open('data\\365 auto bg p2.tif'), dtype=np.int16)

print("~~~~~~~~~~~ Image pair analysis results ~~~~~~~~~~~")

# Do background correction
background_280_mean_p1 = np.mean(background_280_p1)
background_280_mean_p2 = np.mean(background_280_p2)
background_365_mean_p1 = np.mean(background_365_p1)
background_365_mean_p2 = np.mean(background_365_p2)

background_280_auto_mean_p1 = np.mean(auto_280_bg_p1)
background_280_auto_mean_p2 = np.mean(auto_280_bg_p2)
background_365_auto_mean_p1 = np.mean(auto_365_bg_p1)
background_365_auto_mean_p2 = np.mean(auto_365_bg_p2)

bg_280_corrected_p1 = raw_280_p1 - background_280_mean_p1
bg_280_corrected_p2 = raw_280_p2 - background_280_mean_p2
bg_365_corrected_p1 = raw_365_p1 - background_365_mean_p1
bg_365_corrected_p2 = raw_365_p2 - background_365_mean_p2

# Analyse autoflouresent signal distributions at 280nm and 365nm excitation
corrected_280_auto_p1 = auto_280_p1 - background_280_auto_mean_p1
corrected_280_auto_p2 = auto_280_p2 - background_280_auto_mean_p2
corrected_365_auto_p1 = auto_365_p1 - background_365_auto_mean_p1
corrected_365_auto_p2 = auto_365_p2 - background_365_auto_mean_p2

auto_280_triangle_thresh_p1 = skimage.filters.threshold_triangle(corrected_280_auto_p1)
auto_280_triangle_thresh_p2 = skimage.filters.threshold_triangle(corrected_280_auto_p2)
auto_365_triangle_thresh_p1 = skimage.filters.threshold_triangle(corrected_365_auto_p1)
auto_365_triangle_thresh_p2 = skimage.filters.threshold_triangle(corrected_365_auto_p2)

auto_280_triangle_p1 = corrected_280_auto_p1 > auto_280_triangle_thresh_p1
auto_280_triangle_p2 = corrected_280_auto_p2 > auto_280_triangle_thresh_p2
auto_365_triangle_p1 = corrected_365_auto_p1 > auto_365_triangle_thresh_p1
auto_365_triangle_p2 = corrected_365_auto_p2 > auto_365_triangle_thresh_p2

auto_280_signal_p1 = corrected_280_auto_p1[auto_280_triangle_p1]
auto_280_signal_p1[np.where(auto_280_signal_p1 < 0)[0]] = 0
auto_280_signal_p2 = corrected_280_auto_p2[auto_280_triangle_p2]
auto_280_signal_p2[np.where(auto_280_signal_p2 < 0)[0]] = 0

auto_365_signal_p1 = corrected_365_auto_p1[auto_280_triangle_p1]
auto_365_signal_p1[np.where(auto_365_signal_p1 < 0)[0]] = 0
auto_365_signal_p2 = corrected_365_auto_p2[auto_280_triangle_p2]
auto_365_signal_p2[np.where(auto_365_signal_p2 < 0)[0]] = 0

mean_auto_280_p1 = np.mean(auto_280_signal_p1)
mean_auto_280_p2 = np.mean(auto_280_signal_p2)
mean_auto_365_p1 = np.mean(auto_365_signal_p1)
mean_auto_365_p2 = np.mean(auto_365_signal_p2)

# t_stat_signal_p1, p_value_signal_p1 = stats.ttest_ind(auto_280_signal_p1, auto_365_signal_p1, equal_var=False)
# print("T-statistic of two-tailed t-test for the 280nm and 365nm 1st autofluorescent image pair signal distributions is",
#       " %.5f with a p-value %.5f" % (t_stat_signal_p1, p_value_signal_p1))
#
# t_stat_signal_p2, p_value_signal_p2 = stats.ttest_ind(auto_280_signal_p2, auto_365_signal_p2, equal_var=False)
# print("T-statistic of two-tailed t-test for the 280nm and 365nm 2nd autofluorescent image pair signal distributions is",
#       " %.5f with a p-value %.5f" % (t_stat_signal_p2, p_value_signal_p2))

# Generate the binary masks for selecting the cell regions of interest
bg_280_otsu_thresh_p1 = skimage.filters.threshold_otsu(bg_280_corrected_p1)
bg_280_otsu_thresh_p2 = skimage.filters.threshold_otsu(bg_280_corrected_p2)
bg_365_otsu_thresh_p1 = skimage.filters.threshold_otsu(bg_365_corrected_p1)
bg_365_otsu_thresh_p2 = skimage.filters.threshold_otsu(bg_365_corrected_p2)

bg_280_otsu_p1 = bg_280_corrected_p1 > bg_280_otsu_thresh_p1
bg_280_otsu_p2 = bg_280_corrected_p2 > bg_280_otsu_thresh_p2
bg_365_otsu_p1 = bg_365_corrected_p1 > bg_365_otsu_thresh_p1
bg_365_otsu_p2 = bg_365_corrected_p2 > bg_365_otsu_thresh_p2

# Mask out the regions of interest in the imaging data
bg_280_masked_p1 = bg_280_corrected_p1 * bg_280_otsu_p1
bg_280_masked_p2 = bg_280_corrected_p2 * bg_280_otsu_p2
bg_365_masked_p1 = bg_365_corrected_p1 * bg_280_otsu_p1
bg_365_masked_p2 = bg_365_corrected_p2 * bg_280_otsu_p2

# Analyse background corrected signal distributions at 280nm and 365nm excitation
bg_280_signal_p1 = bg_280_corrected_p1[bg_280_otsu_p1] - mean_auto_280_p1
bg_280_signal_p2 = bg_280_corrected_p2[bg_280_otsu_p2] - mean_auto_280_p2
bg_365_signal_p1 = bg_365_corrected_p1[bg_280_otsu_p1] - mean_auto_365_p1
bg_365_signal_p2 = bg_365_corrected_p2[bg_280_otsu_p2] - mean_auto_365_p2

# Assuming the 280nm and 365nm signals are normally distributed, acquire the relevant parameters to reconstruct the
# intensity distributions
bg_280_signal_mean_p1 = np.mean(bg_280_signal_p1)
bg_280_signal_std_dev_p1 = np.sqrt(np.var(bg_280_signal_p1))
print("Mean of 1st 280nm signal is %.5f with a standard deviation of %.5f" % (bg_280_signal_mean_p1, bg_280_signal_std_dev_p1))

bg_280_signal_mean_p2 = np.mean(bg_280_signal_p2)
bg_280_signal_std_dev_p2 = np.sqrt(np.var(bg_280_signal_p2))
print("Mean of 2nd 280nm signal is %.5f with a standard deviation of %.5f" % (bg_280_signal_mean_p2, bg_280_signal_std_dev_p2))

x_280_p1 = np.linspace(np.max([0, (bg_280_signal_mean_p1-(5*bg_280_signal_std_dev_p1))]),
                    (bg_280_signal_mean_p1+(5*bg_280_signal_std_dev_p1)), bg_280_signal_p1.shape[0])
x_280_p2 = np.linspace(np.max([0, (bg_280_signal_mean_p2-(5*bg_280_signal_std_dev_p2))]),
                    (bg_280_signal_mean_p2+(5*bg_280_signal_std_dev_p2)), bg_280_signal_p2.shape[0])

y_280_p1 = np.exp((-(x_280_p1 - bg_280_signal_mean_p1) ** 2) / (2 * bg_280_signal_std_dev_p1 ** 2))
y_280_p2 = np.exp((-(x_280_p2 - bg_280_signal_mean_p2) ** 2) / (2 * bg_280_signal_std_dev_p2 ** 2))

bg_365_signal_mean_p1 = np.mean(bg_365_signal_p1)
bg_365_signal_std_dev_p1 = np.sqrt(np.var(bg_365_signal_p1))
print("Mean of 1st 365nm signal is %.5f with a standard deviation of %.5f" % (bg_365_signal_mean_p1, bg_365_signal_std_dev_p1))

bg_365_signal_mean_p2 = np.mean(bg_365_signal_p2)
bg_365_signal_std_dev_p2 = np.sqrt(np.var(bg_365_signal_p2))
print("Mean of 2st 365nm signal is %.5f with a standard deviation of %.5f" % (bg_365_signal_mean_p2, bg_365_signal_std_dev_p2))

x_365_p1 = np.linspace(np.max([0, (bg_365_signal_mean_p1-(5*bg_365_signal_std_dev_p1))]),
                    (bg_365_signal_mean_p1 + (5*bg_365_signal_std_dev_p1)), bg_280_signal_p1.shape[0])
x_365_p2 = np.linspace(np.max([0, (bg_365_signal_mean_p2 - (5*bg_365_signal_std_dev_p2))]),
                    (bg_365_signal_mean_p2 + (5*bg_365_signal_std_dev_p2)), bg_280_signal_p2.shape[0])

y_365_p1 = np.exp((-(x_365_p1 - bg_365_signal_mean_p1) ** 2) / (2 * bg_365_signal_std_dev_p1 ** 2))
y_365_p2 = np.exp((-(x_365_p2 - bg_365_signal_mean_p2) ** 2) / (2 * bg_365_signal_std_dev_p2 ** 2))

t_stat_signal_p1, p_value_signal_p1 = stats.ttest_ind(bg_280_signal_p1, bg_365_signal_p1, equal_var=False)
print("T-statistic of two-tailed t-test for the 1st 280nm and 365nm image pair is %.5f with a p-value %.5f"
      % (t_stat_signal_p1, p_value_signal_p1))

t_stat_signal_p2, p_value_signal_p2 = stats.ttest_ind(bg_280_signal_p2, bg_365_signal_p2, equal_var=False)
print("T-statistic of two-tailed t-test for the 2nd 280nm and 365nm image pair is %.5f with a p-value %.5f"
      % (t_stat_signal_p2, p_value_signal_p2))

# Analyse the ratio of 280nm:365nm signal intensities

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Remove the pixels with 0 value in the 365nm signal data to ensure no divide by 0 errors in subsequent calculations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
bg_280_signal_no_zero_p1 = bg_280_signal_p1[np.where(bg_365_signal_p1 > 0)[0]]
bg_280_signal_no_zero_p2 = bg_280_signal_p2[np.where(bg_365_signal_p2 > 0)[0]]
bg_365_signal_no_zero_p1 = bg_365_signal_p1[np.where(bg_365_signal_p1 > 0)[0]]
bg_365_signal_no_zero_p2 = bg_365_signal_p2[np.where(bg_365_signal_p2 > 0)[0]]

bg_signal_ratio_p1 = bg_280_signal_no_zero_p1/bg_365_signal_no_zero_p1
bg_signal_ratio_p2 = bg_280_signal_no_zero_p2/bg_365_signal_no_zero_p2

print("Minimum 280nm:365nm signal ratio of the 1st image pair = %.5f" % np.min(bg_signal_ratio_p1))
print("Minimum 280nm:365nm signal ratio of the 2st image pair = %.5f" % np.min(bg_signal_ratio_p2))

# Assuming the 280nm:365nm signal ratio is normally distributed, acquire the relevant parameters to reconstruct the
# intensity ratio distribution
bg_signal_ratio_mean_p1 = np.mean(bg_signal_ratio_p1)
bg_signal_ratio_mean_p2 = np.mean(bg_signal_ratio_p2)
bg_signal_ratio_std_dev_p1 = np.sqrt(np.var(bg_signal_ratio_p1))
bg_signal_ratio_std_dev_p2 = np.sqrt(np.var(bg_signal_ratio_p2))

print("Mean 280nm:365nm ratio of the 1st image pair is %.5f with a standard deviation of %.5f"
      %(bg_signal_ratio_mean_p1, bg_signal_ratio_std_dev_p1))

print("Mean 280nm:365nm ratio of the 2nd image pair is %.5f with a standard deviation of %.5f"
      %(bg_signal_ratio_mean_p2, bg_signal_ratio_std_dev_p2))

x_p1 = np.linspace(np.max([np.min(bg_signal_ratio_p1), (bg_signal_ratio_mean_p1-(5*bg_signal_ratio_std_dev_p1))]),
                (bg_signal_ratio_mean_p1+(5*bg_signal_ratio_std_dev_p1)), 1000)
y_p1 = np.exp((-(x_p1 - bg_signal_ratio_mean_p1) ** 2) / (2 * bg_signal_ratio_std_dev_p1 ** 2))

x_p2 = np.linspace(np.max([np.min(bg_signal_ratio_p2), (bg_signal_ratio_mean_p2-(5*bg_signal_ratio_std_dev_p2))]),
                (bg_signal_ratio_mean_p2+(5*bg_signal_ratio_std_dev_p2)), 1000)
y_p2 = np.exp((-(x_p2 - bg_signal_ratio_mean_p2) ** 2) / (2 * bg_signal_ratio_std_dev_p2 ** 2))

# Repeat the analysis the ratio of 280nm:365nm signal intensities, this time with the 'outlier' data points removed.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Since the 280nm Otsu threshold mask will capture some pixels in the 365nm data which are basically noise, it is
# possible to get ratios which are extremely large (i.e. 200+) but these are not representative of signal-to-signal
# ratios between the two excitation wavelengths. Therefore, we seek to perform the previous statistical analysis while
# excluding these outliers. An 'outlier' intensity ratio is determined to be any ratio which is 5 standard deviations
# from the mean intensity ratio, since this is likely to be due to these noise divisions. Other methods of determining
# outliers are, of course, equally valid and should be specified here.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

outlier_thresh_p1 = bg_signal_ratio_mean_p1 + (5*bg_signal_ratio_std_dev_p1)
outliers_p1 = np.where(bg_signal_ratio_p1 > outlier_thresh_p1)
perc_outliers_p1 = (np.shape(outliers_p1[0])[0]/bg_signal_ratio_p1.shape[0])*100
print("Percentage of ratios in the 1st image pair which are outliers = %.5f%%" % perc_outliers_p1)

outlier_thresh_p2 = bg_signal_ratio_mean_p2 + (5*bg_signal_ratio_std_dev_p2)
outliers_p2 = np.where(bg_signal_ratio_p2 > outlier_thresh_p2)
perc_outliers_p2 = (np.shape(outliers_p2[0])[0]/bg_signal_ratio_p2.shape[0])*100
print("Percentage of ratios in the 2nd image pair which are outliers = %.5f%%" % perc_outliers_p2)

bg_signal_ratio_no_out_p1 = bg_signal_ratio_p1[np.where(bg_signal_ratio_p1 <= outlier_thresh_p1)]
bg_signal_ratio_no_out_p2 = bg_signal_ratio_p2[np.where(bg_signal_ratio_p2 <= outlier_thresh_p2)]

bg_signal_ratio_no_out_mean_p1 = np.mean(bg_signal_ratio_no_out_p1)
bg_signal_ratio_no_out_std_dev_p1 = np.sqrt(np.var(bg_signal_ratio_no_out_p1))
print("Mean 280nm:365nm ratio of the 1st image pair is %.5f with a standard deviation of %.5f"
      % (bg_signal_ratio_no_out_mean_p1, bg_signal_ratio_no_out_std_dev_p1))

bg_signal_ratio_no_out_mean_p2 = np.mean(bg_signal_ratio_no_out_p2)
bg_signal_ratio_no_out_std_dev_p2 = np.sqrt(np.var(bg_signal_ratio_no_out_p2))
print("Mean 280nm:365nm ratio of the 2st image pair is %.5f with a standard deviation of %.5f"
      % (bg_signal_ratio_no_out_mean_p2, bg_signal_ratio_no_out_std_dev_p2))

# Assuming the 280nm:365nm signal ratio without outliers is normally distributed, acquire the relevant parameters to
# reconstruct the intensity ratio distribution
x_out_p1 = np.linspace(np.max([np.min(bg_signal_ratio_p1), (bg_signal_ratio_no_out_mean_p1-(5*bg_signal_ratio_no_out_std_dev_p1))]),
                    (bg_signal_ratio_no_out_mean_p1+(5*bg_signal_ratio_no_out_std_dev_p1)), 1000)
x_out_p2 = np.linspace(np.max([np.min(bg_signal_ratio_p2), (bg_signal_ratio_no_out_mean_p2-(5*bg_signal_ratio_no_out_std_dev_p2))]),
                    (bg_signal_ratio_no_out_mean_p2+(5*bg_signal_ratio_no_out_std_dev_p2)), 1000)

y_out_p1 = np.exp((-(x_out_p1 - bg_signal_ratio_no_out_mean_p1) ** 2) / (2 * bg_signal_ratio_no_out_std_dev_p1 ** 2))
y_out_p2 = np.exp((-(x_out_p2 - bg_signal_ratio_no_out_mean_p2) ** 2) / (2 * bg_signal_ratio_no_out_std_dev_p2 ** 2))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This section is an analysis of the photobleaching of the quantum dots at both 280nm and 365nm excitation wavelengths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load in all the photo bleaching data

photobleaching_365_1 = pd.read_csv('data\\photobleaching_365_1.csv', engine='python')
photobleaching_365_2 = pd.read_csv('data\\photobleaching_365_2.csv', engine='python')
photobleaching_365_3 = pd.read_csv('data\\photobleaching_365_3.csv', engine='python')

photobleaching_280_1 = pd.read_csv('data\\photobleaching_280_1.csv', engine='python')
photobleaching_280_2 = pd.read_csv('data\\photobleaching_280_2.csv', engine='python')
photobleaching_280_3 = pd.read_csv('data\\photobleaching_280_1(1).csv', engine='python')

time_points_photobleaching = np.linspace(0, 8, 49)

intensity_1_365 = photobleaching_365_1['Mean']
normalised_intensity_1_365 = intensity_1_365/max(intensity_1_365)

intensity_2_365 = photobleaching_365_2['Mean']
normalised_intensity_2_365 = intensity_2_365/max(intensity_2_365)

intensity_3_365 = photobleaching_365_3['Mean']
normalised_intensity_3_365 = intensity_3_365/max(intensity_3_365)

normalised_intensities_365 = [normalised_intensity_1_365, normalised_intensity_2_365, normalised_intensity_3_365]

mean_365_photobleaching = np.mean(normalised_intensities_365, axis=0)
std_dev_365_photobleaching = np.std(normalised_intensities_365, axis=0)

intensity_1_280 = photobleaching_280_1['Mean']
normalised_intensity_1_280 = intensity_1_280/max(intensity_1_280)

intensity_2_280 = photobleaching_280_2['Mean']
normalised_intensity_2_280 = intensity_2_280/max(intensity_2_280)

intensity_3_280 = photobleaching_280_3['Mean']
normalised_intensity_3_280 = intensity_3_280/max(intensity_3_280)

normalised_intensities_280 = [normalised_intensity_1_280, normalised_intensity_2_280, normalised_intensity_3_280]

mean_280_photobleaching = np.mean(normalised_intensities_280, axis=0)
std_dev_280_photobleaching = np.std(normalised_intensities_280, axis=0)

intensity_280_change = (1 - (mean_280_photobleaching[-1]/mean_280_photobleaching[0]))*100
intensity_365_change = (1 - (mean_365_photobleaching[-1]/mean_365_photobleaching[0]))*100

print("~~~~~~~~~~~ Photobleaching experiments results ~~~~~~~~~~~")

if intensity_280_change < 0:
    print(
        "Mean emission intensity after %i hours of 280nm excitation increased by %f%%" % (time_points_photobleaching[-1],
                                                                                        abs(intensity_280_change)))
else:
    print(
        "Mean emission intensity after %i hours of 280nm excitation decreased by %f%%" % (time_points_photobleaching[-1],
                                                                                        abs(intensity_280_change)))

if intensity_365_change < 0:
    print(
        "Mean emission intensity after %i hours of 365nm excitation increased by %f%%" % (time_points_photobleaching[-1],
                                                                                        abs(intensity_365_change)))
else:
    print(
        "Mean emission intensity after %i hours of 365nm excitation decreased by %f%%" % (time_points_photobleaching[-1],
                                                                                        abs(intensity_365_change)))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This section is an analysis of cell viability at both 280nm and 365nm excitation wavelengths compared to control
# illumination.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("~~~~~~~~~~~ Cell viability experiments results ~~~~~~~~~~~")

time_points_viability = np.linspace(0, 9, 36)

viable_cells_control_1 = np.array([22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,21,20,19,15,15,14,14,14,10,7,5,5,4,3,2,1,0,0,0])
viable_cells_control_2 = np.array([24,24,24,24,24,24,24,24,24,24,23,23,23,23,23,23,23,23,22,21,18,15,15,15,14,10,9,8,7,7,6,4,4,2,1,1])
viable_cells_control_3 = np.array([30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,29,27,27,25,22,21,20,16,14,13,11,10,10,8,5,5,4,4,2])

viability_percentage_control_1 = (viable_cells_control_1/22)*100
viability_percentage_control_2 = (viable_cells_control_2/24)*100
viability_percentage_control_3 = (viable_cells_control_3/30)*100

a_con = [viability_percentage_control_1, viability_percentage_control_2, viability_percentage_control_3]
mean_con_viability  = np.mean(a_con, axis=0)
std_con_viability  = np.std(a_con, axis=0)

f_con = interpolate.UnivariateSpline(time_points_viability, mean_con_viability, s=0)
xnew_con = np.arange(0, 9, 100)

yToFind = 50
yreduced_con_viability = np.array(mean_con_viability ) - yToFind
freduced_con_viability  = interpolate.UnivariateSpline(time_points_viability, yreduced_con_viability, s=0)
tau_hours_con = freduced_con_viability.roots()

tau_con_mins = tau_hours_con*60

viable_cells_365_1 = np.array([36,36,36,36,36,36,36,36,36,36,36,34,27,21,10,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
viable_cells_365_2 = np.array([22,22,22,22,22,22,22,22,22,17,11,3,3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
viable_cells_365_3 = np.array([24,24,24,24,24,24,24,24,23,19,15,5,3,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

viability_percentage_365_1 = (viable_cells_365_1/36)*100
viability_percentage_365_2 = (viable_cells_365_2/22)*100
viability_percentage_365_3 = (viable_cells_365_3/24)*100

a_365 = [viability_percentage_365_1, viability_percentage_365_2, viability_percentage_365_3]

mean_365_viability = np.mean(a_365, axis=0)
std_dev_365_viability = np.std(a_365, axis=0)

f_365 = interpolate.UnivariateSpline(time_points_viability, mean_365_viability, s=0)
xnew_365 = np.arange(0, 9, 100)

yreduced_365 = np.array(mean_365_viability) - yToFind
freduced_365 = interpolate.UnivariateSpline(time_points_viability, yreduced_365, s=0)
tau_hours_365 = freduced_365.roots()

tau_365_mins = tau_hours_365*60

viable_cells_280_1 = np.array([44,44,20,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
viability_percentage_280_1 = (viable_cells_280_1/44)*100

viable_cells_280_2 = np.array([36,34,33,26,23,8,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
viability_percentage_280_2 = (viable_cells_280_2/36)*100

viable_cells_280_3 = np.array([38,38,36,21,6,5,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
viability_percentage_280_3 = (viable_cells_280_3/38)*100

a_280 = [viability_percentage_280_1, viability_percentage_280_2, viability_percentage_280_3]

mean_280_viability = np.mean(a_280, axis=0)
std_280_viability = np.std(a_280, axis=0)

f_280 = interpolate.UnivariateSpline(time_points_viability, mean_280_viability, s=0)
xnew_280 = np.arange(0, 9, 100)

# To find x at y then do:

yreduced_280 = np.array(mean_280_viability) - yToFind
freduced_280 = interpolate.UnivariateSpline(time_points_viability, yreduced_280, s=0)
tau_hours_280 = freduced_280.roots()

tau_280_mins = tau_hours_280*60

print("50%% of cells illuminated with control illumination remain alive after %i minutes" % tau_con_mins)
print("50%% of cells illuminated with 365nm illumination remain alive after %i minutes" % tau_365_mins)
print("50%% of cells illuminated with 280nm illumination remain alive after %i minutes" % tau_280_mins)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This section saves all the various images and figures used.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Save the generated image data with or without timestamps according to the flags specified at the top of this file.
if save_figures:
    bg_280_corrected_im_p1 = Image.fromarray(bg_280_corrected_p1)
    bg_365_corrected_im_p1 = Image.fromarray(bg_365_corrected_p1)
    bg_280_otsu_im_p1 = Image.fromarray(bg_280_otsu_p1)
    bg_365_otsu_im_p1 = Image.fromarray(bg_365_otsu_p1)
    bg_280_masked_im_p1 = Image.fromarray(bg_280_masked_p1)
    bg_365_masked_im_p1 = Image.fromarray(bg_365_masked_p1)
    bg_280_triangle_im_p1 = Image.fromarray(auto_280_triangle_p1)
    bg_365_triangle_im_p1 = Image.fromarray(auto_365_triangle_p1)

    bg_280_corrected_im_p2 = Image.fromarray(bg_280_corrected_p2)
    bg_365_corrected_im_p2 = Image.fromarray(bg_365_corrected_p2)
    bg_280_otsu_im_p2 = Image.fromarray(bg_280_otsu_p2)
    bg_365_otsu_im_p2 = Image.fromarray(bg_365_otsu_p2)
    bg_280_masked_im_p2 = Image.fromarray(bg_280_masked_p2)
    bg_365_masked_im_p2 = Image.fromarray(bg_365_masked_p2)
    bg_280_triangle_im_p2 = Image.fromarray(auto_280_triangle_p2)
    bg_365_triangle_im_p2 = Image.fromarray(auto_365_triangle_p2)
    if timestamps:
        bg_280_corrected_im_p1.save("output_figures\\280_bg_corrected_p1_%i%i%i_%i%i.tif"
                                 % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                                    time.gmtime()[4]))
        bg_365_corrected_im_p1.save("output_figures\\365_bg_corrected_p1_%i%i%i_%i%i.tif"
                                 % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                                    time.gmtime()[4]))
        bg_280_otsu_im_p1.save("output_figures\\280_otsu_mask_p1_%i%i%i_%i%i.tif"
                            % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                               time.gmtime()[4]))
        bg_365_otsu_im_p1.save("output_figures\\365_otsu_mask_p1_%i%i%i_%i%i.tif"
                            % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                               time.gmtime()[4]))
        bg_280_masked_im_p1.save("output_figures\\280_bg_corrected_w_otsu_mask_p1_%i%i%i_%i%i.tif"
                              % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                                 time.gmtime()[4]))
        bg_365_masked_im_p1.save("output_figures\\365_bg_corrected_w_otsu_mask_p1_%i%i%i_%i%i.tif"
                              % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                                 time.gmtime()[4]))
        bg_280_triangle_im_p1.save("output_figures\\280_auto_triangle_mask_p1_%i%i%i_%i%i.tif"
                                % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                                   time.gmtime()[4]))
        bg_365_triangle_im_p1.save("output_figures\\365_auto_triangle_mask_p1_%i%i%i_%i%i.tif"
                                % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                                   time.gmtime()[4]))

        bg_280_corrected_im_p2.save("output_figures\\280_bg_corrected_p2_%i%i%i_%i%i.tif"
                                    % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                                       time.gmtime()[4]))
        bg_365_corrected_im_p2.save("output_figures\\365_bg_corrected_p2_%i%i%i_%i%i.tif"
                                    % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                                       time.gmtime()[4]))
        bg_280_otsu_im_p2.save("output_figures\\280_otsu_mask_p2_%i%i%i_%i%i.tif"
                               % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                                  time.gmtime()[4]))
        bg_365_otsu_im_p2.save("output_figures\\365_otsu_mask_p2_%i%i%i_%i%i.tif"
                               % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                                  time.gmtime()[4]))
        bg_280_masked_im_p2.save("output_figures\\280_bg_corrected_w_otsu_mask_p2_%i%i%i_%i%i.tif"
                                 % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                                    time.gmtime()[4]))
        bg_365_masked_im_p2.save("output_figures\\365_bg_corrected_w_otsu_mask_p2_%i%i%i_%i%i.tif"
                                 % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                                    time.gmtime()[4]))
        bg_280_triangle_im_p2.save("output_figures\\280_auto_triangle_mask_p2_%i%i%i_%i%i.tif"
                                   % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                                      time.gmtime()[4]))
        bg_365_triangle_im_p2.save("output_figures\\365_auto_triangle_mask_p2_%i%i%i_%i%i.tif"
                                   % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                                      time.gmtime()[4]))
    else:
        bg_280_corrected_im_p1.save("output_figures\\280_bg_corrected_p1.tif")
        bg_365_corrected_im_p1.save("output_figures\\365_bg_corrected_p1.tif")
        bg_280_otsu_im_p1.save("output_figures\\280_otsu_mask_p1.tif")
        bg_365_otsu_im_p1.save("output_figures\\365_otsu_mask_p1.tif")
        bg_280_masked_im_p1.save("output_figures\\280_bg_corrected_w_otsu_mask_p1.tif")
        bg_365_masked_im_p1.save("output_figures\\365_bg_corrected_w_otsu_mask_p1.tif")
        bg_280_triangle_im_p1.save("output_figures\\280_auto_triangle_mask_p1.tif")
        bg_365_triangle_im_p1.save("output_figures\\365_auto_triangle_mask_p1.tif")

        bg_280_corrected_im_p2.save("output_figures\\280_bg_corrected_p2.tif")
        bg_365_corrected_im_p2.save("output_figures\\365_bg_corrected_p2.tif")
        bg_280_otsu_im_p2.save("output_figures\\280_otsu_mask_p2.tif")
        bg_365_otsu_im_p2.save("output_figures\\365_otsu_mask_p2.tif")
        bg_280_masked_im_p2.save("output_figures\\280_bg_corrected_w_otsu_mask_p2.tif")
        bg_365_masked_im_p2.save("output_figures\\365_bg_corrected_w_otsu_mask_p2.tif")
        bg_280_triangle_im_p2.save("output_figures\\280_auto_triangle_mask_p2.tif")
        bg_365_triangle_im_p2.save("output_figures\\365_auto_triangle_mask_p2.tif")

# Generate the desired data plots and save them with or without timestamps according to the flags specified at the top
# of this file.
x_label = ["365 nm excitation", "280 nm excitation"]
x_pos = np.arange(len(x_label))

props = {'connectionstyle': 'bar', 'arrowstyle': '-', 'shrinkA': 20, 'shrinkB': 20, 'linewidth': 2}
maxasterix = 5
p = .05

data_p1 = p_value_signal_p1
text_p1 = ''
while data_p1 < p:
    text_p1 += '*'
    p /= 10.
    if maxasterix and len(text_p1) == maxasterix:
        break

means_p1 = [bg_365_signal_mean_p1, bg_280_signal_mean_p1]
std_devs_p1 = [np.sqrt(np.var(bg_365_signal_p1)), np.sqrt(np.var(bg_280_signal_p1))]
fig1, ax1 = plt.subplots()
ax1.bar("365 nm excitation", means_p1[0], yerr=std_devs_p1[0], label = "365 nm mean = %.3f" %bg_365_signal_mean_p1)
ax1.bar("280 nm excitation", means_p1[1], yerr=std_devs_p1[1], label = "280 nm mean = %.3f" %bg_280_signal_mean_p1)
ax1.annotate(text_p1, xy=(0.45, (1.43*means_p1[1])+std_devs_p1[1]), zorder=10)
ax1.annotate('', xy=(0, means_p1[1]+(0.80*std_devs_p1[1])),
             xytext=(1, means_p1[1]+(0.80*std_devs_p1[1])), arrowprops=props)
ax1.legend(loc="upper left")
ax1.set_ylim([0, 450])
ax1.set_xticklabels(x_label)
ax1.set_ylabel("Intensity")

data_p2 = p_value_signal_p2
text_p2 = ''
while data_p2 < p:
    text_p2 += '*'
    p /= 10.
    if maxasterix and len(text_p2) == maxasterix:
        break

means_p2 = [bg_365_signal_mean_p2, bg_280_signal_mean_p2]
std_devs_p2 = [np.sqrt(np.var(bg_365_signal_p2)), np.sqrt(np.var(bg_280_signal_p2))]
fig2, ax2 = plt.subplots()
ax2.bar("365 nm excitation", means_p2[0], yerr=std_devs_p2[0], label = "365 nm mean = %.3f" %bg_365_signal_mean_p2)
ax2.bar("280 nm excitation", means_p2[1], yerr=std_devs_p2[1], label = "280 nm mean = %.3f" %bg_280_signal_mean_p2)
ax2.annotate(text_p2, xy=(0.45, (1.55*means_p2[1])+std_devs_p2[1]), zorder=10)
ax2.annotate('', xy=(0, means_p2[1]+(0.75*std_devs_p2[1])),
             xytext=(1, means_p2[1]+(0.75*std_devs_p2[1])), arrowprops=props)
ax2.legend(loc="upper left")
ax2.set_ylim([0, 450])
ax2.set_xticklabels(x_label)
ax2.set_ylabel("Intensity")

if save_figures:
    if timestamps:
        fig1.savefig("output_figures\\mean_intensity_comparison_image_pair1_%i%i%i_%i%i.tif"
                    % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                       time.gmtime()[4]))
        fig2.savefig("output_figures\\mean_normalised_intensity_comparison_image_pair2_%i%i%i_%i%i.tif"
                     % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                        time.gmtime()[4]))
    else:
        fig1.savefig("output_figures\\mean_intensity_comparison_image_pair1.tif")
        fig2.savefig("output_figures\\mean_intensity_comparison_image_pair2.tif")

plt.figure(3)
# plt.plot(time_points_photobleaching, mean_280, color="orange", label="280 nm", linewidth=3, alpha=0.5)
# plt.plot(time_points_photobleaching, mean_365, color="blue", label="365 nm", linewidth=3, alpha=0.5)
plt.errorbar(time_points_photobleaching, mean_365_photobleaching, color="blue", label="365 nm", xerr=0,
             yerr=std_dev_365_photobleaching)
plt.errorbar(time_points_photobleaching, mean_280_photobleaching, color="orange", label="280 nm", xerr=0,
             yerr=std_dev_280_photobleaching)
plt.xlim([0, 8])
plt.ylim([0.9, 1.025])
plt.xlabel("Time (Hours)")
plt.ylabel("Normalised Intensity")
plt.legend()

if save_figures:
    if timestamps:
        plt.savefig("output_figures\\photobleaching_%i%i%i_%i%i.png"
                    % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                       time.gmtime()[4]))
    else:
        plt.savefig("output_figures\\photobleaching.png")

plt.figure(4)
plt.bar(time_points_viability, mean_con_viability, width=0.2, alpha=1, yerr=std_con_viability, color='mediumseagreen',
        label="Control")
plt.bar(time_points_viability, mean_365_viability, width=0.2, alpha=1, yerr=std_dev_365_viability, color='steelblue',
        label="365 nm irradiation")
plt.bar(time_points_viability, mean_280_viability, width=0.2, alpha=1, yerr=std_280_viability, color='darkorange',
        label="280 nm irradiation")
plt.ylim(0, 101)
plt.xlim(-0.1, 8.25)
plt.legend()
plt.xlabel("Time (Hours)")
plt.ylabel("Number of viable cells(%)")

if save_figures:
    if timestamps:
        plt.savefig("output_figures\\cell_viability_%i%i%i_%i%i.png"
                    % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                       time.gmtime()[4]))
    else:
        plt.savefig("output_figures\\cell_viability.png")

plt.figure(5)
plt.bar("Control", tau_con_mins, label="50% cell death at 380 minutes", color="mediumseagreen" )
plt.bar("365 nm Irradiation", tau_365_mins, label="50% cell death at 165 minutes", color="steelblue")
plt.bar("280 nm Irradiation", tau_280_mins, label="50% cell death at 43 minutes", color="darkorange")
plt.legend()
plt.ylabel("Time at 50% Cell Death (Minutes)")

plt.show()


