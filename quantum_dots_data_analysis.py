import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import skimage.filters
import time
import scipy.stats as stats

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
#
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
x_label = ["365nm autofluorescence signal", "280nm autofluorescence signal"]
x_pos = np.arange(len(x_label))

means_p1 = [bg_365_signal_mean_p1, bg_280_signal_mean_p1]
std_devs_p1 = [np.sqrt(np.var(bg_365_signal_p1)), np.sqrt(np.var(bg_280_signal_p1))]
fig1, ax1 = plt.subplots()
ax1.bar("365 nm excitation", means_p1[0], yerr=std_devs_p1[0], label = "365 nm mean = %.3f" %bg_365_signal_mean_p1)
ax1.bar("280 nm excitation", means_p1[1], yerr=std_devs_p1[1], label = "280 nm mean = %.3f" %bg_280_signal_mean_p1)
ax1.legend(loc="upper left")
ax1.set_ylim([0, 300])
ax1.set_xticklabels(x_label)

means_p2 = [bg_365_signal_mean_p2, bg_280_signal_mean_p2]
std_devs_p2 = [np.sqrt(np.var(bg_365_signal_p2)), np.sqrt(np.var(bg_280_signal_p2))]
fig2, ax2 = plt.subplots()
ax2.bar("365 nm excitation", means_p2[0], yerr=std_devs_p2[0], label = "365 nm mean = %.3f" %bg_365_signal_mean_p2)
ax2.bar("280 nm excitation", means_p2[1], yerr=std_devs_p2[1], label = "280 nm mean = %.3f" %bg_280_signal_mean_p2)
ax2.legend(loc="upper left")
ax2.set_ylim([0, 300])
ax2.set_xticklabels(x_label)

if save_figures:
    if timestamps:
        fig1.savefig("output_figures\\mean_intensity_comparison_image_pair1_%i%i%i_%i%i.tif"
                    % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                       time.gmtime()[4]))
        fig2.savefig("output_figures\\mean_intensity_comparison_image_pair2_%i%i%i_%i%i.tif"
                    % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                       time.gmtime()[4]))
    else:
        fig1.savefig("output_figures\\mean_intensity_comparison_image_pair1.tif")
        fig2.savefig("output_figures\\mean_intensity_comparison_image_pair2.tif")

plt.show()
