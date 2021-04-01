import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import skimage.filters
import time
import scipy.stats as stats

save_figures = True
timestamps = False

# Load the raw, background and autofluorescent data images
raw_280_load = Image.open('images\\280 QD.tif')
raw_280 = np.asarray(raw_280_load, dtype=np.int16)

raw_365_load = Image.open('images\\365 QD.tif')
raw_365 = np.asarray(raw_365_load, dtype=np.int16)

background_280_load = Image.open('images\\280 bg.tif')
background_280 = np.asarray(background_280_load, dtype=np.int16)

background_365_load = Image.open('images\\365 bg.tif')
background_365 = np.asarray(background_365_load, dtype=np.int16)

auto_280_load = Image.open('images\\280 auto.tif')
auto_280 = np.asarray(auto_280_load, dtype=np.int16)

auto_365_load = Image.open('images\\365 auto.tif')
auto_365 = np.asarray(auto_365_load, dtype=np.int16)

# Do background correction
background_280_mean = np.mean(background_280)
background_365_mean = np.mean(background_365)

bg_280_corrected = raw_280 - background_280_mean
bg_365_corrected = raw_365 - background_365_mean

# Analyse autoflouresent signal distributions at 280nm and 365nm excitation
corrected_280_auto = auto_280 - background_280_mean
corrected_365_auto = auto_365 - background_365_mean

auto_280_triangle_thresh = skimage.filters.threshold_triangle(corrected_280_auto)
auto_365_triangle_thresh = skimage.filters.threshold_triangle(corrected_365_auto)

auto_280_triangle = corrected_280_auto > auto_280_triangle_thresh
auto_365_triangle = corrected_365_auto > auto_365_triangle_thresh

auto_280_signal = corrected_280_auto[auto_280_triangle]
auto_280_signal[np.where(auto_280_signal < 0)[0]] = 0
auto_365_signal = corrected_365_auto[auto_280_triangle]
auto_365_signal[np.where(auto_365_signal < 0)[0]] = 0

mean_auto_280 = np.mean(auto_280_signal)
mean_auto_365 = np.mean(auto_365_signal)

t_stat_signal, p_value_signal = stats.ttest_ind(auto_280_signal, auto_365_signal, equal_var=False)
print("T-statistic of two-tailed t-test for the 280nm and 365nm signal distributions is %.5f with a p-value %.5f"
      % (t_stat_signal, p_value_signal))

# Generate the binary masks for selecting the cell regions of interest
bg_280_otsu_thresh = skimage.filters.threshold_otsu(bg_280_corrected)
bg_365_otsu_thresh = skimage.filters.threshold_otsu(bg_365_corrected)

bg_280_otsu = bg_280_corrected > bg_280_otsu_thresh
bg_365_otsu = bg_365_corrected > bg_365_otsu_thresh

# Mask out the regions of interest in the imaging data
bg_280_masked = bg_280_corrected * bg_280_otsu
bg_365_masked = bg_365_corrected * bg_280_otsu

plt.figure(0)
plt.imshow(bg_280_masked, cmap='gray')
plt.figure(1)
plt.imshow(bg_365_masked, cmap='gray')
plt.show()

# Analyse background corrected signal distributions at 280nm and 365nm excitation
bg_280_signal = bg_280_corrected[bg_280_otsu] - mean_auto_280
bg_365_signal = bg_365_corrected[bg_280_otsu] - mean_auto_365

# Assuming the 280nm and 365nm signals are normally distributed, acquire the relevant parameters to reconstruct the
# intensity distributions
bg_280_signal_mean = np.mean(bg_280_signal)
bg_280_signal_std_dev = np.sqrt(np.var(bg_280_signal))
print("Mean of 280nm signal is %.5f with a standard deviation of %.5f" % (bg_280_signal_mean,bg_280_signal_std_dev))

x_280 = np.linspace(np.max([0, (bg_280_signal_mean-(5*bg_280_signal_std_dev))]),
                    (bg_280_signal_mean+(5*bg_280_signal_std_dev)), bg_280_signal.shape[0])

y_280 = np.exp((-(x_280 - bg_280_signal_mean) ** 2) / (2 * bg_280_signal_std_dev ** 2))

bg_365_signal_mean = np.mean(bg_365_signal)
bg_365_signal_std_dev = np.sqrt(np.var(bg_365_signal))

print("Mean of 365nm signal is %.5f with a standard deviation of %.5f" % (bg_365_signal_mean,bg_365_signal_std_dev))

x_365 = np.linspace(np.max([0, (bg_365_signal_mean-(5*bg_365_signal_std_dev))]),
                    (bg_365_signal_mean+(5*bg_365_signal_std_dev)), bg_280_signal.shape[0])
y_365 = np.exp((-(x_365 - bg_365_signal_mean) ** 2) / (2 * bg_365_signal_std_dev ** 2))

t_stat_signal, p_value_signal = stats.ttest_ind(bg_280_signal, bg_365_signal, equal_var=False)
print("T-statistic of two-tailed t-test for the 280nm and 365nm signal distributions is %.5f with a p-value %.5f"
      % (t_stat_signal, p_value_signal))

# Analyse the ratio of 280nm:365nm signal instenisties

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Remove the pixels with 0 value in the 365nm signal data to ensure no devide by 0 errors in subsequent calculations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
bg_280_signal_no_zero = bg_280_signal[np.where(bg_365_signal > 0)[0]]
bg_365_signal_no_zero = bg_365_signal[np.where(bg_365_signal > 0)[0]]

bg_signal_ratio = bg_280_signal_no_zero/bg_365_signal_no_zero

# Assuming the 280nm:365nm signal ratio is normally distributed, acquire the relevant parameters to reconstruct the
# intensity ratio distribution
bg_signal_ratio_mean = np.mean(bg_signal_ratio)
bg_signal_ratio_std_dev = np.sqrt(np.var(bg_signal_ratio))

print("Minimum 280nm:365nm signal ratio = %.5f" % np.min(bg_signal_ratio))

x = np.linspace(np.max([np.min(bg_signal_ratio), (bg_signal_ratio_mean-(5*bg_signal_ratio_std_dev))]),
                (bg_signal_ratio_mean+(5*bg_signal_ratio_std_dev)), 1000)

y = np.exp((-(x - bg_signal_ratio_mean) ** 2) / (2 * bg_signal_ratio_std_dev ** 2))

print("Mean ratio of 280nm:365nm signal is %.5f with a standard deviation of %.5f" % (bg_signal_ratio_mean,
                                                                                      bg_signal_ratio_std_dev))

# Repeat the analysis the ratio of 280nm:365nm signal intensities, this time with the 'outlier' data points removed.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Since the 280nm Otsu threshold mask will capture some pixels in the 365nm data which are basically noise, it is
# possible to get ratios which are extremely large (i.e. 200+) but these are not representative of signal-to-signal
# ratios between the two excitation wavelengths. Therefore, we seek to perform the previous statistical analysis while
# excluding these outliers. An 'outlier' intensity ratio is determined to be any ratio which is 5 standard deviations
# from the mean intensity ratio, since this is likely to be due to these noise divisions. Other methods of determining
# outliers are, of course, equally valid and should be specified here.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

outlier_thresh = bg_signal_ratio_mean + (5*bg_signal_ratio_std_dev)
outliers = np.where(bg_signal_ratio > outlier_thresh)
perc_outliers = (np.shape(outliers[0])[0]/bg_signal_ratio.shape[0])*100

print("Percentage of ratios which are outliers = %.5f%%" % perc_outliers)

bg_signal_ratio_no_out = bg_signal_ratio[np.where(bg_signal_ratio <= outlier_thresh)]

bg_signal_ratio_no_out_mean = np.mean(bg_signal_ratio_no_out)
bg_signal_ratio_no_out_std_dev = np.sqrt(np.var(bg_signal_ratio_no_out))

# Assuming the 280nm:365nm signal ratio without outliers is normally distributed, acquire the relevant parameters to
# reconstruct the intensity ratio distribution
x_out = np.linspace(np.max([np.min(bg_signal_ratio), (bg_signal_ratio_no_out_mean-(5*bg_signal_ratio_no_out_std_dev))]),
                    (bg_signal_ratio_no_out_mean+(5*bg_signal_ratio_no_out_std_dev)), 1000)

y_out = np.exp((-(x_out - bg_signal_ratio_no_out_mean) ** 2) / (2 * bg_signal_ratio_no_out_std_dev ** 2))

print("Mean ratio of 280nm:365nm signal is %.5f with a standard deviation of %.5f"
      % (bg_signal_ratio_no_out_mean, bg_signal_ratio_no_out_std_dev))

# Save the generated image data with or without timestamps according to the flags specified at the top of this file.
if save_figures:
    bg_280_corrected_im = Image.fromarray(bg_280_corrected)
    bg_365_corrected_im = Image.fromarray(bg_365_corrected)
    bg_280_otsu_im = Image.fromarray(bg_280_otsu)
    bg_365_otsu_im = Image.fromarray(bg_365_otsu)
    bg_280_masked_im = Image.fromarray(bg_280_masked)
    bg_365_masked_im = Image.fromarray(bg_365_masked)
    bg_280_triangle_im = Image.fromarray(auto_280_triangle)
    bg_365_triangle_im = Image.fromarray(auto_365_triangle)
    if timestamps:
        bg_280_corrected_im.save("output_figures\\280_bg_corrected_%i%i%i_%i%i.tif"
                                 % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                                    time.gmtime()[4]))
        bg_365_corrected_im.save("output_figures\\365_bg_corrected_%i%i%i_%i%i.tif"
                                 % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                                    time.gmtime()[4]))
        bg_280_otsu_im.save("output_figures\\280_otsu_mask_%i%i%i_%i%i.tif"
                            % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                               time.gmtime()[4]))
        bg_365_otsu_im.save("output_figures\\365_otsu_mask_%i%i%i_%i%i.tif"
                            % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                               time.gmtime()[4]))
        bg_280_masked_im.save("output_figures\\280_bg_corrected_w_otsu_mask_%i%i%i_%i%i.tif"
                              % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                                 time.gmtime()[4]))
        bg_365_masked_im.save("output_figures\\365_bg_corrected_w_otsu_mask_%i%i%i_%i%i.tif"
                              % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                                 time.gmtime()[4]))
        bg_280_triangle_im.save("output_figures\\280_auto_triangle_mask_%i%i%i_%i%i.tif"
                                % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                                   time.gmtime()[4]))
        bg_365_triangle_im.save("output_figures\\365_auto_triangle_mask_%i%i%i_%i%i.tif"
                                % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                                   time.gmtime()[4]))
    else:
        bg_280_corrected_im.save("output_figures\\280_bg_corrected.tif")
        bg_365_corrected_im.save("output_figures\\365_bg_corrected.tif")
        bg_280_otsu_im.save("output_figures\\280_otsu_mask.tif")
        bg_365_otsu_im.save("output_figures\\365_otsu_mask.tif")
        bg_280_masked_im.save("output_figures\\280_bg_corrected_w_otsu_mask.tif")
        bg_365_masked_im.save("output_figures\\365_bg_corrected_w_otsu_mask.tif")
        bg_280_triangle_im.save("output_figures\\280_auto_triangle_mask.tif")
        bg_365_triangle_im.save("output_figures\\365_auto_triangle_mask.tif")

# Generate the desired data plots and save them with or without timestamps according to the flags specified at the top
# of this file.
plt.figure(0)
plt.hist(auto_280_signal, bins=100, label="280nm autofluorescence signal")
plt.hist(auto_365_signal, bins=100, label="365nm autofluorescence signal")
if save_figures:
    if timestamps:
        plt.savefig("output_figures\\autofluorescence_signal_distributions_%i%i%i_%i%i.tif"
                    % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3],
                       time.gmtime()[4]))
    else:
        plt.savefig("output_figures\\autofluorescence_signal_distributions.tif")

plt.figure(1)
plt.hist(bg_280_signal, bins=100, alpha=0.8, label="280nm signal")
plt.hist(bg_365_signal, bins=100, alpha=0.8, label="365nm signal")
plt.xlim([0, 10000])
plt.title("Histograms of the signal values")
plt.legend()
if save_figures:
    if timestamps:
        plt.savefig("output_figures\\Fluorescent_signal_histograms_%i%i%i_%i%i.png"
                    % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3], time.gmtime()[4]),
                    bbox_inches='tight')
    else:
        plt.savefig("output_figures\\Fluorescent_signal_histograms.png", bbox_inches='tight')

plt.figure(2)
plt.plot(x_280, y_280, label="280nm, mean = %.3f" % bg_280_signal_mean)
plt.plot(x_365, y_365, 'r--', label="365nm, mean = %.3f" % bg_365_signal_mean)
plt.title("Distributions of fluorescent signal")
plt.legend()
if save_figures:
    if timestamps:
        plt.savefig("output_figures\\Fluorescent_signal_distributions_%i%i%i_%i%i.png"
                    % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3], time.gmtime()[4]),
                    bbox_inches='tight')
    else:
        plt.savefig("output_figures\\Fluorescent_signal_distributions.png", bbox_inches='tight')

plt.figure(3)
plt.hist(bg_signal_ratio, bins=1000, label="280nm:365nm ratio")
plt.axvline(np.min(bg_signal_ratio), c='r',
            label="Minimum 280nm:365nm signal ratio = %.5f" % np.min(bg_signal_ratio))
plt.title("Histograms of the ratio of signal values")
plt.legend()
if save_figures:
    if timestamps:
        plt.savefig("output_figures\\Fluorescent_signal_ratio_histogram_%i%i%i_%i%i.png"
                    % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3], time.gmtime()[4]),
                    bbox_inches='tight')
    else:
        plt.savefig("output_figures\\Fluorescent_signal_ratio_histogram.png", bbox_inches='tight')

plt.figure(4)
plt.plot(x, y, label="mean = %.3f, stddev = %.3f" % (bg_signal_ratio_mean, bg_signal_ratio_std_dev))
plt.title("Distribution of 280nm:365nm signal")
plt.legend()
if save_figures:
    if timestamps:
        plt.savefig("output_figures\\Fluorescent_signal_ratio_%i%i%i_%i%i.png"
                    % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3], time.gmtime()[4]),
                    bbox_inches='tight')
    else:
        plt.savefig("output_figures\\Fluorescent_signal_ratio.png", bbox_inches='tight')

plt.figure(5)
plt.hist(bg_signal_ratio_no_out, bins=1000, label="280nm:365nm ratio")
plt.axvline(np.min(bg_signal_ratio), c='r',
            label="Minimum 280nm:365nm signal ratio = %.5f" % np.min(bg_signal_ratio))
plt.title("Histograms of the ratio of signal values")
plt.legend()
if save_figures:
    if timestamps:
        plt.savefig("output_figures\\Fluorescent_signal_ratio_histogram_no_out_%i%i%i_%i%i.png"
                    % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3], time.gmtime()[4]),
                    bbox_inches='tight')
    else:
        plt.savefig("output_figures\\Fluorescent_signal_ratio_histogram_no_out.png", bbox_inches='tight')

plt.figure(6)
plt.plot(x_out, y_out, label="mean = %.3f, stddev = %.3f" % (bg_signal_ratio_no_out_mean, bg_signal_ratio_no_out_std_dev))
plt.title("Distribution of 280nm:365nm signal")
plt.legend()
if save_figures:
    if timestamps:
        plt.savefig("output_figures\\Fluorescent_signal_ratio_no_out_%i%i%i_%i%i.png"
                    % (time.gmtime()[2], time.gmtime()[1], time.gmtime()[0], time.gmtime()[3], time.gmtime()[4]),
                    bbox_inches='tight')
    else:
        plt.savefig("output_figures\\Fluorescent_signal_ratio_no_out.png", bbox_inches='tight')

plt.show()
