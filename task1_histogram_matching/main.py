# Oytun Demirbilek
# 150150032

import numpy as np
import os
import cv2
import moviepy.editor as mpy
import matplotlib.pyplot as plt

def remapping(np_array,lookups):
    # Input: Flattened Image Shape: (pixel count, 3) type: numpy ndarray
    # Input: Look up tables of all channels: (lookup table, 3) type: tuple of 3 dictionaries

    # Get remapping of each channels. Replacing the values according to lookup table.
    red_remapped = np.vectorize(lookups[0].get)(np_array[:,0])
    green_remapped = np.vectorize(lookups[1].get)(np_array[:,1])
    blue_remapped = np.vectorize(lookups[2].get)(np_array[:,2])

    # Vertical stacking of all remapped channels gives us the shape: (3, pixel count)
    remapped = np.vstack((red_remapped,green_remapped,blue_remapped))
    # But we need the shape: (pixel count, 3), so calculate transpose.
    remapped = np.transpose(remapped)

    return remapped

def calculate_lookup(hist1,hist2):
    # Input histograms for single channel (hist1 and hist2) Shape: (255,1) Type: Numpy Ndarray
    LUT = {}
    # Iterate over first histogram - 255 iterations.
    for idx,value in enumerate(hist1):
        # Find the element index with closest cdf value.
        index = np.argmin(np.abs(hist2 - value))
        # New key-value pair.
        LUT[idx] = index
    return LUT

def add_noise(hist):
    # Random normal distribution with 0 mean. So the overall sum will not change, which must be 1.
    # Scale is also must be a very small number since the probabilities are very small. 
    # If it is high, multiple values can be remapped to a single value, which will reduce the resolution. 
    noise = np.random.normal(loc=0,scale=0.0001,size=hist.shape[1])
    # Return added noise to each histogram.
    return np.array([hist[0] + noise, hist[1] + noise, hist[2] + noise])

def histogram_matching(hist1,hist2):
    # Matching algorithm prepares a Look-up Table for each channel.
    # Input histograms (hist1 and hist2) Shape: (255,3) Type: Numpy Ndarray
    # Output Look-up tables Shape: 255 key-value pairs Type: Dictionary

    # Cumulative Sum to convert probability densities to cumulative densities.
    CDF1_red, CDF1_green, CDF1_blue = hist1[0].cumsum(),hist1[1].cumsum(),hist1[2].cumsum()
    CDF2_red, CDF2_green, CDF2_blue = hist2[0].cumsum(),hist2[1].cumsum(),hist2[2].cumsum()

    # Convert cumulative densities to Look-up tables.
    LUT_red = calculate_lookup(CDF1_red,CDF2_red)
    LUT_green = calculate_lookup(CDF1_green,CDF2_green)
    LUT_blue = calculate_lookup(CDF1_blue,CDF2_blue)

    return LUT_red, LUT_green, LUT_blue
    
def plot_target_hist(hist):
    # Plot all the channels in a single plot for target histogram.
    red_counts1, green_counts1, blue_counts1 = hist[0],hist[1],hist[2]
    
    plt.plot(red_counts1, color = 'red')
    plt.plot(green_counts1, color = 'green')
    plt.plot(blue_counts1, color = 'blue')
    plt.title("Target Histogram")
    plt.savefig("target_pdf.png")
    plt.close()

def image_histogram(flattened_image):
    # Input shape: (pixel_count, channels) Type: numpy ndarray

    # Allocate the Multi-Band Histogram
    red_counts = np.zeros((256,))
    green_counts = np.zeros((256,))
    blue_counts = np.zeros((256,))

    # First, find which RGB values occured how many times.
    # Select elements with multi-indexing and assign number of occurences.
    # Divide number of occurences by pixel count to obtain probability density function.
    pixel_values, counts = np.unique(flattened_image[:,0], return_counts=True)
    red_counts[list(pixel_values)] = counts / flattened_image.shape[0]

    pixel_values, counts = np.unique(flattened_image[:,1], return_counts=True)
    green_counts[list(pixel_values)] = counts / flattened_image.shape[0]

    pixel_values, counts = np.unique(flattened_image[:,2], return_counts=True)
    blue_counts[list(pixel_values)] = counts / flattened_image.shape[0]

    return np.array([red_counts, green_counts, blue_counts])

def plot_hist(first, second):
    # Collect histograms for all channels for both cats and plot them in a single image.
    red_counts1, green_counts1, blue_counts1 = first[0],first[1],first[2]
    red_counts2, green_counts2, blue_counts2 = second[0],second[1],second[2]

    fig, axs = plt.subplots(1,2,figsize=(24,12))
    
    axs[0].plot(red_counts1, color = 'red')
    axs[0].plot(green_counts1, color = 'green')
    axs[0].plot(blue_counts1, color = 'blue')
    axs[0].title.set_text("Cat 1")

    axs[1].plot(red_counts2, color = 'red')
    axs[1].plot(green_counts2, color = 'green')
    axs[1].plot(blue_counts2, color = 'blue')
    axs[1].title.set_text("Cat 2")

    plt.savefig("averaged_histograms_remapped.png")
    plt.close()
    
def image_to_frame(image, backg, target=None,insert_type="normal",gamma=1,matching=False):
    # Getting RGB channels from cat image separately.
    red = image[:,:,0]
    green = image[:,:,1]
    blue = image[:,:,2]

    frame = backg.copy() # Background is taken.
    
    front = np.logical_or(green<180,red>150) # True values marking the cat.
    x, y = np.nonzero(front) # Index values (locations) of the cat pixels.

    cat_values = image[x,y,:] # Cat image is taken.
    before = image_histogram(cat_values) # Calculate histogram.
    # Using gamma correction transform mapping, we can obtain a darker or lighter image.
    cat_values = np.array(255*(cat_values / 255) ** (1/gamma), dtype = 'uint8')
    
    if matching:
                
        # The cat itself is placed.
        if insert_type=="normal": 
            # Cat 1 operations in part 4.
            noised = add_noise(before) # Randomly perturb the cat’s own histogram
            LUT = histogram_matching(before,noised) # Make histogram matching to cat’s perturbed histogram
            cat_values = remapping(cat_values,LUT) # Remapping to get new image.
            frame[x,y,:] = cat_values 
            return frame

        # The cat reflection is placed.
        if insert_type=="reflect": 
            # Cat 2 operations in part 4.
            after_darken = image_histogram(cat_values)
            target = add_noise(target) # Randomly perturb the target histogram
            LUT = histogram_matching(after_darken,target) # Make histogram matching to target perturbed histogram
            cat_values = remapping(cat_values,LUT) # Remapping to get new image.
            after = image_histogram(cat_values) 
            frame[x,-y,:] = cat_values

            return frame, before, after
    
    else: 
        if insert_type=="normal": frame[x,y,:] = cat_values 
        if insert_type=="reflect": frame[x,-y,:] = cat_values

        return frame
    
def average_hist(hist1_list,hist2_list):
    # Collect all histograms from all frames of both cats.
    # Plot the average.
    hist1_list = np.array(hist1_list)
    hist2_list = np.array(hist2_list)

    # Calculate mean values for each channel for cat 1.
    red_avg1 = np.mean(hist1_list[:,0], axis=0)
    green_avg1 = np.mean(hist1_list[:,1], axis=0)
    blue_avg1 = np.mean(hist1_list[:,2], axis=0)

    # Calculate mean values for each channel for cat 2.
    red_avg2 = np.mean(hist2_list[:,0], axis=0)
    green_avg2 = np.mean(hist2_list[:,1], axis=0)
    blue_avg2 = np.mean(hist2_list[:,2], axis=0)

    # Calculate averages.
    avg_hist1 = np.array([red_avg1,green_avg1,blue_avg1])
    avg_hist2 = np.array([red_avg2,green_avg2,blue_avg2])

    # Plot in one.
    plot_hist(avg_hist1,avg_hist2)

    return avg_hist1, avg_hist2

def part1():
    malibu = cv2.imread('Malibu.jpg')

    bg_height = malibu.shape[0]
    bg_width = malibu.shape[1]

    ratio = 360/bg_height

    malibu = cv2.resize(malibu,(int(bg_width*ratio), 360))

    image_list = []

    for i in range(180):
        cat = cv2.imread(f"./cat/cat_{i}.png")
        # ----------------
        # Part 1
        # ----------------
        onecat = image_to_frame(cat, malibu)
        onecat = onecat[:,:,[2,1,0]] # Moviepy uses BGR, thus reversing the channels
        image_list.append(onecat) # Appending the image sequence.
        print(f"File Read Success: cat_{i}.png")

    # Creating the video.
    clip = mpy.ImageSequenceClip(image_list,fps=25)
    audio = mpy.AudioFileClip("selfcontrol_part.wav").set_duration(clip.duration)
    clip = clip.set_audio(audioclip=audio)
    clip.write_videofile("test_part1_video.mp4",codec="libx264")

def part2():
    malibu = cv2.imread('Malibu.jpg')

    bg_height = malibu.shape[0]
    bg_width = malibu.shape[1]

    ratio = 360/bg_height

    malibu = cv2.resize(malibu,(int(bg_width*ratio), 360))

    image_list = []

    for i in range(180):
        cat = cv2.imread(f"./cat/cat_{i}.png")
        # ----------------
        # Part 2
        # ----------------
        onecat = image_to_frame(cat, malibu)
        twocats = image_to_frame(cat, onecat, insert_type="reflect")
        twocats = twocats[:,:,[2,1,0]] # Moviepy uses BGR, thus reversing the channels
        image_list.append(twocats) # Appending the image sequence.
        print(f"File Read Success: cat_{i}.png")

    # Creating the video.
    clip = mpy.ImageSequenceClip(image_list,fps=25)
    audio = mpy.AudioFileClip("selfcontrol_part.wav").set_duration(clip.duration)
    clip = clip.set_audio(audioclip=audio)
    clip.write_videofile("test_part2_video.mp4",codec="libx264")

def part3():
    malibu = cv2.imread('Malibu.jpg')

    bg_height = malibu.shape[0]
    bg_width = malibu.shape[1]

    ratio = 360/bg_height

    malibu = cv2.resize(malibu,(int(bg_width*ratio), 360))

    image_list = []

    for i in range(180):
        cat = cv2.imread(f"./cat/cat_{i}.png")
        # ----------------
        # Part 3
        # ----------------
        onecat = image_to_frame(cat, malibu)
        twocats = image_to_frame(cat, onecat, insert_type="reflect", gamma=0.2)
        twocats = twocats[:,:,[2,1,0]] # Moviepy uses BGR, thus reversing the channels
        image_list.append(twocats) # Appending the image sequence.
        print(f"File Read Success: cat_{i}.png")

    # Creating the video.
    clip = mpy.ImageSequenceClip(image_list,fps=25)
    audio = mpy.AudioFileClip("selfcontrol_part.wav").set_duration(clip.duration)
    clip = clip.set_audio(audioclip=audio)
    clip.write_videofile("test_part3_video.mp4",codec="libx264")

def part4():
    
    malibu = cv2.imread('Malibu.jpg')

    bg_height = malibu.shape[0]
    bg_width = malibu.shape[1]

    ratio = 360/bg_height

    malibu = cv2.resize(malibu,(int(bg_width*ratio), 360))

    target = cv2.imread("./pacha.png")
    target = target.reshape((target.shape[0]*target.shape[1], target.shape[2]))
    target_hist = image_histogram(target)
    plot_target_hist(target_hist)

    image_list = []
    hist1_list = []
    hist2_list = []
    for i in range(180):
        cat = cv2.imread(f"./cat/cat_{i}.png")
        # ----------------
        # Part 4
        # ----------------
        onecat = image_to_frame(cat, malibu, target=target_hist, matching=True)
        twocats, hist1, hist2 = image_to_frame(cat, onecat, target=target_hist, insert_type="reflect", gamma=0.2, matching=True)

        hist1_list.append(hist1)
        hist2_list.append(hist2)

        twocats = twocats[:,:,[2,1,0]] # Moviepy uses BGR, thus reversing the channels
        image_list.append(twocats) # Appending the image sequence.
        print(f"File Read Success: cat_{i}.png")

    cat1_avg, cat2_avg = average_hist(hist1_list,hist2_list)
    # Creating the video.
    clip = mpy.ImageSequenceClip(image_list,fps=25)
    audio = mpy.AudioFileClip("selfcontrol_part.wav").set_duration(clip.duration)
    clip = clip.set_audio(audioclip=audio)
    clip.write_videofile("test_part4_video.mp4",codec="libx264")

part4()