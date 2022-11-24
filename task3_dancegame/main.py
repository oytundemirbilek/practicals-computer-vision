# Oytun Demirbilek
# 150150032

import pyautogui
import time
import os
import subprocess

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.stats as st

# You need to set the path to the game in order to open it in the program.
GAMEPATH = "./saturdaynightfilter_pc/SaturdayNightFilter.exe"

# Part1 function.
def sobel_filter():
    shapes = cv2.imread("shapes.png")

    # Define Sobel Operator Filters. Horizontal and vertical masks.
    mask_row = np.array([1, 2, 1])
    y_mask = np.array([-1 * mask_row, np.zeros(3), mask_row])
    x_mask = y_mask.transpose()

    # Image Smoothing before Edge Detection.
    shapes = cv2.blur(shapes, (5, 5))

    # Convolutions implemented with Sobel masks. Horizontal and vertical separately.
    x_edges = cv2.filter2D(shapes, -1, x_mask)
    y_edges = cv2.filter2D(shapes, -1, y_mask)

    # You can compare it with built-in Sobel results.
    # sobelx = cv2.Sobel(shapes,cv2.CV_64F,1,0,ksize=5)
    # sobely = cv2.Sobel(shapes,cv2.CV_64F,0,1,ksize=5)
    # plt.imshow(sobelx)
    # plt.show()
    # plt.close()
    # plt.imshow(sobely)
    # plt.show()
    # plt.close()

    # Save Results.
    plt.imshow(y_edges)
    plt.savefig("sobel_y.png")
    plt.close()

    plt.imshow(x_edges)
    plt.savefig("sobel_x.png")
    plt.close()


# Part3 function.
def generate_gaussian(ksize=7, sigma=5):
    # Utility to generate a gaussian window.
    x = np.linspace(-sigma, sigma, ksize + 1)
    dist = np.diff(st.norm.cdf(x))
    kernel = np.outer(dist, dist)
    return kernel / kernel.sum()


# Part3 function.
def hardcode_corners(image, kernel):
    import math

    # Size of the image and the kernel.
    (i_height, i_width) = image.shape[:2]
    (k_height, k_width) = kernel.shape[:2]

    # Half of the kernel size is the padding size.
    ksize = (k_width - 1) // 2
    # We need to pad the image in order to avoid indexes are not in image.
    image = cv2.copyMakeBorder(image, ksize, ksize, ksize, ksize, cv2.BORDER_REPLICATE)
    # Initialize output matrix. Will be the same size as image.
    output = np.zeros((i_height, i_width), dtype="float32")

    # Sliding window iteration.
    for y in np.arange(ksize, i_height + ksize):
        for x in np.arange(ksize, i_width + ksize):
            # Estimate mean difference between pixels.
            Ix = np.diff(image[y, x - ksize : x + ksize + 1]).mean()
            Iy = np.diff(image[y - ksize : y + ksize + 1, x]).mean()

            # Convolve image derivatives y-wise and x-wise separately.
            roi_x = Ix * kernel
            roi_y = Iy * kernel

            # Construct the image structure tensor.
            b = (roi_x * Iy).sum()
            a = (roi_x * roi_x).sum()
            d = (Iy * Iy).sum()
            # Calculate min eigenvalue and it is the kernel center.
            output[y - ksize, x - ksize] = (
                (a + d) - math.sqrt((a - d) ** 2 + 4 * (b**2))
            ) / 2

    return output


# Part3 function.
def corner_detector(image):

    # Sobel windows.
    mask_row = np.array([1, 2, 1])
    y_mask = np.array([-1 * mask_row, np.zeros(3), mask_row])
    x_mask = y_mask.transpose()

    # Gaussian window.
    gaussian_mask = generate_gaussian()

    # Method 1: Use separate kernels in the built-in convolution function.
    G12 = cv2.sepFilter2D(
        image, -1, kernelX=np.array([-1, 0, 1]), kernelY=np.array([-1, 0, 1])
    )  # Ix.Iy --> This can be problematic.
    G11 = cv2.sepFilter2D(
        image**2, -1, kernelX=np.array([-1, 0, 1]), kernelY=np.array([0, 0, 0])
    )  # Ix^2
    G22 = cv2.sepFilter2D(
        image**2, -1, kernelX=np.array([0, 0, 0]), kernelY=np.array([-1, 0, 1])
    )  # Iy^2

    # Calculate min eigenvalue matrices using filtered structures.
    first = G11 + G22
    second = np.sqrt(np.square(G11 - G22) + 4 * np.square(G12))
    min_eigens = (first - second) / 2
    # max_eigens = (first + second) / 2
    # corners = min_eigens*max_eigens - 0.04*np.square(min_eigens+max_eigens)

    # Method 2: Hardcoded convolution. Note that python iterations much slower and this method will take some time.
    # Everything is implemented with formulas.
    min_eigens = hardcode_corners(image, gaussian_mask)

    np.save("star_corners.npy", min_eigens)

    # In order to highlight corners in the image.
    # If the minimum eigenvalue takes very high values, the point is very much likely to be a corner.
    # image = image // 2
    # image[np.where(min_eigens>10000)] = 255
    return min_eigens


# Part4 function.
def detect(c):
    shape = "A"  # DEFAULT

    # Perimeter of the contour is needed for the approximation algorithm.
    peri = cv2.arcLength(c, True)

    # Contour approximation uses Ramer-Douglas-Peucker algorithm
    # to reduce the number of points in the curve.
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    # Assign keyboard presses for each type of result of the approximation.
    if len(approx) < 4:
        shape = "A"  # TRIANGLE
    elif len(approx) == 4:
        shape = "S"  # SQUARE
    elif len(approx) == 6:
        shape = "F"  # HEXAGON
    elif len(approx) > 6:
        shape = "D"  # STAR
    return shape


# Part4 function.
def see_and_respond(ss):
    screensize = pyautogui.size()
    # To grayscale.
    ss = cv2.cvtColor(ss, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    # Getting the frame from the image.
    # Shape fits between x = (1050, 1350)  y = (821, 1080) Tested on 1920 x 1080 resolution.
    # ** These parameters are really tricky, you need to catch the shape in a frame.
    # ** Game results may vary due to the resolution differences.
    slide_y = int(screensize[1] * 0.76) + 1
    slide_x_down = int(screensize[0] * 0.5468) + 1
    slide_x_up = int(screensize[0] * 0.7031) + 1
    frame = ss[slide_y:, slide_x_down:slide_x_up]

    # Get edges and contours.
    edges = cv2.Canny(frame, 400, 500)
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # If no contours are found, there are no shapes in the frame yet.
    if len(contours) == 0:
        return "A"

    # cnts is a coordinate array defining the boundary curve.
    cnts = contours[0]
    if len(cnts.shape) == 3:
        # Since there should be only one shape at a time,
        # Get rid of the 1-sized dimension.
        cnts = cnts.squeeze(1)

    # Contours can be used to detect number of edges of a shape.
    polygon = detect(cnts)
    print("PRESS:", polygon)

    return polygon


# Part4 function.
def play_game():
    game = subprocess.Popen(GAMEPATH)
    screensize = pyautogui.size()
    time.sleep(6)

    # Play Vabank
    pyautogui.click(screensize[0] * 0.9, screensize[1] * 0.25)
    game_end = False
    prev_ss = pyautogui.screenshot()
    time.sleep(1)

    while not game_end:
        ss = pyautogui.screenshot()
        if ss == prev_ss:
            game_end = True
        prev_ss = ss
        key = see_and_respond(np.array(ss))
        pyautogui.press(key)

    pyautogui.press("Esc")

    # Play Shame
    pyautogui.click(screensize[0] * 0.9, screensize[1] * 0.36)
    game_end = False
    prev_ss = pyautogui.screenshot()
    time.sleep(1)

    while not game_end:
        ss = pyautogui.screenshot()
        if ss == prev_ss:
            game_end = True
        prev_ss = ss
        key = see_and_respond(np.array(ss))
        pyautogui.press(key)

    game.terminate()


# Part 1 & 2 function
def get_shapes():
    # Open the game as a subprocess in the program.
    game = subprocess.Popen(GAMEPATH)
    # Screen size will be used to find the button locations
    # Therefore the screen resolution itself will not change things.
    screensize = pyautogui.size()
    time.sleep(5)
    # Click All Shapes
    pyautogui.click(screensize[0] * 0.9, screensize[1] * 0.13)
    # Get screenshot of All Shapes
    ss = pyautogui.screenshot()

    ss.save("shapes.png")
    # Go to the main page from All Shapes.
    pyautogui.click(screensize[0] * 0.9, screensize[1] * 0.9)
    game.terminate()


def part1():
    # Get the All Shapes page screenshot from the game and save.
    get_shapes()
    # Apply sobel filter to saved image from All Shapes page and save the results.
    sobel_filter()


def part2():
    try:
        # If the All Shapes is already saved, read it.
        img = cv2.imread("shapes.png")
    except:
        # If the file does not exist, get the screenshot again.
        get_shapes()
        img = cv2.imread("shapes.png")

    # Canny edge detection.
    edges = cv2.Canny(img, 400, 500)
    plt.imshow(edges, cmap="gray")
    plt.savefig("canny_edges.png")
    plt.close()


def part3():
    try:
        # If the All Shapes is already saved, read it.
        img = cv2.imread("shapes.png")
    except:
        # If the file does not exist, get the screenshot again.
        get_shapes()
        img = cv2.imread("shapes.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    all_shapes = np.array_split(img, 4, axis=1)
    newimage = corner_detector(all_shapes[2])
    corners = np.load("star_corners.npy")
    # corners = cv2.cornerHarris(all_shapes[2],2,3,0.04)

    plt.imshow(newimage)
    plt.show()


def part4():

    # You can check the methods on all shapes page.
    # Works perfectly when the shape is not moving.

    # all_shapes = cv2.imread('shapes.png')

    ## Changes with resolution. **********************************************************
    # all_shapes = all_shapes[420:700,:]

    # all_shapes = cv2.cvtColor(all_shapes, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    # all_shapes = np.array_split(all_shapes, 4,axis=1)

    # edgelist=[]
    # for shape in all_shapes:
    #    edges = cv2.Canny(shape,400,500)
    #    edgelist.append(edges)
    #    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #    print(contours[0].shape)
    #    detect(contours[0].squeeze())

    play_game()


part4()
