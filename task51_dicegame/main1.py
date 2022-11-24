import pyautogui
import time
import os
import subprocess

import numpy as np
import cv2
import matplotlib.pyplot as plt

# You need to set the path to the game in order to open it in the program.
GAMEPATH = "./DiceGame/DiceGame.exe"


def circle_count(img):
    # Image to grayscale.
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Bilateral filter is a bit slower, but increases accuracy of edge detection.
    blurred = cv2.bilateralFilter(img, 50, 80, 95)

    # Edge detection before circle detection increases accuracy of circle detection.
    edges = cv2.Canny(blurred, 100, 100)

    # Hough circle detection.
    circles1 = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        1,
        33,
        param1=100,
        param2=25,
        minRadius=0,
        maxRadius=80,
    )
    if circles1 is not None:
        circles1 = np.uint16(np.around(circles1))
        for i in circles1[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 5, 1)
        # Return count of the circles detected.
        a = len(circles1[0, :])
    else:
        # If there is no circles detected.
        a = 0
    return a


def play_part1(screensize):
    myImg = pyautogui.screenshot()

    myImg = np.array(myImg)
    # Numpy arrays has opposite index, X are y and Y are x.
    dice1 = myImg[: int(screensize[1] * 0.5), : int(screensize[0] * 0.33)]
    a = circle_count(dice1)
    dice2 = myImg[
        : int(screensize[1] * 0.5),
        int(screensize[0] * 0.33) : int(screensize[0] * 0.66),
    ]
    b = circle_count(dice2)
    dice3 = myImg[: int(screensize[1] * 0.5), int(screensize[0] * 0.66) :]
    c = circle_count(dice3)

    # Compare results and make a decision.
    if a >= b and a >= c:
        pyautogui.press("a")
    elif b > a and b >= c:
        pyautogui.press("s")
    elif c > a and c > b:
        pyautogui.press("d")
    time.sleep(0.1)


def play_part2(screensize, n_trials=8):

    circles1 = []
    circles2 = []
    circles3 = []

    # Try to detect circles n_trials times.
    for i in range(n_trials):
        myImg = pyautogui.screenshot()
        myImg = np.array(myImg)
        # Numpy arrays has opposite index, X are y and Y are x.
        dice1 = myImg[: int(screensize[1] * 0.5), : int(screensize[0] * 0.33)]
        circles1.append(circle_count(dice1))

        dice2 = myImg[
            : int(screensize[1] * 0.5),
            int(screensize[0] * 0.33) : int(screensize[0] * 0.66),
        ]
        circles2.append(circle_count(dice2))

        dice3 = myImg[: int(screensize[1] * 0.5), int(screensize[0] * 0.66) :]
        circles3.append(circle_count(dice3))

    # Average circles detected for each dice.
    a = sum(circles1) / n_trials
    b = sum(circles2) / n_trials
    c = sum(circles3) / n_trials

    if a >= b and a >= c:
        pyautogui.press("a")
    elif b > a and b >= c:
        pyautogui.press("s")
    elif c > a and c > b:
        pyautogui.press("d")


def play_game(n_rounds=20):
    # Open game as a subprocess.
    game = subprocess.Popen(GAMEPATH)
    screensize = pyautogui.size()

    # Wait for the game to be loaded.
    time.sleep(6)
    pyautogui.click(screensize[0] * 0.5, screensize[1] * 0.5)  # Part 1 button
    for round in range(n_rounds):
        play_part1(screensize)
    pyautogui.press("Esc")  # Go back to main menu.
    pyautogui.click(screensize[0] * 0.5, screensize[1] * 0.6)  # Part 2 button
    for round in range(n_rounds):
        play_part2(screensize)
    pyautogui.press("Esc")  # Go back to main menu.
    pyautogui.click(screensize[0] * 0.5, screensize[1] * 0.75)  # Exit button

    # Close the game.
    game.terminate()


play_game()
