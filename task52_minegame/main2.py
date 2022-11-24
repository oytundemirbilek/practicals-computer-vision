import pyautogui
import time
import os
import subprocess

import numpy as np
import cv2
import matplotlib.pyplot as plt
import dlib
import math

# You need to set the path to the game in order to open it in the program.
GAMEPATH = "./MineGame/MineGame.exe"
# You need to set the path to the shape detector in order to use it in the program. Download from dlib.
DETECTORPATH = "./shape_predictor_68_face_landmarks.dat"


def detect_face(img):
    # Load the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DETECTORPATH)

    # Detect rectangles on grayscale.
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    rectangles = detector(gray)  # Use detector to find landmarks
    landmarks = []
    # There should be 1 rectangle in this sense.
    for i in rectangles:
        x1 = i.left()
        y1 = i.top()
        x2 = i.right()
        y2 = i.bottom()
        # Get facial landmark points.
        points = predictor(gray, rectangles[0])
        for n in range(0, 68):
            # Get coordinates of landmark point.
            x = points.part(n).x
            y = points.part(n).y
            # Mark landmark points as circles for visualization.
            cv2.circle(img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
            # Collect all coordinates.
            landmarks.append([x, y])

    return np.array(landmarks)


def opposite_direction(direction):
    # Function to determine the opposite of each direction.
    if direction == "w":
        opp = "s"
    if direction == "s":
        opp = "w"
    if direction == "a":
        opp = "d"
    if direction == "d":
        opp = "a"
    return opp


def go_opposite(direction):
    # Function to perform going to the opposite of the current direction.
    go = opposite_direction(direction)
    pyautogui.keyDown(go)
    time.sleep(0.5)
    pyautogui.keyUp(go)
    time.sleep(0.7)


def go_next_grid(direction):
    # Function to perform going to a neighbor grid.
    pyautogui.keyDown(direction)
    time.sleep(1.9)
    pyautogui.keyUp(direction)
    time.sleep(0.7)


def is_edge(ss, screensize, direction):
    # Function to determine if there are any neighbors in the direction, or space.
    x, y = int(screensize[1] * 0.27), int(screensize[0] * 0.5)
    print("CENTER:", x, y)
    # Check N points away of each direction. N is automatically assigned using screen size.
    if direction == "a":
        n_x = x
        n_y = int(y - screensize[0] * 0.05)
    if direction == "d":
        n_x = x
        n_y = int(y + screensize[0] * 0.05)

    if direction == "w":
        n_x = int(x - screensize[1] * 0.09)
        n_y = y
    if direction == "s":
        n_x = int(x + screensize[1] * 0.09)
        n_y = y

    neighbor_grid = np.array(ss)[n_x, n_y]
    print(n_x, n_y, neighbor_grid)
    # If neighbor grid's color is same as space, it is space.
    if (neighbor_grid - np.array([255, 111, 114])).mean() == 0:
        return True
    else:
        return False


def is_safe(direction, screensize):
    # Come closer to the neighbor.
    pyautogui.keyDown(direction)
    time.sleep(0.5)
    pyautogui.keyUp(direction)
    time.sleep(0.7)
    ss = pyautogui.screenshot()
    # plt.imshow(ss)
    # plt.show()
    if is_edge(ss, screensize, direction):
        # If there is not a neighbor, you are very close to the edge of the board!!
        print(direction, "is edge.")
        go_opposite(direction)
        # It is not safe to walk to the space.
        return False
    else:
        # If there is a neighbor, next task is to control if there is a mine on it.
        print(direction, "is not edge.")
        go_opposite(direction)

    # Face at the right bottom.
    cropped = np.array(ss)[int(screensize[1] * 0.78) :, int(screensize[0] * 0.8766) :]
    # Get coordinates of facial landmarks.
    landmarks = detect_face(cropped)
    # Average coordinates for X and Y axis.
    face_mean = landmarks.mean(axis=0)

    # Checking for only X axis was enough.
    if face_mean[0] > screensize[1] * 0.13 and face_mean[0] < screensize[1] * 0.18:
        # It is not safe to walk to the mine.
        return False
    else:
        return True


def play_game():
    # Open game as a subprocess.
    game = subprocess.Popen(GAMEPATH)
    screensize = pyautogui.size()
    directions = ["w", "a", "s", "d"]
    # Wait for the game to be loaded.
    time.sleep(6)
    last_direction = "s"
    while True:
        safe_ways = []
        for direction in directions:
            # Chack each direction for safety.
            if direction != last_direction:
                safety = is_safe(direction, screensize)
            else:
                safety = False
            if safety:
                # If it is safe, consider it as an option.
                safe_ways.append(direction)

        if len(safe_ways) > 0:
            # If there is a safe option, go to the next grid.
            go_next_grid(safe_ways[0])
            print("NEXT GRID")
            # Update the last direction to not to come back.
            last_direction = opposite_direction(safe_ways[0])

    # Close the game.
    game.terminate()


play_game()
