import moviepy.video.io.VideoFileClip as mpy 
import moviepy.editor as edit
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

MOVIEPATH = "movie_001.avi"

def mark_circle_center(img, prm2=25):
    # Marking circle center gives the features of trackability.
    detected_circles = cv2.HoughCircles(img,  
                                        cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
                                        param2 = prm2, minRadius = 0, maxRadius = 0) 

    # Draw circles that are detected. 
    if detected_circles is not None: 
    
        # Convert the circle parameters a, b and r to integers. 
        detected_circles = np.uint16(np.around(detected_circles)) 
        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2] 
    
            # Draw a small rectangle (of radius 1) to show the center. 
            cv2.rectangle(img, (a-2, b-2), (a+2, b+2), 0, -1) # Mark center with a rectangle.
        detected_centers = detected_circles[:, :, :2].transpose((1,0,2))
        detected_radius = detected_circles[:, :, 2]
        return detected_centers, detected_circles.shape[1]
    else: 
        print("(WARNING) NO Circles. Param2: ", prm2)
        return None, 0

def lucas_canade(frame0,frame1, x,y, window_size=5):
    # Get index range of window.
    window_start= window_size // 2
    window_end= window_size // 2 + 1

    # Take derivatives.
    Ix = cv2.filter2D(frame0,-1,np.array([[-1,1],[-1,1]]))[ x-window_start : x+window_end, y-window_start : y+window_end]
    Iy = cv2.filter2D(frame0,-1,np.array([[-1,-1],[1,1]]))[ x-window_start : x+window_end, y-window_start : y+window_end]
    It = (cv2.filter2D(frame0,-1,np.ones((2,2))) + cv2.filter2D(frame1,-1,np.ones((2,2))*-1))[ x-window_start : x+window_end, y-window_start : y+window_end]

    # Find image structure tensor.
    A_T = np.array([Ix.flatten(),Iy.flatten()])
    A = np.array([Ix.flatten(),Iy.flatten()]).transpose()
    b = np.array([It.flatten()]).transpose()
    # Build equation: (u,v) = -(A_T.A)^-1.A_T.b
    motion_vector = np.matmul(np.matmul( -1*(np.linalg.pinv(np.matmul(A_T,A)) ),A_T),b)
    return motion_vector.squeeze()

def find_motion():
    cap = cv2.VideoCapture(MOVIEPATH) 
    
    # Take first frame and find corners in it 
    ret, old_frame = cap.read() 
    old_gray = cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY) 
    old_gray = cv2.GaussianBlur(old_gray,(5,5),2,2)
    p0, n_circles = mark_circle_center(old_gray)
    print("BALLS FOUND:",n_circles)

    all_of = np.zeros((0,5))
    image_list = []
    while True: 

        # Create a mask image for drawing purposes 
        ret, frame = cap.read() 
        try: frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        except: break
        frame_gray = cv2.GaussianBlur(frame_gray,(5,5),2,2)
        p1, n_circles_new = mark_circle_center(frame_gray)
        # Optical flow 
        OFs = []
        for pt in p0.squeeze():
            x, y = int(pt[0]),int(pt[1])
            motion = lucas_canade(old_gray,frame_gray,y,x)
            OFs.append(motion.tolist() + frame[y,x].tolist())
            cv2.arrowedLine(frame,(x,y),(x+int(motion[0]), y+int(motion[1])),(255,255,255))

        all_of = np.concatenate((all_of,OFs),axis=0)
    
        # Updating Previous frame and points  
        image_list.append(frame)
        old_gray = frame_gray.copy() 
        p0 = p1.reshape(-1, 1, 2) 

    clip = edit.ImageSequenceClip(image_list,fps=25)
    clip.write_videofile("ball_vectors.mp4",codec="libx264")

    return all_of

def separate_balls(results):
    clusterer = KMeans(n_clusters=5)
    clusters = clusterer.fit_predict(results[:,2:])
    colors = clusterer.cluster_centers_.astype(np.int32)
    ball_velocities_x = []
    ball_velocities_y = []
    ball_number = 0
    for i in range(5):
        if np.all(colors[i] < 205) and np.all(colors[i] > 190):
            print("Flat area is filtered out.")
            continue
        ball_number += 1
        ball=results[clusters==i]
        flow_x = ball[:,0]
        flow_y = ball[:,1]

        flow_x = flow_x[np.nonzero(flow_x)]
        flow_y = flow_y[np.nonzero(flow_y)]

        avg_speed_x = np.around(np.abs(flow_x).mean(),3)
        avg_speed_y = np.around(np.abs(flow_y).mean(),3)

        print(f"Ball {ball_number} Speed: {avg_speed_x} + {avg_speed_y} = {round(avg_speed_x + avg_speed_y, 3)}, Color: {colors[i]}")
        
        ball_velocities_x.append(flow_x)
        ball_velocities_y.append(flow_y)

    return ball_velocities_x, ball_velocities_y

allmotion = find_motion()
np.save("circle_flow.npy",allmotion)

allmotion = np.load("circle_flow.npy")
xs,ys = separate_balls(allmotion)
