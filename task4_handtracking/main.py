# Oytun Demirbilek
# 150150032

import moviepy.video.io.VideoFileClip as mpy 
import moviepy.editor as edit
import cv2
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def MSE(y_pred, y_true):
    return np.sqrt(np.square(y_pred - y_true)).mean()

def get_nfold_split(X, number_of_folds, current_fold_id):
    kf = KFold(n_splits=number_of_folds)
    split_indices = kf.split(range(X.shape[0]))
    train_indices, test_indices = [(list(train), list(test)) for train, test in split_indices][current_fold_id]
    #Split train and test
    X_train = X[train_indices]
    X_test = X[test_indices]
    return X_train, X_test

def fix_camera(samples, truths, axis, n_folds=5):
    np.random.seed(1)

    # In part 3, we need to find a learned function that maps the OF with camera motion to the fixed one.
    max_err = 9999999
    best_model = None
    for f in range(n_folds):
        predictor = SVR(C=1.0, epsilon=0.2)
        X_train, X_test = get_nfold_split(samples,n_folds,f)
        Y_train, Y_test = get_nfold_split(truths,n_folds,f)
        predictor.fit(X_train,Y_train[:,axis])
        y_pred = predictor.predict(X_test)
        error = MSE(y_pred,Y_test[:,axis])
        if error < max_err: best_model = predictor
        print(f"MSE Fold {f}:",error)

    return best_model

def lucas_canade(frame0,frame1, x,y, window_size=5,camera_model=None):
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
    if camera_model is not None:
        x = camera_model[0].predict(motion_vector.T)[0]
        y = camera_model[1].predict(motion_vector.T)[0]
        return np.array([x,y])
    return motion_vector.squeeze()

def optical_flow(video, points=None, background_motion=None, next_images=1, update_pos=True, mark_region=False, camera_model=None):
    frame_count = video.reader.nframes
    video_fps = video.fps
    image_list = []
    frame_displacements = []
    if points is None: find_points = True 
    else: find_points = False
    for i in range(frame_count-next_images):
        # Get video frames.
        walker_frame0 = video.get_frame(i*1.0/video_fps)
        if mark_region: 
            # In part 2, region is marked as a red rectangle.
            walker_frame0 = cv2.rectangle(walker_frame0, tuple(points[0]), tuple(points[3]), color=(255,0,0), thickness=2) 
        # Reduce noise and take the grayscale.
        im0 = cv2.medianBlur(walker_frame0,5)
        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if find_points: 
            points = find_hand(walker_frame0)
            print("HAND LOC:",points)
        corner_displacements = []
        motion_list = []
        for p in points:
            x = p[0]
            y = p[1]
            # Calculate vectors for the next N frame sequence.
            for j in range(1,next_images+1):
                # j will be delta-t
                walker_frame1 = video.get_frame((i+j)*1.0/video_fps)
                # Reduce noise and take the grayscale.
                im1 = cv2.medianBlur(walker_frame1,5)
                im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY).astype(np.float32)
                # Find optical flow of a given point between 2 images.
                motion = lucas_canade(im0,im1,y,x,window_size=5,camera_model=camera_model)
                print("OF:",motion)
                
                if background_motion is not None: 
                    # In part 2, we subtract the background motion from calculated hand motion to reduce error.
                    # Background motion is the average/absmax of the calculated motion of the selected region on the wall.
                    fix = background_motion.mean(axis=(0,1))
                    # fix = absmaxND(fix,axis=0) # Take absolute max of the OF vectors of corners.
                    print("BACKGROUND:",fix)
                    motion = motion - fix
                # Normalize calculated velocity vector and find displacement by multiplying with delta-t
                if np.linalg.norm(motion) != 0: motion_list.append(motion*j)# / np.linalg.norm(motion))
                else: motion_list.append(np.zeros(2))
                #motion_list.append(motion)

            # Average displacement in the next N frame sequence.
            avg_displacement = np.round(np.array(motion_list).mean(axis=0)).astype(np.int32)
            displacement_x, displacement_y = avg_displacement[0], avg_displacement[1]
            corner_displacements.append(avg_displacement)
            #print("DISPLACEMENT:",avg_displacement)
            print(f"MOVES FROM: {x},{y} TO: {x-displacement_x},{y-displacement_y}")

            # Draw arrow. Exaggerated for better visualization.
            cv2.arrowedLine(walker_frame0,(x,y),(x+displacement_x*10, y+displacement_y*10),(255,255,255))

            if update_pos:
                # In order to track the hand, we will need to update its position.
                p[0] = x - displacement_x
                p[1] = y - displacement_y
        frame_displacements.append(np.array(corner_displacements))
        image_list.append(walker_frame0)
    return image_list, np.array(frame_displacements)

def part1():
    biped_vid = mpy.VideoFileClip("biped_1.avi")
    
    # (x,y) Corner of the green hand: (401,336) at first frame. 
    x, y = 401, 334
    image_list,frame_displacements = optical_flow(biped_vid,[[x,y]],next_images=1,update_pos=True)

    clip = edit.ImageSequenceClip(image_list,fps=25)
    clip.write_videofile("test_hand_corners_part1_video.mp4",codec="libx264")
    np.save("truth_biped_1_hand.npy",np.array(frame_displacements))


def part2():
    biped_vid = mpy.VideoFileClip("biped_2.avi")

    # Region motion.
    corners = [[210,183],[308,183],[210,315],[308,315]]
    image_list,region_displacements = optical_flow(biped_vid,corners,next_images=1, update_pos=False, mark_region=True)

    clip = edit.ImageSequenceClip(image_list,fps=25)
    clip.write_videofile("test_part2_video_region.mp4",codec="libx264")
    np.save("truth_biped_2_region.npy",np.array(region_displacements))

    # Hand motion.
    x, y = 401, 334
    region_displacements = np.load("truth_biped_2_region.npy")
    image_list,frame_displacements = optical_flow(biped_vid,[[x,y]],next_images=1, background_motion=None) # Considering background motion did not give better results.

    clip = edit.ImageSequenceClip(image_list,fps=25)
    clip.write_videofile("test_part2_video_hand_wallignored.mp4",codec="libx264")
    np.save("truth_biped_2_hand_wallignored.npy",np.array(frame_displacements))


def part3():
    biped_vid = mpy.VideoFileClip("biped_3.avi")

    hand_truth = np.load("truth_biped_1_hand.npy")
    region_truth = np.load("truth_biped_2_region.npy")

    # Region motion.
    corners = [[210,183],[308,183],[210,315],[308,315]]
    image_list,region_displacements = optical_flow(biped_vid,corners,next_images=1,update_pos=False, mark_region=True)

    clip = edit.ImageSequenceClip(image_list,fps=25)
    clip.write_videofile("test_part3_video_region.mp4",codec="libx264")
    np.save("truth_biped_3_region.npy",np.array(region_displacements))
    
    region_displacements= np.load("truth_biped_3_region.npy")
    print("Mean Squared Error (REGION): ",MSE(region_displacements,region_truth))
    region_shape = region_truth.shape
    print("-------------------------------------------------------------------")
    print("Model X Axis")
    model_x = fix_camera(region_displacements.reshape((region_shape[0]*region_shape[1],region_shape[2])),region_truth.reshape((region_shape[0]*region_shape[1],region_shape[2])), axis=0)
    print("-------------------------------------------------------------------")
    print("Model Y Axis")
    model_y = fix_camera(region_displacements.reshape((region_shape[0]*region_shape[1],region_shape[2])),region_truth.reshape((region_shape[0]*region_shape[1],region_shape[2])), axis=1)
    print("-------------------------------------------------------------------")

    # Hand motion.
    x, y = 401, 334
    image_list,frame_displacements = optical_flow(biped_vid,[[x,y]],next_images=1,camera_model=(model_x,model_y))

    clip = edit.ImageSequenceClip(image_list,fps=25)
    clip.write_videofile("test_part3_video_hand.mp4",codec="libx264")
    np.save("truth_biped_3_hand.npy",np.array(frame_displacements))

    hand_displacements= np.load("truth_biped_3_hand.npy")
    print("Mean Squared Error (HAND): ",MSE(hand_displacements,hand_truth))

