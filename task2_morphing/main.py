# Oytun Demirbilek
# 150150032

import dlib
import cv2
import numpy as np
import moviepy.editor as mpy

def match_triangulation(tri_indexes,landmarks_points):
    # Add coords of the triangle corners on the edge of the image.
    edges = np.array([[0,0],
                      [0,200],
                      [0,399],
                      [200,0],
                      [399,200],
                      [200,399],
                      [399,0],
                      [399,399]])
    landmarks_points = np.append(landmarks_points,edges,0)
    triangles = []
    for index in tri_indexes:
        # Create a triangle using given indexes.
        pt1 = landmarks_points[index[0]]
        pt2 = landmarks_points[index[1]]
        pt3 = landmarks_points[index[2]]
        triangles.append(np.array([pt1[0],pt1[1],pt2[0],pt2[1],pt3[0],pt3[1]]))
    return np.array(triangles)

def find_index(nparray):
    # Some indexes can be empty. Represent them as -1
    index = -1
    # Some indexes can have varying length. Return first one.
    if len(nparray[0]) != 0: return nparray[0][0]
    else: return index

def draw_triangles(image,triangles):
    for t in triangles:
        p1 = (t[0],t[1])
        p2 = (t[2],t[3])
        p3 = (t[4],t[5])
        # Draw lines between triangle corners.
        # Lines are green and their thickness is 1.
        cv2.line(image,p1,p2,(0,255,0),1)
        cv2.line(image,p2,p3,(0,255,0),1)
        cv2.line(image,p1,p3,(0,255,0),1)
    return image

def get_triangle_indexes(triangles,points):
    # This is a crucial function to matsh landmark ids of triangles. 
    # So we can get the same triangle in both images in order to match.
    indexes_triangles = []
    for t in triangles:
        index1 = np.where((points == (t[0], t[1])).all(axis=1))
        index1 = find_index(index1)
        index2 = np.where((points == (t[2], t[3])).all(axis=1))
        index2 = find_index(index2)
        index3 = np.where((points == (t[4], t[5])).all(axis=1))
        index3 = find_index(index3)
        if index1 != -1 and index2 != -1 and index3 != -1:
            triangle = [index1, index2, index3]
            indexes_triangles.append(triangle)
    return indexes_triangles

def delaunay_triangulation(image,points):
    # Takes image and landmark points (ndarray) as input.
    subdiv = cv2.Subdiv2D((0,0,image.shape[0],image.shape[1]))
    edges = np.array([[0,0],
                      [0,200],
                      [0,399],
                      [200,0],
                      [399,200],
                      [200,399],
                      [399,0],
                      [399,399]])
    points = np.append(points,edges,0)
    for i in range(76):
        # Inserting all landmark points.
        subdiv.insert((points[i,0],points[i,1]))
    # Insert points on the image frame.
    triangles = subdiv.getTriangleList()
    
    idx_triangles = get_triangle_indexes(triangles,points)
    return triangles,idx_triangles

def mark_landmarks(image,points=None,only_points=False):
    if points is None:
        # If there is not any points specified, model will calculate them.
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        rectangles = detector(gray)
        if len(rectangles) == 0: print("WARNING: No Face Found.")
        points = predictor(gray,rectangles[0])

        # Create landmark points array.
        vec=[]
        for i in range(68):
            vec.append([points.part(i).x,points.part(i).y])
        vec = np.array(vec)
        if only_points: return vec
        # Mark on image as green.
        image[vec[:,1],vec[:,0]] = np.array([0,255,0])
        # Mark neighboring pixels to make the points more visible. Shaped as (+).
        image[vec[:,1]+1,vec[:,0]] = np.array([0,255,0])
        image[vec[:,1],vec[:,0]+1] = np.array([0,255,0])
        image[vec[:,1],vec[:,0]-1] = np.array([0,255,0])
        image[vec[:,1]-1,vec[:,0]] = np.array([0,255,0])
        return image,vec
    else:
        # If points are precalculated, just mark them on the image.
        if only_points: return points
        image[points[:,1],points[:,0]] = np.array([0,255,0])
        image[points[:,1]+1,points[:,0]] = np.array([0,255,0])
        image[points[:,1],points[:,0]+1] = np.array([0,255,0])
        image[points[:,1],points[:,0]-1] = np.array([0,255,0])
        image[points[:,1]-1,points[:,0]] = np.array([0,255,0])
        return image,points

def make_homogeneous(tri):
    # (C)
    # Necessary for calculating homogeneous coordinates.
    # Using Homogeneous coordinates makes it possible for geometric transforms.
    # (e.g. translation or affine) to be represented
    # as matrix vector multiplication, i.e. a linear transformation
    # Reshape triangle matrix as: 
    # | x1 x2 x3 |
    # | y1 y2 y3 |
    # |  1  1  1 |
    return np.array([tri[::2],tri[1::2],[1,1,1]]) 

def calc_transform(tri1,tri2):
    # Calculates coefficients of affine transformation matrix for: A' = M.A
    source = make_homogeneous(tri1).T
    target = tri2

    # (D) This will give the coefficients needed for the affine transformation.
    # A.k.a: Estimation through correspondences.
    # First we must define correspondence matrix (Mtx) as:
    # | x1 y1  1  0  0  0|
    # |  0  0  0 x1 y1  1|
    # | x2 y2  1  0  0  0|
    # |  0  0  0 x2 y2  1|
    # | x3 y3  1  0  0  0|
    # |  0  0  0 x3 y3  1|
    Mtx = np.array([np.concatenate((source[0], np.zeros(3))),
                    np.concatenate((np.zeros(3), source[0])),

                    np.concatenate((source[1], np.zeros(3))),
                    np.concatenate((np.zeros(3), source[1])),

                    np.concatenate((source[2], np.zeros(3))),
                    np.concatenate((np.zeros(3), source[2]))
                    ])
    # (E) Implementing formula: a = M^-1.q 
    # This will give us necessary coefficients to apply transform: q = M.a
    # Calculate pseudo inverse of matrix (Mtx) using its singular-value decomposition.
    # Then matrix multiplication with the target gives: 
    # | a1 a2 a3 a4 a5 a6 |
    coefs = np.matmul(np.linalg.pinv(Mtx), target) 
    # (F) Reconstruct q (coef matrix) as an affine transformation:
    # | a1 a2 a3 |
    # | a4 a5 a6 |
    # |  0  0  1 |
    trans = np.array([coefs[:3], coefs[3:], [0,0,1]])
    return  trans

def vectorised_Bilinear(coords, target_img, size):
    # Coordinates taken as floating points. Turn them into indexes.
    coords[0] = np.clip(coords[0], 0, size[0]-1)
    coords[1] = np.clip(coords[1], 0, size[1]-1)
    lower = np.floor(coords).astype(np.uint32)
    upper = np.ceil(coords).astype(np.uint32)


    err = coords - lower
    residual = 1 - err

    # (G) Implement bilinear interpolation formula: 
    # In order to calculate the interpolated point (xf,yf), we define a square. 
    # Then linearly interpolate between upper two poins and lower two points.
    top_left = np.multiply(np.multiply(residual[0],residual[1]).reshape(coords.shape[1],1),target_img[lower[0],lower[1],:])
    top_right = np.multiply(np.multiply(residual[0],err[1]).reshape(coords.shape[1],1),target_img[lower[0],upper[1],:])
    bot_left = np.multiply(np.multiply(err[0],residual[1]).reshape(coords.shape[1],1),target_img[upper[0],lower[1],:])
    bot_right = np.multiply(np.multiply(err[0],err[1]).reshape(coords.shape[1],1),target_img[upper[0],upper[1],:])
    
    # (H) Implement bilinear interpolation formula: I(xf,yf) = (1-a)(1-b).I(x,y) + a(1-b).I(x+1,y) + (1-a)b.I(x,y+1) + ab.I(x+1,y+1)
    return np.uint8(np.round(top_left+top_right+bot_left+bot_right)) 

def image_morph(image1,image2,tri1,tri2,transforms,t):
    inter1 = np.zeros(image1.shape).astype(np.uint8)
    inter2 = np.zeros(image2.shape).astype(np.uint8)
    for i, trans in enumerate(transforms):
        # (I) Calculate interpolation using homegenous matrices of corresponding triangles.
        homo_inter_tri = (1-t)*make_homogeneous(tri1[i])+t*make_homogeneous(tri2[i])
        
        # (J) Find position of each triangle and fill them with white. 
        polygon_mask = np.zeros(image1.shape[:2],dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [np.int32(np.round(homo_inter_tri[1::-1,:].T))], color=255) 
        
        # (K) Get all the coordinates where the triangle is positioned.
        seg = np.where(polygon_mask==255)

        # (L) Collect all the triangle coordinates in a numpy matrix.
        # Shape: (n_points, 3) 
        # | all_x all_y 1 |
        mask_points = np.vstack((seg[0],seg[1],np.ones(len(seg[0]))))
        
        # (M) Get interpolated triangle corner coordinates: | x1 y1 x2 y2 x3 y3 |
        inter_tri = homo_inter_tri[:2].flatten(order="F") 

        # (M) Calculate affine transformation matrix in order to convert:
        # triangle1 --> interpolated_triangle
        # triangle2 --> interpolated_triangle
        # Where interpolated triangle is some triangle in between.
        inter_to_img1 = calc_transform(inter_tri,tri1[i])
        inter_to_img2 = calc_transform(inter_tri,tri2[i])

        # (N) Multiplying coordinates of the triangle with the transformation matrix.
        # This will give the coordinates for warping, a.k.a displaced control points.
        mapped1 = np.matmul(inter_to_img1,mask_points)[:-1] 
        mapped2 = np.matmul(inter_to_img2,mask_points)[:-1]

         # (O) Perform the warping operation for both images using bilinear interpolation.
        inter1[seg[0],seg[1],:] = vectorised_Bilinear(mapped1,image1,inter1.shape)
        inter2[seg[0],seg[1],:] = vectorised_Bilinear(mapped2,image2,inter2.shape)

    # (P) Morph Formula: M(t) = (1 - t).I(t) + t.J(t) 
    result = (1-t)*inter1 + t*inter2 
    return result.astype(np.uint8)

def morphing(image1,image2):
    p1 = mark_landmarks(image1,only_points=True)
    p2 = mark_landmarks(image2,only_points=True)

    # Calculate Delaunay Triangulation using Dlib for the first image.
    tri1,indexes = delaunay_triangulation(image1,p1)
    # Calculate Delaunay Triangulation for second image using the first image's landmarks.
    tri2 = match_triangulation(indexes,p2)

    tri1 = tri1[:,[1,0,3,2,5,4]]
    tri2 = tri2[:,[1,0,3,2,5,4]]
    transforms = np.zeros((len(tri1), 3, 3))
    
    for i,tri in enumerate(tri1):
        # (A) Calculate affine transformation matrices 
        # for each triangle in image 1 to image 2
        transforms[i] = calc_transform(tri, tri2[i])
    morphs = []
    for t in np.arange(0,1.0001,0.02): 
        # (B) Interpolating between 2 images.
        # Get step by step morphs from image1 until we get image2 
        # for different values of t between 0 <= t <= 1
        print("Processing:\t",t*100,"%")
        morphs.append(image_morph(image1,image2,tri1,tri2,transforms,t)[:,:,::-1])
    return morphs

def part1_main():
    deniro = cv2.imread("deniro.jpg")
    akbas = cv2.imread("aydemirakbas.png")
    kimbodnia = cv2.imread("kimbodnia.png")
    gorilla = cv2.imread("gorilla.jpg")
    cat = cv2.imread("cat.jpg")
    panda = cv2.imread("panda.jpg")

    deniro,points = mark_landmarks(deniro)
    akbas,points = mark_landmarks(akbas)
    kimbodnia,points = mark_landmarks(kimbodnia)

    gorilla_marks = np.load("gorilla_landmarks.npy")
    cat_marks = np.load("cat_landmarks.npy")
    panda_marks = np.load("panda_landmarks.npy")

    gorilla,points = mark_landmarks(gorilla,gorilla_marks)
    cat,points = mark_landmarks(cat, cat_marks)
    panda,points = mark_landmarks(panda, panda_marks)

    all_animals = np.append(cat,panda,axis=1)
    all_animals = np.append(all_animals,gorilla,axis=1)

    all_humans = np.append(deniro,akbas,axis=1)
    all_humans = np.append(all_humans,kimbodnia,axis=1)

    all_img = np.append(all_humans,all_animals,axis=0)
    
    cv2.imshow("landmarks",all_img)
    cv2.waitKey(0)

def part2_main():
    deniro = cv2.imread("deniro.jpg")
    akbas = cv2.imread("aydemirakbas.png")
    deniro, deniro_marks = mark_landmarks(deniro)
    akbas, akbas_marks = mark_landmarks(akbas)

    deniro_triangles, indexes = delaunay_triangulation(deniro,deniro_marks)
    akbas_triangles = match_triangulation(indexes,akbas_marks)

    deniro = draw_triangles(deniro,deniro_triangles)
    akbas = draw_triangles(akbas,akbas_triangles)
    
    both = np.append(deniro,akbas,axis=1)
    cv2.imshow("landmarks",both)
    cv2.waitKey(0)

def part3_main():
    deniro = cv2.imread("deniro.jpg")
    akbas = cv2.imread("aydemirakbas.png")
    kimbodnia = cv2.imread("kimbodnia.png")
    
    morph_list = morphing(deniro,akbas)
    clip = mpy.ImageSequenceClip(morph_list,fps=25)
    clip.write_videofile("deniro_akbas_morph_video.mp4",codec="libx264")

    morph_list = morphing(deniro,kimbodnia)
    clip = mpy.ImageSequenceClip(morph_list,fps=25)
    clip.write_videofile("deniro_kimbodnia_morph_video.mp4",codec="libx264")

    morph_list = morphing(kimbodnia,akbas)
    clip = mpy.ImageSequenceClip(morph_list,fps=25)
    clip.write_videofile("kimbodnia_akbas_morph_video.mp4",codec="libx264")

def play(name1,name2):
    im1 = cv2.imread(f"{name1}.jpg")
    im2 = cv2.imread(f"{name2}.jpg")
    morph_list = morphing(im1,im2)
    clip = mpy.ImageSequenceClip(morph_list,fps=25)
    clip.write_videofile(f"{name1}_{name2}_morph_video.mp4",codec="libx264")

#play("oytun","saruman")
part1_main()