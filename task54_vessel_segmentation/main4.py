import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2

IMAGEPATH = "V.nii"
GTPATH = "V_seg.nii"

def dice_score_one_label(seg,seg_gt):
    diff = seg * seg_gt
    return 2 * np.count_nonzero(diff) / (seg[seg==1].size + seg_gt[seg_gt==1].size)

def dice_score_all_labels(seg,seg_gt):
    diff = seg - seg_gt
    return 2 * np.count_nonzero(diff==0) / (seg.size + seg_gt.size)

def difference(mean, candidate):
    return abs(mean - candidate) / (mean + 0.0000001)

def prepare_neighbors3d(n=26):
    locations = []
    if n == 26:
        for i in range(-1,2):
            for j in range(-1,2):
                for k in range(-1,2):
                    if i == 0 and j == 0 and k == 0:
                        continue
                        # All 0 means pixel itself.
                    locations.append([i,j,k])
    if n == 6:
        for i in range(-1,2):
            for j in range(-1,2):
                for k in range(-1,2):
                    if (i == 0 and j == 0 and k != 0) or (i == 0 and j != 0 and k == 0) or (i != 0 and j == 0 and k == 0):
                        # All 0 means pixel itself.
                        locations.append([i,j,k])
    return locations

def prepare_neighbors2d(n=8):
    locations = []
    if n == 8:
        for i in range(-1,2):
            for j in range(-1,2):
                if i == 0 and j == 0:
                    continue
                    # All 0 means pixel itself.
                locations.append([i,j])
    if n == 4:
        for i in range(-1,2):
            for j in range(-1,2):
                if i == 0 or j == 0:
                    locations.append([i,j])
    return locations

def region_growing3d(img, seeds = [[12,15,17]], n_count = 26, eps = 0.20):
    h, w, d = img.shape
    segmap = np.zeros_like(img)
    # Mark region as 2
    label = 1
    next = []
    neighbor_loc = prepare_neighbors3d(n_count)
    for seed in seeds:
        sx,sy,sz = seed[0],seed[1],seed[2]
        region_size = 1
        region_sum = img[sx,sy,sz]
        segmap[sx,sy,sz] = label
        next.append(seed)
        while len(next) > 0: # If there are still current voxels.
            #print(len(next),region_size)
            current = next.pop(0)
            x = current[0]
            y = current[1]
            z = current[2]
            segmap[x,y,z] = label
            for n_loc in neighbor_loc:
                # Neighbor pixels.
                xx = n_loc[0] + x
                yy = n_loc[1] + y
                zz = n_loc[2] + z

                if xx < 0 or yy < 0 or zz < 0 or xx >= h or yy >= w or zz >= d:
                    # Check boundaries.
                    continue

                region_mean = region_sum / region_size
                diff = difference(region_mean, img[xx,yy,zz])
                
                if diff < eps and segmap[xx,yy,zz] == 0:
                    region_size += 1
                    region_sum += img[xx,yy,zz]
                    segmap[xx,yy,zz] = label
                    next.append([xx,yy,zz])

    return segmap

def region_growing2d(img, seeds = [[12,15]], n_count = 8, eps = 0.20):
    h, w = img.shape
    segmap = np.zeros_like(img)
    # Mark region as 2
    label = 1
    next = []
    neighbor_loc = prepare_neighbors2d(n_count)
    for seed in seeds:
        sx,sy = seed[0],seed[1]
        region_size = 1
        region_sum = img[sx,sy]
        segmap[sx,sy] = label
        next.append(seed)
        while len(next) > 0: # If there are still current voxels.
            #print(len(next),region_size)
            current = next.pop(0)
            x = current[0]
            y = current[1]
            segmap[x,y] = label
            for n_loc in neighbor_loc:
                # Neighbor pixels.
                xx = n_loc[0] + x
                yy = n_loc[1] + y

                if xx < 0 or yy < 0 or xx >= h or yy >= w:
                    # Check boundaries.
                    continue

                region_mean = region_sum / region_size
                diff = difference(region_mean, img[xx,yy])
                
                if diff < eps and segmap[xx,yy] == 0:
                    region_size += 1
                    region_sum += img[xx,yy]
                    segmap[xx,yy] = label
                    next.append([xx,yy])
    return segmap

def region_growing_zsliced(img,seeds,n_count):
    imgT = img.transpose((2,0,1))
    segs = []
    for im2d in imgT:
        seg = region_growing2d(im2d,seeds,n_count,0.20)
        segs.append(seg)
    segs = np.array(segs)
    return segs.transpose((1,2,0))

def segmentation_task1():
    img_nii = nib.load(IMAGEPATH)
    img_gt_nii = nib.load(GTPATH)

    img = np.array(img_nii.dataobj).astype(np.float64)
    img_gt = np.array(img_gt_nii.dataobj)

    seeds = [[169,70],[23,17],[46,62],[70,25],[139,12]]
    segmented = region_growing_zsliced(img,seeds=seeds, n_count=8)
    np.save("segmented2d_8N.npy",segmented)

    segmented = np.load("segmented2d_8N.npy")
    print("Task 1: Dice Score All Labels =",dice_score_all_labels(segmented,img_gt))
    print("Task 1: Dice Score One Label(1) =",dice_score_one_label(segmented,img_gt))
    nft_img = nib.Nifti1Image(segmented, affine=np.eye(4))
    nib.save(nft_img, "segmented2d_8N.nii")

def segmentation_task2():
    img_nii = nib.load(IMAGEPATH)
    img_gt_nii = nib.load(GTPATH)

    img = np.array(img_nii.dataobj).astype(np.float64)
    img_gt = np.array(img_gt_nii.dataobj)

    seeds = [[169,70],[23,17],[46,62],[70,25],[139,12]]
    segmented = region_growing_zsliced(img,seeds=seeds, n_count=4)
    np.save("segmented2d_4N.npy",segmented)

    segmented = np.load("segmented2d_4N.npy")
    print("Task 2: Dice Score All Labels =",dice_score_all_labels(segmented,img_gt))
    print("Task 2: Dice Score One Label(1) =",dice_score_one_label(segmented,img_gt))
    nft_img = nib.Nifti1Image(segmented, affine=np.eye(4))
    nib.save(nft_img, "segmented2d_4N.nii")

def segmentation_task3():
    img_nii = nib.load(IMAGEPATH)
    img_gt_nii = nib.load(GTPATH)

    img = np.array(img_nii.dataobj).astype(np.float64)
    img_gt = np.array(img_gt_nii.dataobj)

    seeds = [[169,70,50],[23,17,50],[46,62,50],[70,25,50],[139,12,50]]
    segmented = region_growing3d(img,seeds=seeds, n_count=26)
    np.save("segmented3d_26N.npy",segmented)

    segmented = np.load("segmented3d_26N.npy")
    print("Task 3: Dice Score All Labels =",dice_score_all_labels(segmented,img_gt))
    print("Task 3: Dice Score One Label(1) =",dice_score_one_label(segmented,img_gt))
    nft_img = nib.Nifti1Image(segmented, affine=np.eye(4))
    nib.save(nft_img, "segmented3d_26N.nii")

def segmentation_task4():
    img_nii = nib.load(IMAGEPATH)
    img_gt_nii = nib.load(GTPATH)

    img = np.array(img_nii.dataobj).astype(np.float64)
    img_gt = np.array(img_gt_nii.dataobj)

    seeds = [[169,70,50],[23,17,50],[46,62,50],[70,25,50],[139,12,50]]
    segmented = region_growing3d(img,seeds=seeds, n_count=6)
    np.save("segmented3d_6N.npy",segmented)

    segmented = np.load("segmented3d_6N.npy")
    print("Task 4: Dice Score All Labels =",dice_score_all_labels(segmented,img_gt))
    print("Task 4: Dice Score One Label(1) =",dice_score_one_label(segmented,img_gt))
    nft_img = nib.Nifti1Image(segmented, affine=np.eye(4))
    nib.save(nft_img, "segmented3d_6N.nii")

segmentation_task1()
segmentation_task2()
segmentation_task3()
segmentation_task4()