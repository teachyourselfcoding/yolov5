'''
SLIDESHOW USING OPENCV-PYTHON
It is a simple commandline based python script to make a slideshow instantly with blur effect and background
music with next, previous and pause feature.
Features-

   Slideshow has the following buttons-
        (1) 'a' - Previous Image
        (2) 's' - Pause the Slideshow
        (3) 'd' - Next Image
        (4) 'q' - Exit the Slideshow

'''

import os
import sys
import cv2
import numpy as np
from img_process import img_process
from yolov5_detector import YOLOv5Detector
CONFIDENCE=0.4
cat_map = {}
all_cat_count = {}
all_cat_map = {}
final_map = 0

def getcatmap():
    global all_cat_count
    global cat_map
    global all_cat_map
    dect = cat_map
    true = all_cat_count
    map = 0
    no_dect = 0
    no_true = 0

    for cat in true.keys():
        no_true = no_true + true[cat]
        if cat in dect.keys():
            no_dect = no_dect + dect[cat]
            try:
                p = dect[cat]/true[cat]
            except:
                p = 0
        else:
            p = 0
        if cat in all_cat_map.keys():
            all_cat_map[cat] = p
        else:
            all_cat_map[cat] = p
    try:
        final_map = no_dect/no_true
    except:
        final_map = 0
    print(all_cat_map)
    print(final_map)



def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def getoverlap(true_bbox, true_cat, dect_bbox):
    max_iou = 0
    for box in dect_bbox:
        if box['class'] == true_cat:
            iou = bb_intersection_over_union(true_bbox, box['box'])
            if iou > max_iou:
                max_iou = iou
    return max_iou



def slideshow():
    image_path = "/media/trajic/FIELDTEST/Obeject_Detect/BDD100K/Raw/purenight/images/test"
    annotation_path = "/media/trajic/FIELDTEST/Obeject_Detect/BDD100K/Raw/purenight/labels/test"
    image_names = os.listdir(image_path)  # making a list of image names
    no_of_images = len(image_names)  # calculating number of images
    weights_path = "/home/trajic/yolov5(old)/runs/train/bdd50%syntheticday12nov/weights/best.pt"
    image_id = 0  # initially we are starting from 0th image (1st image)

    detector = YOLOv5Detector(weights_path, CONFIDENCE)

    item_both = 0
    item_dect_only =0
    item_true_only =0

    Total_dect_only = 0
    Total_truth_only = 0
    Total_dect_and_truth = 0

    Pause = False
    Single_play_back = False
    Single_play_forward = False

    while image_id < no_of_images:  # iterating until we get at the end of the images list
            print(Pause, Single_play_back, Single_play_forward)
            print(item_dect_only, item_true_only, item_both)
            print(Total_dect_only, Total_truth_only, Total_dect_and_truth)
            if Pause == False:
                image = image_names[image_id]

                img_name = os.path.join(image_path, image)
                img = cv2.imread(img_name)  # reading an image as per the index value

                height = 640  # height of our resized image
                dim = (int((height / img.shape[0]) * img.shape[1]), height)  # manipulating width by maintaining constant ratio

                img = cv2.resize(img, dim)  # resizing our image
                img_copy = img.copy()

                dect_img, dect_bbox = detector(img)

                true_img, true_bbox = img_process(img_copy, image, annotation_path)

                print(dect_bbox, true_bbox)
                vis = np.hstack((dect_img,true_img))
                vis, item_both, item_dect_only, item_true_only = process_bbox(dect_bbox, true_bbox, vis)

                try:
                    Current_Precision = (float(item_both)/float(item_both + item_dect_only)) * 100
                except:
                    Current_Precision = 0
                try:
                    Current_Recall = (float(item_both)/float(item_both + item_true_only))*100
                except:
                    Current_Recall = 0


                if Single_play_back == False:
                    Total_dect_only = Total_dect_only + item_dect_only
                    Total_truth_only = Total_truth_only + item_true_only
                    Total_dect_and_truth = Total_dect_and_truth + item_both

                try:
                    Total_Precision = (float(Total_dect_and_truth)/float(Total_dect_and_truth+Total_dect_only))*100
                except: Total_Precision = 0

                try:
                    Total_Recall = (float(Total_dect_and_truth)/float(Total_dect_and_truth+Total_truth_only))*100
                except:
                    Total_Recall = 0


                cv2.putText(vis, f"Current Precision: {round(Current_Precision,4)} %", (500, 510),  cv2.FONT_HERSHEY_DUPLEX, 0.75, (0,255,0), 1)
                cv2.putText(vis, f"Current Recall: {round(Current_Recall,4)} %", (500,540),  cv2.FONT_HERSHEY_DUPLEX, 0.75, (0,255,0), 1)

                cv2.putText(vis, f"Total Precision: {round(Total_Precision,4)} %", (500, 570),  cv2.FONT_HERSHEY_DUPLEX, 0.75, (0,255,0), 1)
                cv2.putText(vis, f"Total Recall: {round(Total_Recall,4)} %", (500, 600),  cv2.FONT_HERSHEY_DUPLEX, 0.75, (0,255,0), 1)

                if Single_play_back == False and Single_play_forward == False:
                    image_id += 1  # If no keys are pressed, then image_id incremented for next image

                if Single_play_back == True:
                    Pause = True
                    Single_play_back = False

                if Single_play_forward == True:
                    Pause = True
                    Single_play_forward = False



            cv2.imshow('Slideshow', vis)  # displaying clear image
            key = cv2.waitKey(1)  # taking key from user with 1000 ms delay
            if key == ord('q'):  # If 'q' pressed (User wants to quit when slideshow is displaying clear image)
                sys.exit(0)
            if key == ord('s'):  # If 's' pressed (User wants to pause when slideshow is displaying clear image)
                if Pause == True:
                    Pause = False
                else:
                    Pause = True
                    pause_key = cv2.waitKey()


            if key == ord('a') and image_id != 0:  # If 'a' pressed (Previous Image)
                print("back")
                image_id -= 1  # decrementing image_id to get previous image id
                Total_dect_only = Total_dect_only - item_dect_only
                Total_truth_only = Total_truth_only - item_true_only
                Total_dect_and_truth = Total_dect_and_truth - item_both
                Pause = False
                Single_play_back = True
                continue


            if key == ord('d'):  # If 'd' pressed (Next Image)
                image_id += 1  # incrementing image_id to get next image id
                # Total_dect_only = Total_dect_only + item_dect_only
                # Total_truth_only = Total_truth_only + item_true_only
                # Total_dect_and_truth = Total_dect_and_truth + item_both
                Pause = False
                Single_play_forward = True
                continue

    Pause == False  #Pause when last image reached
    print("final map")
    print(final_map)
    print( f"Total image count: {no_of_images} \n "
           f"Total Precision: {(float(Total_dect_and_truth)/float(Total_dect_and_truth+Total_dect_only))*100} \n%" 
           f"Total Recall: {(float(Total_dect_and_truth)/float(Total_dect_and_truth+Total_truth_only))*100} %")


def process_bbox(dect_bbox, true_bbox, img):
    dect_bbox_categories = []
    true_bbox_categories = []
    item_both=0
    item_dect_only=0
    item_true_only=0
    height, width = img.shape[0], img.shape[1]
    text_x = 50
    text_y = 30
    for index, box in enumerate(dect_bbox):
        dect_bbox_categories.append(box['class'])
        cv2.putText(img, f"Category: {box['class']} , Confidence: {box['confidence']}", (text_x, text_y*(index+1)), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0,255,0), 1)
    for index, box in enumerate(true_bbox):
        true_bbox_categories.append(box['class'])
        true_cat = box['class']
        true_box = box['box']
        max_iou = getoverlap(true_box, true_cat, dect_bbox)
        if max_iou > 0.5:
            if true_cat in cat_map.keys():
                cat_map[true_cat] = cat_map[true_cat] + 1
            else:
                cat_map[true_cat] = 0

        if true_cat in all_cat_count.keys():
            all_cat_count[true_cat] = all_cat_count[true_cat] + 1
        else:
            all_cat_count[true_cat] = 0
        getcatmap()

        cv2.putText(img, f"Category: {box['class']} ", (700, text_y * (index + 1)), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 1)

    for item in dect_bbox_categories:
        if item in true_bbox_categories:
            item_both = item_both +1
            true_bbox_categories.remove(item)
        else:
            item_dect_only = item_dect_only + 1
    item_true_only = len(true_bbox_categories)

    return img, item_both, item_dect_only, item_true_only




def main():
    print(''' 
                                -------------------
                                |    SLIDESHOW    |
                                -------------------

    ~~~~~~~~~~~~~~
    || Hot Keys ||
    ~~~~~~~~~~~~~~
    1.  'a' for previous picture
    2.  'd' for next picture
    3.  's' to pause the slideshow
    4.  'q' to exit the slideshow
    ''')

    slideshow()


if __name__ == "__main__":
    main()