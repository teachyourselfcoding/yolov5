from pathlib import Path

import numpy as np

from utils.plots import plot_one_box, colors
from yolov5_detector import YOLOv5Detector
import os

def img_process(img, img_name, annotation_folder):
    img_name = img_name.split(".")[0]
    results = find_tag(annotation_folder, img_name)
    height = img.shape[0]
    width = img.shape[1]
    for box in results:
        bbox_x = float(box['box'][0])*width
        bbox_y = float(box['box'][1])*height
        bbox_width = float(box['box'][2])*width
        bbox_height = float(box['box'][3])*height
        bbox = [bbox_x - bbox_width/2, bbox_y - bbox_height/2, bbox_x + bbox_width/2, bbox_y + bbox_height/2]
        color = list(np.random.choice(range(256), size=3))
        color = (int(color[0]), int(color[1]), int(color[2]))
        plot_one_box(bbox, img, label=box['class'], color=color,line_width=1)
        box['box'] = bbox
    return img, results

def find_tag(annotation_folder, img_name):
    annotation_path = os.path.join(f"{annotation_folder}/{img_name}.txt")
    results = []
    with open(annotation_path) as f:
        for line in f:
            line = line.split(" ")
            dict = {}
            dict['class'] = line[0]
            dict['box'] = [line[1], line[2], line[3], line[4]]
            results.append(dict)
    return results


if __name__ == '__main__':
    find_tag("/home/trajic/Annotation/labels/val/", "0")
