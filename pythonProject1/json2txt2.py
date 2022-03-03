import os
import json
from pathlib import Path

import PIL.Image as Image
from tqdm import tqdm

json_dir = '/home/trajic/Annotation/annotations.json'  # json文件路径
data_dir = '/home/trajic/TsingHua100k/data'
out_dir = '/home/trajic/Annotation/'  # 输出的 txt 文件路径
category_list = {}
train_list = []
test_list = []
other_list = []

def main():
    global category_list
    with open(json_dir, 'r') as load_f:
        content = json.load(load_f)
        # print(type(content))
        # print(content.keys())
        # print(type(content["imgs"]))

    for item in content:
        if item == "imgs":
            for number in tqdm(content[item].keys()):
                img_name = f"{number}.jpg"
                folder_path = content[item][number]["path"].split("/")[0]

                if folder_path == "train":
                    index = len(train_list)
                    train_list.append(number)
                    folder_path = "train"
                if folder_path == "test":
                    index = len(test_list)
                    test_list.append(number)
                    folder_path = "test"
                if folder_path == "other":
                    index = len(other_list)
                    other_list.append(number)
                    folder_path = "other"

                img_path = Path(f"{data_dir}/{folder_path}")
                img_path = os.path.join(img_path, img_name)

                img = Image.open(img_path)
                img = img.convert("RGB")

                img_width, img_height = img.size

                image_name = f"{index}.jpg"
                image_path = Path(f"{out_dir}/images/{folder_path}")
                image_path.mkdir(parents=True, exist_ok=True)
                img.save(os.path.join(image_path, image_name))

                labels_path = Path(f"{out_dir}/labels/{folder_path}")
                labels_path.mkdir(parents=True, exist_ok=True)

                label_name = f"{index}.txt"

                with (labels_path / label_name).open(mode="w") as label_file:
                    objects = content[item][number]["objects"]
                    for object in objects:
                        category = object["category"]
                        category_id = addnewcategory(category)

                        bbox = object["bbox"]
                        xmin = bbox["xmin"]
                        xmax = bbox["xmax"]
                        ymin = bbox["ymin"]
                        ymax = bbox["ymax"]

                        bbox_width = xmax - xmin
                        bbox_height = ymax - ymin

                        label_file.write( f"{category_id} {(xmin + bbox_width / 2) / img_width} {(ymin + bbox_height / 2) / img_height} {(bbox_width) / img_width} {(bbox_height) / img_height}\n")

    category_list_path = Path(f"{out_dir}/categories.txt")
    with category_list_path.open(mode="w") as category_file:
        print(f"Total number of categories: {len(category_list)}")
        category_list_new = sorted(category_list.items(), key=lambda k: k[1])
        category_file.write(f"Total categories: {len(category_list)}")
        print(category_list_new)
        for key, value in category_list.items():
            category_file.write(f"{key} : {value}\n")


def addnewcategory(category):
    global category_list
    if category not in category_list.keys():
        category_list[category] = len(category_list)
        return category_list[category]
    else:
        return category_list[category]


if __name__ == '__main__':
    main()