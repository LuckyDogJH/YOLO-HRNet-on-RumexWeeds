# Author: Jiahao Li
# CreatTime: 2022/10/27
# FileName: 
# Description: None

import os
import numpy as np
from tqdm import tqdm
from lxml import etree
import argparse
import shutil
import json
import time

def parse_xml_to_dict(xml):
    if len(xml) == 0:
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)
        if child.tag != 'image':
            result[child.tag] = child_result[child.tag]
        else:
            result[child.items()[0][1]] = dict()
            helper = result[child.items()[0][1]]

            helper['size'] = dict()
            for key, value in child.items():
                if key == 'name':
                    continue
                else:
                    helper['size'][key] = value

            helper['object'] = []
            for box_tag in child.iterfind('box'):
                each_bbox_info = dict()
                for key, value in box_tag.items():
                    each_bbox_info[key] = value
                helper['object'].append(each_bbox_info)

            helper['root_info'] = []
            for center_tag in child.iterfind('ellipse'):
                each_ellipse_info = dict()
                for key, value in center_tag.items():
                    each_ellipse_info[key] = value
                helper['root_info'].append(each_ellipse_info)
    return {xml.tag: result}


def convert(size, bbox):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    cx = (bbox[0] + bbox[1]) / 2.0
    cy = (bbox[2] + bbox[3]) / 2.0
    w = bbox[1] - bbox[0]
    h = bbox[3] - bbox[2]
    cx = round(cx * dw, 6)
    w = round(w * dw, 6)
    cy = round(cy * dh, 6)
    h = round(h * dh, 6)
    return [cx, cy, w, h]

def convert_root(size, root_info):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    cx, cy, rx, ry = root_info[0], root_info[1], root_info[2], root_info[3]
    cx = round(cx * dw, 6)
    rx = round(rx * dw, 6)
    cy = round(cy * dh, 6)
    ry = round(ry * dh, 6)
    return [cx, cy, rx, ry]

def match_root(root_info, bbox_info, image_path):
    assert len(bbox_info) > 0, f"'{image_path}' has a wrong annotation, duplicated root point"
    bbox_info = np.array(bbox_info)
    root_center = np.array((root_info[1], root_info[2]))
    bbox_center = np.array(((bbox_info[:, 1] + bbox_info[:, 2])/2.0, (bbox_info[:, 3] + bbox_info[:, 4])/2.0)).T
    match_ind = np.sqrt(np.sum(np.abs(np.power(root_center, 2) - np.power(bbox_center, 2)), 1)).argmin()
    return match_ind


def main(args):
    root = args.root
    target_root = args.target_dir

    if not os.path.exists(target_root):
        os.mkdir(target_root)
        os.mkdir(os.path.join(target_root, 'images'))
        os.mkdir(os.path.join(target_root, 'labels'))

    if not os.path.exists(root):
        raise FileNotFoundError(f"❌ RumexWeeds data dose not exist in path: {root}")

    with open('./rumex_weeds_classes.json') as f:
        class_dict = json.load(f)


    for list_file in ['train.txt', 'val.txt','test.txt']:
        with open(os.path.join(root, list_file), encoding='utf-8') as f:
            txt_content = f.readlines()
            img_list = sorted([os.path.join(root, img_name.strip('\n')) for img_name in txt_content])

        filtered_img_list = []
        annotation_dict = dict()
        path_set = set()
        for img_path in img_list:
            if not os.path.exists(img_path):
                print(f"⚠️ Warning: not found '{img_path}', skip this annotation file.")
                continue
            dir_path = '/'.join(img_path.split('/')[:-2])

            # parse .xml file in each directory
            if not dir_path in path_set:
                path_set.add(dir_path)
                with open(os.path.join(dir_path, 'annotations.xml')) as f:
                    xml_string = f.read()
                    xml = etree.fromstring(xml_string)
                    data = parse_xml_to_dict(xml)['annotations']
                    annotation_dict[dir_path] = data

            # # remove those img without bbox
            # if annotation_dict[dir_path][img_path.split('/')[-1]]['object']:
            filtered_img_list.append(img_path)
        time.sleep(0.2)
        print(f"✅ All {list_file.split('.')[0]} Images are Checked ")
        assert len(img_list) > 0, f"in '{root}' file does not find any information."

        image_dir = os.path.join(target_root + '/images', list_file.split('.')[0])
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)
        label_dir = os.path.join(target_root + '/labels', list_file.split('.')[0])
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)

        for image_path in tqdm(filtered_img_list):
            folder_name = '/'.join(image_path.split('/')[:-2])
            image_name = image_path.split('/')[-1]
            annotation = annotation_dict[folder_name][image_name]
            txt_file = open(os.path.join(label_dir, image_name.split('.')[0] + '.txt'), 'w')
            bbox_info, root_info = [], []
            if len(annotation['object']):
                # Image size
                w, h = float(annotation['size']['width']), float(annotation['size']['height'])

                for i in range(len(annotation['object'])):
                    category = class_dict[annotation['object'][i]['label']]
                    # Bbox info
                    xmin, xmax, ymin, ymax = int(float(annotation['object'][i]['xtl'])), int(float(annotation['object'][i]['xbr'])), int(float(annotation['object'][i]['ytl'])), int(float(annotation['object'][i]['ybr']))
                    bbox = [xmin, xmax, ymin, ymax]
                    bbox_info.append([category] + bbox)

                for i in range(len(annotation['root_info'])):
                    category = 1
                    root_xmin, root_ymin, root_w, root_h = float(annotation['root_info'][i]['cx']), float(annotation['root_info'][i]['cy']), float(annotation['root_info'][i]['rx']), float(annotation['root_info'][i]['ry'])
                    cx, cy, rx, ry = root_xmin + root_w/2.0, root_ymin + root_h/2.0, root_w/2.0, root_h/2.0
                    weeds_root = [cx, cy, rx, ry]
                    root_info.append([category] + weeds_root)

                # Match Root
                matched_res = []
                for i in range(len(root_info)):
                    matched_ind = match_root(root_info[i], bbox_info, image_path)
                    matched_res.append(bbox_info[matched_ind] + root_info[i])
                    bbox_info.remove(bbox_info[matched_ind])
                if len(bbox_info) > 0:  # No root point
                    for i in range(len(bbox_info)):
                        matched_res.append(bbox_info[i] + [0 for _ in range(5)])

                for match in matched_res:
                    # convert bbox to normalized [center_x, center_y, bbox_w, bbox_h]
                    normalized_bbox_info = convert((w, h), match[1:5])
                    # convert root point to normalized [center_x, center_y, radius_w, radius_h]
                    normalized_root_info = convert_root((w, h), match[6:])
                    final_res = [match[0]] + normalized_bbox_info + [match[5]] + normalized_root_info
                    txt_file.write(' '.join([str(j) for j in final_res]) + '\n')
            txt_file.close()
            shutil.copy(image_path, image_dir)

    print('👌 All xml Annotation are converted to YOLO txt format')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='../RumexWeeds_root', help='Dataset root path')
    parser.add_argument('--target_dir', default='../RumexWeeds_YOLOtxt', help='target fold for converted data')
    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)


