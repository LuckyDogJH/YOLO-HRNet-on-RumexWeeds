# YOLO-HRNet-on-RumexWeeds
This is the repository of YOLO-HRNet single-class implementation on RumexWeeds Dataset.

# Usage
## Environment
* Python >= 3.6
* Pytorch >= 1.8.0
* Detailed requirments are shown in `requirements.txt`

## Data preparation
For RumexWeeds data, please download from `to do`. The original annotations are saved in xml format. We need to convert them into txt format. Please run
```
# python3 convert_XML_to_COCO_in_YOLO.py --root=$your_data_path --target_dir='../RumexWeeds_root_YOLOtxt'
```
