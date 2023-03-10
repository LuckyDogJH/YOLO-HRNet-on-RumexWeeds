# YOLO-HRNet-on-RumexWeeds
This is the repository of YOLO-HRNet single-class implementation on RumexWeeds Dataset.
![YOLO_HRNet.png](YOLO_HRNet.png)

# Usage
## Environment
* Python >= 3.6
* Pytorch >= 1.8.0
* Detailed requirments are shown in `requirements.txt`

## Data preparation
Please download [RumexWeeds Dataset](https://figshare.com/s/287873a5ac9297f181cc). We use \$your-data-path\$ to denote your saved data path. The original annotations are saved in xml format. We need to convert them into txt format. Please run
```
python3 convert_XML_to_COCO_in_YOLO.py --root=$your-data-path$ --target_dir='../RumexWeeds_root_YOLOtxt'
```
Please also change the train/val/test dataset information in `DataInfo.yaml`

## Training and Testing
### Training on RumexWeeds dataset
```
python3 train.py --batch-size=8 --optimizer='Adam' --weights='./yolov5m.pt' --hyp='./hyps/hyp.scratch-high.yaml' --img=640 --epochs=100 --cos-lr --roi_output_size=16
```

### Testing on RumenWeeds dataset
please put the trained model named `best.pt` under `./saved_model` folder
```
python3 Test.py --roi_output_size=16
```
You can download trained model for YOLO-HRNet via https://drive.google.com/drive/folders/1aLnNtIUV9C_HlPrJbyoQm6jGlTiPHmgW?usp=share_link

# Results Visualization
![P_C.jpg](P_C.jpg)
Link to [YOLO-Pose](https://github.com/LuckyDogJH/YOLO-Pose-on-RumexWeeds)

# Code References
* [YOLOv5 offical repository](https://github.com/ultralytics/yolov5)
* [HRNet offical repository](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
