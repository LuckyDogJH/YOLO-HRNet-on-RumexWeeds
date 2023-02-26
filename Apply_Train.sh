#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J YOLO-HRNet-SingleCLS
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 32GB of system-memory
#BSUB -R "rusage[mem=32GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s210313@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the outputs and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/11.3
module load python3/3.9.11
pip install -r requirements.txt
# python3 convert_XML_to_COCO_in_YOLO.py --root='/work3/s210313/RumexWeeds' --target_dir='/work3/s210313/RumexWeeds_root_YOLOtxt'
python3 train.py --batch-size=6 --optimizer='Adam' --weights='./yolov5x.pt' --hyp='./hyps/hyp.scratch-med.yaml' --img=640 --patience=100 --epochs=50 --cos-lr --roi_output_size=16 --root_conf_thresh=0.0 --seed=1
# rm -rf /work3/s210313/RumexWeeds_YOLOtxt

