python3 train.py --data_config config/coco.data  --pretrained_weights weights/darknet53.conv.74 --checkpoint_interval 10

python3 test.py --weights_path weights/yolov3_ckpt_99.pth

python3 detect.py --weights_path weights/yolov3_origin.pth --image_folder data/samples/

python3 detect.py --weights_path weights/best_ckpt.pth --image_folder data/samples/ --pr_cfg 0.4 0.4 0.5 0.5 0.6

python3 test.py --weights_path weights/yolov3_prune.pth  --pr_cfg 0.3 0.3 0.4 0.4 0.5

python3 test.py --weights_path experiments/class_6_pr_1/best_ckpt.pth --pr_cfg 0.4 0.4 0.5 0.5 0.6

python3 test.py --weights_path experiments/class_6_baseline/yolov3_ckpt_last.pth



python3 train.py --data_config config/coco.data  --job_dir ./experiments/baseline --pretrained_weights weights/darknet53.conv.74 --pr_cfg 0 0 0 0 0

python3 train.py --data_config config/coco.data  --job_dir ./experiments/pr_1 --pretrained_weights weights/darknet53.conv.74 --pr_cfg 0.3 0.3 0.3 0.3 0.3

python3 train.py --data_config config/coco.data  --job_dir ./experiments/class_6_baseline --pretrained_weights weights/darknet53.conv.74 --lr_type ori

python3 train.py --data_config config/coco.data  --job_dir ./experiments/class_6_pr_1 --pretrained_weights weights/darknet53.conv.74 --lr_type ori --pr_cfg 0.4 0.4 0.5 0.5 0.6

python3 train.py --data_config config/coco.data  --job_dir ./experiments/class_6_pr_2 --pretrained_weights weights/darknet53.conv.74 --lr_type ori --pr_cfg 0.3 0.3 0.4 0.4 0.5

python3 train.py --data_config config/coco.data  --job_dir ./experiments/class_6_pr_3 --pretrained_weights weights/darknet53.conv.74 --lr_type ori --pr_cfg 0.15 0.2 0.25 0.3 0.35

python3 extract_new.py --save_path /media/disk2/zyx/coco_class_2 --data_dir /media/disk2/zyx/coco