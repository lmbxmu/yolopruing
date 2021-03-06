Pruning yolov3 and train on 6 class of COCO2014: person, bicycle, car, motorbike, bus, truck.

## Installation
##### Download pretrained weights
    $ cd weights/
    $ bash download_weights.sh

##### Download COCO dataset
    $ cd data/
    $ bash get_coco_dataset.sh

##### Split COCO dataset

```
$ python3 python3 extract.py --save_path /media/coco_class_6 --data_dir /media/coco/
--save_path           target path to save specific class of coco dataset
--data_dir           path of coco dataset
```

## Train

#### Example (COCO)
To train on COCO using a Darknet-53 backend pretrained on ImageNet run: 
```shell
python3 train.py --data_config config/coco.data  --job_dir ./experiments/class_6_pr_1 --pretrained_weights weights/darknet53.conv.74  --pr_cfg 0.4 0.4 0.5 0.5 0.6
```

#### Training log
```
---- [Epoch 7/100, Batch 7300/14658] ----
+------------+--------------+--------------+--------------+
| Metrics    | YOLO Layer 0 | YOLO Layer 1 | YOLO Layer 2 |
+------------+--------------+--------------+--------------+
| grid_size  | 16           | 32           | 64           |
| loss       | 1.554926     | 1.446884     | 1.427585     |
| x          | 0.028157     | 0.044483     | 0.051159     |
| y          | 0.040524     | 0.035687     | 0.046307     |
| w          | 0.078980     | 0.066310     | 0.027984     |
| h          | 0.133414     | 0.094540     | 0.037121     |
| conf       | 1.234448     | 1.165665     | 1.223495     |
| cls        | 0.039402     | 0.040198     | 0.041520     |
| cls_acc    | 44.44%       | 43.59%       | 32.50%       |
| recall50   | 0.361111     | 0.384615     | 0.300000     |
| recall75   | 0.222222     | 0.282051     | 0.300000     |
| precision  | 0.520000     | 0.300000     | 0.070175     |
| conf_obj   | 0.599058     | 0.622685     | 0.651472     |
| conf_noobj | 0.003778     | 0.004039     | 0.004044     |
+------------+--------------+--------------+--------------+
Total Loss 4.429395
---- ETA 0:35:48.821929
```

#### Tensorboard
Track training progress in Tensorboard:
* Initialize training
* Run the command below
* Go to http://localhost:6006/

```
$ tensorboard --logdir='logs' --port=6006
```

## Inference

| Model(link)                                                  | Cfg                    | Parameter(prune) | FLOPs(prune)   | Map    |
| ------------------------------------------------------------ | ---------------------- | ---------------- | -------------- | ------ |
| [Baseline](https://drive.google.com/file/d/1C2mjAH5bjvk-zRf5dytGoVrRTQehDQpn/view?usp=sharing) | 0 0 0 0 0              | 61.5M(0%)        | 12435.1M(0%)   | 0.4208 |
| [Pr1](https://drive.google.com/file/d/1CRZcXqNFhWmFBeNXvL-v1wUlhGajGj9-/view?usp=sharing) | 0.4 0.4 0.5 0.5 0.6    | 11.8M(81.9%)     | 3204.9M(74.3%) | 0.3751 |
| [Pr2](https://drive.google.com/file/d/1hsUUBLoqfHhrnufc2WIaK6y4tYLgZ0Xm/view?usp=sharing) | 0.3 0.3 0.4 0.4 0.5    | 17.8M(71.1%)     | 4507.5M(63.8%) | 0.4060 |
| [Pr3](https://drive.google.com/file/d/1_f3PH5mVpoQii2HnVxb206K88_ub0KV9/view?usp=sharing) | 0.6 0.6 0.55 0.55 0.5  | 14.3M(76.8%)     | 2682.7M(78.5%) | 0.3648 |
| [Pr4](https://drive.google.com/file/d/1cJ_3Xfsf5Xyldhwp7gJV3ZU-qqEP3OWt/view?usp=sharing) | 0.65 0.65 0.6 0.6 0.55 | 11.5M(82.4%)     | 2152.7M(82.7%) | 0.3573 |
| [Pr5](https://drive.google.com/file/d/1_NMK9uIwRkKsIK4CEW7A-8SrbDZbiesE/view?usp=sharing) | 0.5 0.55 0.55 0.6 0.6  | 10.0M(83.8%)     | 2403.6M(81.7%) | 0.3641 |

Evaluates the model on COCO test.

    $ python3 test.py --weights_path experiments/baseline/best_ckpt.pth --pr_cfg 0 0 0 0 0 

```
$ python3 test.py --weights_path experiments/pr_1/best_ckpt.pth --pr_cfg 0.4 0.4 0.5 0.5 0.6 
```

```
$ python3 test.py --weights_path experiments/pr_2/best_ckpt.pth --pr_cfg 0.3 0.3 0.4 0.4 0.5 
```

```
$ python3 test.py --weights_path experiments/pr_3/best_ckpt.pth --pr_cfg 0.6 0.6 0.55 0.55 0.5
```

```
$ python3 test.py --weights_path experiments/pr_4/best_ckpt.pth --pr_cfg 0.65 0.65 0.6 0.6 0.55
```

```
$ python3 test.py --weights_path experiments/pr_5/best_ckpt.pth --pr_cfg 0.5 0.55 0.55 0.6 0.6
```

#### Other Arguments

```shell
  -h, --help            Show this help message and exit
  --epochs EPOCHS	Num of epochs. default:100
  --batch_size 	Size of each image batch. default:8
  --gradient_accumulations Number of gradient accums before step. default:2
  --model_def		Path to model definition file. default:"config/yolov3.cfg"
  --data_config 	Path to data config file. default:"config/coco.data"
  --pretrained_weights  if specified starts from checkpoint model
  --n_cpu 	number of cpu threads to use during batch generation.default:8 
  --img_size 	size of each image dimension. default:8
  --evaluation_interval interval evaluations on validation set. default:1
  --compute_map 	if True computes mAP every tenth batch. default:False
  --lr 	learning rate. default:0.001
  --job_dir JOB_DIR     The directory where the summaries will be stored.
                        default:./experiments
  --lr_type   lr scheduler. default: step. optional:exp/cos/step/fixed
  --pr_cfg 		prune percentage cofiguration of blocks. eg: 0.3 0.3 0.3 0.3 0.3
  --random_rule		random_rule of preserving filters defaut:random_pretrain. optional:l1_pretrain
```

