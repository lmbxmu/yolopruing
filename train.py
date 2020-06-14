from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate
import utils.logger as utils


from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

def adjust_learning_rate(optimizer, epoch, opts):
    #Warmup
    if opts.lr_type == 'step':
        factor = epoch // 30

        if epoch >= 80:
            factor = factor + 1

        lr = opts.lr * (0.1 ** factor)

    elif opts.lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * opts.lr * (1 + math.cos(math.pi * epoch / opts.epochs))
    elif opts.lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = opts.lr * (decay ** (epoch // step))
    elif opts.lr_type == 'ori':
        lr = opts.lr
    print('=> lr: {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_darknet_model(model, oristate_dict, random_rule):
    state_dict = model.state_dict()
    last_select_index = None #Conv index selected in the previous layer
    index = 0
    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):

            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name + '.weight']
            #print('layer:{}\tori:{}\tcur:{}\t'.format(index,oriweight.size(),curweight.size()))
            index += 1
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                select_num = currentfilter_num
                if random_rule == 'random_pretrain':
                    select_index = random.sample(range(0, orifilter_num-1), select_num)
                    select_index.sort()
                else:
                    l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                    select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                    select_index.sort()
                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:
                    if oriweight.size(1) != curweight.size(1):
                        inchannel_index = random.sample(range(0, oriweight.size(1)), curweight.size(1))
                        inchannel_index.sort()
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(inchannel_index):
                                state_dict[name + '.weight'][index_i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name + '.weight'][index_i] = \
                                oristate_dict[name + '.weight'][i]

                last_select_index = select_index

            else:
                if last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][index_i][j]
                else:
                    state_dict[name + '.weight'] = oriweight
                last_select_index = None

    model.load_state_dict(state_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--pr_cfg", type=float, nargs='+', default=[0] * 5, help="target prune ratio for each residual block")
    parser.add_argument("--lr", type=float,  default=0.001, help="learning_rate")
    parser.add_argument("--random_rule", type=str,  default='random_pretrain', help="random_rule of preserving filters")
    parser.add_argument('--job_dir',type=str,default='experiments/',help='The directory where the summaries will be stored. default:./experiments')
    parser.add_argument("--lr_type", type=str, default="cos", help="lr_type")
    opt = parser.parse_args()
    logger = utils.get_logger(os.path.join(opt.job_dir + 'logger.log'))
    logger.info(opt)
    
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(3)
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    print(data_config)
    logger.info(data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    if not os.path.exists(opt.job_dir):
        os.makedirs(opt.job_dir)
    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)
            darknet_ckpt = model.state_dict()
            model = Darknet(opt.model_def, pr_cfg = opt.pr_cfg).to(device)
            load_darknet_model(model,darknet_ckpt,opt.random_rule)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(),lr=opt.lr)

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    best_map = 0

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        adjust_learning_rate(optimizer, epoch, opt)
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            #print(targets)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                #logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)
        logger.info(
                    'Epoch[{}] :\t'
                    'Total_loss {}\t'.format(
                        epoch, loss.item()
                    ))

        if epoch > 50:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            #logger.list_of_scalars_summary(evaluation_metrics, epoch)
            logger.info(
                    'Epoch[{}] :\t'
                    'Map {:.4f}\t'.format(
                        epoch, AP.mean()
                    ))
            if AP.mean() > best_map:
                torch.save(model.state_dict(), f"{opt.job_dir}/best_ckpt.pth")
                logger.info(
                    'Best_map {:.4f}\t'.format(
                        AP.mean()
                    ))
                best_map = AP.mean()

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        torch.save(model.state_dict(), f"{opt.job_dir}/yolov3_ckpt_last.pth")

    logger.info(
                    'Best_map {:.4f}\t'.format(
                        best_map
                    ))

