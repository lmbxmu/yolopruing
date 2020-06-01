from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw

import xml.etree.ElementTree as ET
import pickle
from os import listdir, getcwd
from os.path import join

import argparse

#if the dir is not exists,make it,else delete it
def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default='/media/coco_class_6', help="save_path")
parser.add_argument("--data_dir", type=str, default='/media/coco', help="data_dir")
args = parser.parse_args()


#the path you want to save your results for coco to voc
savepath = args.save_path 
dataDir = args.data_dir 
mkr(savepath)
datasets_list=['train2014', 'val2014']
classes_names = ["person","bicycle","car","motorbike", "bus", "truck"] 

headstr = """\
<annotation>
    <folder>VOC</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>COCO</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""

tailstr = '''\
</annotation>
'''



def id2name(coco):
    classes=dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']]=cls['name']
    return classes

def write_xml(anno_path,head, objs, tail):
    f = open(anno_path, "w")
    f.write(head)
    for obj in objs:
        f.write(objstr%(obj[0],obj[1],obj[2],obj[3],obj[4]))
    f.write(tail)

def save_annotations_and_imgs(coco,dataset,filename,objs,anno_dir,img_dir):
    #eg:COCO_train2014_000000196610.jpg-->COCO_train2014_000000196610.xml
    anno_path=anno_dir+filename[:-3]+'xml'
    img_path=dataDir+'images/'+dataset+'/'+filename
    # print(img_path)sss
    dst_imgpath=img_dir+filename

    img=cv2.imread(img_path)
    # print(img)
    if (img.shape[2] == 1):
        print(filename + " not a RGB image")
        return

    shutil.copy(img_path, dst_imgpath)

    head=headstr % (filename, img.shape[1], img.shape[0], img.shape[2])
    tail = tailstr
    write_xml(anno_path,head, objs, tail)

def showimg(coco,dataset,img,classes,cls_id,show=True):
    global dataDir
    I=Image.open('%s/%s/%s/%s'%(dataDir,'images',dataset,img['file_name']))
    #Get the annotated information by ID
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    # print(annIds)
    anns = coco.loadAnns(annIds)
    # print(anns)
    # coco.showAnns(anns)
    objs = []
    for ann in anns:
        class_name=classes[ann['category_id']]
        if class_name in classes_names:
            print(class_name)
            if 'bbox' in ann:
                bbox=ann['bbox']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                obj = [class_name, xmin, ymin, xmax, ymax]
                objs.append(obj)
                draw = ImageDraw.Draw(I)
                draw.rectangle([xmin, ymin, xmax, ymax])
    if show:
        plt.figure()
        plt.axis('off')
        plt.imshow(I)
        plt.show()

    return objs
 
def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
 
def convert_annotation(image_id, in_path, out_path):
    in_file = open(in_path+'%s.xml'%(image_id))
    out_file = open(out_path+'/%s.txt'%(image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
 
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        print(cls)
        if cls not in classes_names or int(difficult)==1:
            continue
        cls_id = classes_names.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
 
mkr(savepath+'images/')
mkr(savepath+'Annotations/')

for dataset in datasets_list:
    #./COCO/annotations/instances_train2014.json
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataset)

    img_dir=savepath+'images/'+dataset+'/'
    anno_dir=savepath+'Annotations/'+dataset+'/'
    mkr(img_dir)
    mkr(anno_dir)
    #COCO API for initializing annotated data
    coco = COCO(annFile)
    '''
    When the COCO object is created, the following information will be output:
    loading annotations into memory...
    Done (t=0.81s)
    creating index...
    index created!
    So far, the JSON script has been parsed and the images are associated with the corresponding annotated data.
    '''
    #show all classes in coco
    classes = id2name(coco)
    print(classes)
    #[1, 2, 3, 4, 6, 8]
    classes_ids = coco.getCatIds(catNms=classes_names)
    print(classes_ids)
    # exit()
    for cls in classes_names:
        #Get ID number of this class
        cls_id=coco.getCatIds(catNms=[cls])
        img_ids=coco.getImgIds(catIds=cls_id)
        print(cls,len(img_ids))
        # imgIds=img_ids[0:10]
        for imgId in tqdm(img_ids):
            img = coco.loadImgs(imgId)[0]
            filename = img['file_name']
            # print(filename)
            objs=showimg(coco, dataset, img, classes,classes_ids,show=False)
            print(objs)
            save_annotations_and_imgs(coco, dataset, filename, objs, anno_dir,img_dir)

    cnt = 0

    for i, file_name in enumerate(os.listdir(anno_dir)):
        fsize = os.path.getsize(os.path.join(anno_dir,file_name))
        if fsize == 410:
            print('removing {} of size{}'.format(file_name,fsize))
            os.remove(os.path.join(img_dir, file_name[:-3]+'jpg'))
            os.remove(os.path.join(anno_dir, file_name))
            cnt += 1 

    print('remove {} files'.format(cnt))

    data_path = savepath+'images/'+dataset
    img_names = os.listdir(data_path)
    
    if dataset == 'train2014':
        list_file = open(savepath+'class_train.txt', 'w')
    else:
        list_file = open(savepath+'class_val.txt', 'w')

    in_path = anno_dir
    out_path = savepath+'labels/'+dataset
    for img_name in img_names:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
     
        list_file.write(img_dir+'%s\n'%img_name)
        image_id = img_name[:-4]
        convert_annotation(image_id, in_path, out_path)
     
    list_file.close()




