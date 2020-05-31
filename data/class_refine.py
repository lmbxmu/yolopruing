import os

Dir = './coco_class_6/Annotations/val2014'
ImageDir = './coco_class_6/images/val2014'
cnt = 0
for i, file_name in enumerate(os.listdir(Dir)):
    fsize = os.path.getsize(os.path.join(Dir,file_name))
    if fsize == 410:
        print('removing {} of size{}'.format(file_name,fsize))
        os.remove(os.path.join(ImageDir, file_name[:-3]+'jpg'))
        os.remove(os.path.join(Dir, file_name))
        cnt += 1 

print('remove {} files'.format(cnt))

