import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display, HTML 
from torch.autograd import Variable
import arabic_reshaper as ar
from bidi.algorithm import get_display


plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['B Nazanin', 'Tahoma']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.unicode_minus'] = False


coco_imgs_dir = "D:/datasets/image_captioning/coco_persian/images/"
flickr_imgs_dir = "D:/datasets/image_captioning/flicker/images/"


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def load_json(filename='data/fa_images_captions_train.json'):
    with open(filename, 'r') as f:
        annotations = json.load(f, encoding='utf8')
    return annotations
    

def to_persion_text(text):
    return get_display(ar.reshape(text))


def show_image(image_file, caption):
    img = plt.imread(image_file)
    plt.imshow(img)
    plt.axis('off')
    plt.title(to_persion_text(caption), size='large')

    
def show_persian_image_and_caption(caption, image, number=None):
    result = ''
    if number:
        if number % 5 == 0:
            result += '<br><br><img align=center src=%s>' % image
        result += '<font face="B Nazanin" dir="rtl" color="#0000FF" size="4">%d: %s</font>' % (number+1, caption)
    else:
        result += '<br><img align=center src=%s>' % image
        result += '<font face="B Nazanin" dir="rtl" color="#FF000FF" size="4">%s</font>' % (caption, )
    
    return HTML(result)
    

def show_persian_captions(captions):
    result = ''
    for i, cap in enumerate(captions):
        result += '<p align="right"><font face="B Nazanin" dir="rtl" color="#0000FF" size="4">%d: %s</font></p>' % (i+1, cap)
    return HTML(result)
    

def show_random_image_with_caption(coco):
    # Choose a random caption
    N = len(coco['annotations'])
    idx = np.random.choice(range(N))
    
    # get the caption and the coresponding image
    item = coco['annotations'][idx]
    image_filename = item['file_name']

    if image_filename.startswith('COCO'):
        images_dir = coco_imgs_dir
    else:
        images_dir = flickr_imgs_dir
    
    caption = item['caption']
    image_file = os.path.join(images_dir, image_filename)
    
    # plot the image with its caption
    show_image(image_file, caption)
