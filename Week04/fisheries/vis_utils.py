import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from tqdm import tqdm
from PIL import Image

import torch
from torchvision import transforms
from torch.autograd import Variable

from utils import to_var


totensor = transforms.Compose([transforms.ToTensor()])
topilimg = transforms.Compose([transforms.ToPILImage()])
cuda = torch.cuda.is_available()


fish_class = ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]

class_id = {k: v for v, k in enumerate(fish_class)}
id_class = {k: v for k, v in enumerate(fish_class)}


def imshow(inp, title=None):
    """Imshow for Tensor.
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.axis('off')
    if title is not None:
        plt.title(title)


def find_cls(x):
    x = x.data.max(0)[1]
    return x.sum()


def plot_bbox(img, bbox, w, h, color='red'):
    """ Plot bounding box on the image tensor. 
    """
    img = img.cpu().numpy().transpose((1, 2, 0))  # (H, W, C)
    
    # denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    # bounding box
    bb = np.array(bbox, dtype=np.float32)
    bx, by = bb[0] * w, bb[1] * h
    bw, bh = bb[2] * w, bb[3] * h
        
    # scale image
    img = cv2.resize(img, (w, h))
    
    # create BB rectangle
    rect = plt.Rectangle((bx, by), bw, bh, color=color, fill=False, lw=3)
    
    # plot
    plt.figure(figsize=(12, 8))
    plt.axis('off')
    plt.imshow(img)
    plt.gca().add_patch(rect)
    plt.show()


def get_bbox_corners(loc, im_w, im_h):
    bx, by, bw, bh = loc[:]
    
    bx, bw = bx * im_w, bw * im_w
    by, bh = by * im_h, bh * im_h
       
    c0 = int(max(bx, 0))
    r0 = int(max(by, 0))
    c1 = int(min(bx + bw, im_w) - 1)
    r1 = int(min(by + bh, im_h) - 1)
    
    return c0, r0, c1, r1


def create_bbox(image_path, image_size, model, dpi=120):
    model.eval()

    image = Image.open(image_path)
    im_w, im_h = image.size

    # unsqueeze for make batch_size = 1
    input = totensor(image.resize([image_size, image_size])).unsqueeze(0)
    output_loc, output_cls = model(to_var(input))

    if cuda:
        output_loc = output_loc.cpu()
        output_cls = output_cls.cpu()
    
    # squeeze because batch is 1
    output_cls.data.squeeze_(0)
    output_loc.data.squeeze_(0)
    
    # load image for matplotlib
    image = mpimg.imread(image_path)
    fig = Figure(figsize=(im_w/dpi, im_h/dpi), dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.imshow(image)

    c0, r0, c1, r1 = get_bbox_corners(output_loc.data, im_w, im_h)
    ax.plot([c0, c1, c1, c0, c0], [r0, r0, r1, r1, r0])
    ax.text(c0, r0, f"{id_class[find_cls(output_cls)]}", bbox={'alpha': 0.5, 'pad': 0.2})
    ax.axis('off')

    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    # numpy array -> PIL data -> tensor
    tensor = totensor(Image.fromarray(image))
    # image is numpy array and tensor is tensor
    return image, tensor


class PlotLoss:
    def __init__(self, vis, win=None, opts=None):
        self.vis = vis
        self.win = win
        self.opts = opts
        if self.win is None:
            self.win = self.opts.get("title")
        self.__iteration = 0
        self.container = [[], []]

    def append(self, data):
        self.__iteration += 1
        if isinstance(data, list):
            assert len(data) <= 2, "data length should be 1 or 2"
            self.container[0].append(data[0])
            self.container[1].append(data[1])
            X = np.array([range(self.__iteration), range(self.__iteration)]).T
            Y = np.array(self.container).T
        else:
            self.container[0].append(data)
            X = np.array([range(self.__iteration)])
            Y = np.array(self.container[0])

        self.vis.line(X=X, Y=Y, win=self.win, opts=self.opts)


class ShowSample:
    def __init__(self, model, vis, image_path, image_size, win=None, opts=None):
        self.model = model
        assert os.path.exists(image_path)
        self.image_path = image_path
        self.image_size = image_size
        self.vis = vis
        self.win = win
        self.opts = opts

    def show(self):
        _, tensor = create_bbox(self.image_path, self.image_size, self.model)
        self.vis.image(img=tensor, win=self.win, opts=self.opts)