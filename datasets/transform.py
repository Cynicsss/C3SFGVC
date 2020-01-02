from PIL import Image
import random


class resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, im_lb):
        img = im_lb['im']
        lb = im_lb['lb']
        return dict(im = img.resize(self.size, Image.BILINEAR),
                    lb = lb)

class HorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            img = im_lb['im']
            lb = im_lb['lb']
            return dict(im=img.transpose(Image.FLIP_LEFT_RIGHT),
                        lb = lb)


class Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for action in self.do_list:
            im_lb = action(im_lb)

        return im_lb