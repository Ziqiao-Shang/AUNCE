from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

class PlaceCrop(object): 
    def __init__(self, size, start_x, start_y):
        if isinstance(size, int): 
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))

class SetFlip(object): 
    def __init__(self, flip):
        self.flip = flip

    def __call__(self, img): 
        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class image_train(object): 
    def __init__(self, crop_size=176):
        self.crop_size = crop_size

    def __call__(self, img, flip, offset_x, offset_y):
        transform = transforms.Compose([ 
            PlaceCrop(self.crop_size, offset_x, offset_y),
            SetFlip(flip),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        img = transform(img)
        return img
    
class image_train_enhance(object):
    def __init__(self, crop_size=176):
        self.crop_size = crop_size

    def __call__(self, img, flip, offset_x, offset_y):
        transform = transforms.Compose([
            PlaceCrop(self.crop_size, offset_x, offset_y), 
            SetFlip(flip),                              
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=int(0.1 * 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        img = transform(img)
        return img

class image_test(object): 
    def __init__(self, crop_size=176):
        self.crop_size = crop_size

    def __call__(self, img):
        transform = transforms.Compose([ 
            transforms.CenterCrop(self.crop_size), 
            transforms.ToTensor(),                 
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        img = transform(img) 
        return img       

class GaussianBlur(object):
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample

class land_transform(object): 
    def __init__(self, img_size, flip_reflect):
        self.img_size = img_size
        self.flip_reflect = flip_reflect.astype(int) - 1 

    def __call__(self, land, flip, offset_x, offset_y):
        land[0:len(land):2] = land[0:len(land):2] - offset_x
        land[1:len(land):2] = land[1:len(land):2] - offset_y
        if flip:
            land[0:len(land):2] = self.img_size - 1 - land[0:len(land):2]
            land[0:len(land):2] = land[0:len(land):2][self.flip_reflect] 
            land[1:len(land):2] = land[1:len(land):2][self.flip_reflect]
        return land