from torchvision import transforms
from PIL import Image

class PlaceCrop(object): 
#     Crops the given PIL.Image at the particular index. 
#     Args:
#         size (sequence or int): Desired output size of the crop. If size is an
#             int instead of sequence like (w, h), a square crop (size, size) is
#             made.
    def __init__(self, size, start_x, start_y):
        if isinstance(size, int): 
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        # Args:
        #     img (PIL.Image): Image to be cropped.
        # Returns:
        #     PIL.Image: Cropped image.
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))

class SetFlip(object): 
    def __init__(self, flip):
        self.flip = flip

    def __call__(self, img): 
        # Args:
        #     img (PIL.Image): Image to be flipped.
        # Returns:
        #     PIL.Image: Randomly flipped image.
        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class image_train(object): 
    def __init__(self, crop_size=224):
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
    def __init__(self, crop_size=224):
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
    def __init__(self, crop_size=224):
        self.crop_size = crop_size

    def __call__(self, img):
        transform = transforms.Compose([ 
            transforms.CenterCrop(self.crop_size), 
            transforms.ToTensor(),                 
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        img = transform(img) 
        return img       
