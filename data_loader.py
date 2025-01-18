import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os


def make_dataset(image_list, label_list):
    len_ = len(image_list)
    images = [(image_list[i].strip(), label_list[i, :]) for i in range(len_)]
    return images

def make_dataset_landmark(image_list, label_list, landmark_list):
    len_ = len(image_list)
    images = [(image_list[i].strip(), label_list[i, :], landmark_list[i, :]) for i in range(len_)]
    return images

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)

class BP4D(Dataset):
    def __init__(self, root_path, train = True, if_200 = True, info_nce = 'enhance', fold = 1, transform = None, target_transform = None, crop_size = 176, loader=default_loader):

        assert fold > 0 and fold <= 3, 'The fold num must be restricted from 1 to 3'
        self._root_path = root_path
        self._train = train
        self._if_200 = if_200
        self._info_nce = info_nce
        self._transform = transform
        self._target_transform = target_transform
        self._crop_size = crop_size
        self._loader = loader
        
        if self._train:
            if self._if_200:
                self._img_folder_path = os.path.join(root_path,'img_200x200')
                train_image_list_path = os.path.join(root_path, 'list_200x200', 'BP4D_train_img_path_fold' + str(fold) +'.txt')
                train_image_list = open(train_image_list_path).readlines() 
                train_label_list_path = os.path.join(root_path, 'list_200x200', 'BP4D_train_label_fold' + str(fold) + '.txt')
                train_label_list = np.loadtxt(train_label_list_path)

            else:
                self._img_folder_path = os.path.join(root_path,'img_256x256')
                train_image_list_path = os.path.join(root_path, 'list_256x256', 'BP4D_train_img_path_fold' + str(fold) +'.txt')
                train_image_list = open(train_image_list_path).readlines() 
                train_label_list_path = os.path.join(root_path, 'list_256x256', 'BP4D_train_label_fold' + str(fold) + '.txt')
                train_label_list = np.loadtxt(train_label_list_path)
                
            self._data_list = make_dataset(train_image_list, train_label_list)

        else:
            if self._if_200:
                self._img_folder_path = os.path.join(root_path,'img_200x200')
                test_image_list_path = os.path.join(root_path, 'list_200x200', 'BP4D_test_img_path_fold' + str(fold) +'.txt')
                test_image_list = open(test_image_list_path).readlines() 
                test_label_list_path = os.path.join(root_path, 'list_200x200', 'BP4D_test_label_fold' + str(fold) + '.txt')
                test_label_list = np.loadtxt(test_label_list_path)

            else:
                self._img_folder_path = os.path.join(root_path,'img_256x256')
                test_image_list_path = os.path.join(root_path, 'list_256x256', 'BP4D_test_img_path_fold' + str(fold) +'.txt')
                test_image_list = open(test_image_list_path).readlines() 
                test_label_list_path = os.path.join(root_path, 'list_256x256', 'BP4D_test_label_fold' + str(fold) + '.txt')
                test_label_list = np.loadtxt(test_label_list_path)

            self._data_list = make_dataset(test_image_list, test_label_list)

    def __getitem__(self, index):
        if self._train:
            img, label = self._data_list[index]
            img = self._loader(os.path.join(self._img_folder_path, img))

            if self._info_nce == 'enhance':
                img_target = img
                w, h = img.size
                offset_y = random.randint(0, h - self._crop_size)
                offset_x = random.randint(0, w - self._crop_size)

                offset_target_y = random.randint(0, h - self._crop_size)
                offset_target_x = random.randint(0, w - self._crop_size)

                flip = random.randint(0, 1)
                flip_target = random.randint(0, 1)
                
                if self._transform and self._target_transform is not None:
                    img = self._transform(img, flip, offset_x, offset_y)
                    img_target = self._target_transform(img_target, flip_target, offset_target_x, offset_target_y)                      

                return img, img_target, label
            
            else:
                w, h = img.size
                offset_y = random.randint(0, h - self._crop_size) 
                offset_x = random.randint(0, w - self._crop_size)
                flip = random.randint(0, 1)
                if self._transform is not None:
                    img = self._transform(img, flip, offset_x, offset_y)
                return img, label
            
        else:
            img, label = self._data_list[index]
            img = self._loader(os.path.join(self._img_folder_path, img))
            if self._transform is not None:
                img = self._transform(img)
            return img, label

    def __len__(self):
        return len(self._data_list)

class DISFA(Dataset):
    def __init__(self, root_path, train = True, if_200 = True, info_nce = 'enhance', fold = 1, transform = None, 
                 target_transform = None, crop_size = 176, loader=default_loader):

        assert fold > 0 and fold <= 3, 'The fold num must be restricted from 1 to 3'
        self._root_path = root_path
        self._train = train
        self._if_200 = if_200
        self._info_nce = info_nce
        self._transform = transform
        self._target_transform = target_transform
        self._crop_size = crop_size
        self._loader = loader
        
        if self._train:
            if self._if_200:
                self._img_folder_path = os.path.join(root_path,'img_200x200')
                train_image_list_path = os.path.join(root_path, 'list_200x200', 'DISFA_train_img_path_fold' + str(fold) +'.txt')
                train_image_list = open(train_image_list_path).readlines() 
                train_label_list_path = os.path.join(root_path, 'list_200x200', 'DISFA_train_label_fold' + str(fold) + '.txt')
                train_label_list = np.loadtxt(train_label_list_path)

            else:
                self._img_folder_path = os.path.join(root_path,'img_256x256')
                train_image_list_path = os.path.join(root_path, 'list_256x256', 'DISFA_train_img_path_fold' + str(fold) +'.txt')
                train_image_list = open(train_image_list_path).readlines() 
                train_label_list_path = os.path.join(root_path, 'list_256x256', 'DISFA_train_label_fold' + str(fold) + '.txt')
                train_label_list = np.loadtxt(train_label_list_path)
                
            self._data_list = make_dataset(train_image_list, train_label_list)

        else:
            if self._if_200:
                self._img_folder_path = os.path.join(root_path,'img_200x200')
                test_image_list_path = os.path.join(root_path, 'list_200x200', 'DISFA_test_img_path_fold' + str(fold) +'.txt')
                test_image_list = open(test_image_list_path).readlines() 
                test_label_list_path = os.path.join(root_path, 'list_200x200', 'DISFA_test_label_fold' + str(fold) + '.txt')
                test_label_list = np.loadtxt(test_label_list_path)

            else:
                self._img_folder_path = os.path.join(root_path,'img_256x256')
                test_image_list_path = os.path.join(root_path, 'list_256x256', 'DISFA_test_img_path_fold' + str(fold) +'.txt')
                test_image_list = open(test_image_list_path).readlines() 
                test_label_list_path = os.path.join(root_path, 'list_256x256', 'DISFA_test_label_fold' + str(fold) + '.txt')
                test_label_list = np.loadtxt(test_label_list_path)

            self._data_list = make_dataset(test_image_list, test_label_list)

    def __getitem__(self, index):
        if self._train:
            img, label = self._data_list[index]
            img = self._loader(os.path.join(self._img_folder_path, img))

            if self._info_nce == 'enhance':
                img_target = img
                w, h = img.size
                offset_y = random.randint(0, h - self._crop_size)
                offset_x = random.randint(0, w - self._crop_size)

                offset_target_y = random.randint(0, h - self._crop_size)
                offset_target_x = random.randint(0, w - self._crop_size)

                flip = random.randint(0, 1)
                flip_target = random.randint(0, 1)
                
                if self._transform and self._target_transform is not None:
                    img = self._transform(img, flip, offset_x, offset_y)
                    img_target = self._target_transform(img_target, flip_target, offset_target_x, offset_target_y)                      

                return img, img_target, label
            
            else:
                w, h = img.size
                offset_y = random.randint(0, h - self._crop_size) 
                offset_x = random.randint(0, w - self._crop_size)
                flip = random.randint(0, 1)
                if self._transform is not None:
                    img = self._transform(img, flip, offset_x, offset_y)
                return img, label
            
        else:
            img, label = self._data_list[index]
            img = self._loader(os.path.join(self._img_folder_path, img))
            if self._transform is not None:
                img = self._transform(img)
            return img, label

    def __len__(self):
        return len(self._data_list)

class BP4D_Landmark(Dataset):
    def __init__(self, root_path, train = True, info_nce = 'enhance', fold = 1, transform = None, target_transform = None, 
                 landmark_transform = None, landmark_target_transform = None, crop_size = 176, loader=default_loader):

        assert fold > 0 and fold <= 3, 'The fold num must be restricted from 1 to 3'
        self._root_path = root_path
        self._train = train
        self._info_nce = info_nce
        self._transform = transform
        self._target_transform = target_transform
        self._landmark_transform = landmark_transform
        self._landmark_target_transform = landmark_target_transform
        self._crop_size = crop_size
        self._loader = loader
        
        if self._train:
            self._img_folder_path = os.path.join(root_path,'img_200x200')
            train_image_list_path = os.path.join(root_path, 'list_200x200', 'BP4D_train_img_path_fold' + str(fold) +'.txt')
            train_image_list = open(train_image_list_path).readlines() 
            train_label_list_path = os.path.join(root_path, 'list_200x200', 'BP4D_train_label_fold' + str(fold) + '.txt')
            train_label_list = np.loadtxt(train_label_list_path)
            train_landmark_list_path = os.path.join(root_path, 'list_200x200', 'BP4D_train_landmark_fold' + str(fold) + '.txt')
            train_landmark_list = np.loadtxt(train_landmark_list_path)
            self._data_list = make_dataset_landmark(train_image_list, train_label_list, train_landmark_list)

        else:
            self._img_folder_path = os.path.join(root_path,'img_200x200')
            test_image_list_path = os.path.join(root_path, 'list_200x200', 'BP4D_test_img_path_fold' + str(fold) +'.txt')
            test_image_list = open(test_image_list_path).readlines() 
            test_label_list_path = os.path.join(root_path, 'list_200x200', 'BP4D_test_label_fold' + str(fold) + '.txt')
            test_label_list = np.loadtxt(test_label_list_path)
            test_landmark_list_path = os.path.join(root_path, 'list_200x200', 'BP4D_test_landmark_fold' + str(fold) + '.txt')
            test_landmark_list = np.loadtxt(test_landmark_list_path)
            self._data_list = make_dataset_landmark(test_image_list, test_label_list, test_landmark_list)

    def __getitem__(self, index):
        img, label, landmark = self._data_list[index]
        img = self._loader(os.path.join(self._img_folder_path, img))
        if self._train:
            if self._info_nce == 'enhance':
                img_target = img
                landmark_target = landmark
                w, h = img.size
                offset_y = random.randint(0, h - self._crop_size)
                offset_x = random.randint(0, w - self._crop_size)

                offset_target_y = random.randint(0, h - self._crop_size)
                offset_target_x = random.randint(0, w - self._crop_size)

                flip = random.randint(0, 1)
                flip_target = random.randint(0, 1)
                
                if self._transform and self._target_transform is not None:
                    img = self._transform(img, flip, offset_x, offset_y)
                    img_target = self._target_transform(img_target, flip_target, offset_target_x, offset_target_y)         

                if self._landmark_transform and self._landmark_target_transform is not None:
                    landmark = self._landmark_transform(landmark, flip, offset_x, offset_y)       
                    landmark_target = self._landmark_target_transform(landmark_target, flip_target, offset_target_x, offset_target_y)      

                return img, img_target, landmark, landmark_target, label
            
            else:
                w, h = img.size
                offset_y = random.randint(0, h - self._crop_size) 
                offset_x = random.randint(0, w - self._crop_size)
                flip = random.randint(0, 1)
                if self._transform is not None:
                    img = self._transform(img, flip, offset_x, offset_y)
                if self._landmark_transform is not None:
                    landmark = self._landmark_transform(landmark, flip, offset_x, offset_y)
                return img, landmark, label
            
        else:
            w, h = img.size
            offset_y = (h - self._crop_size)/2
            offset_x = (w - self._crop_size)/2
            if self._transform is not None:
                img = self._transform(img)
            if self._landmark_transform is not None:
                landmark = self._landmark_transform(landmark, 0, offset_x, offset_y)
            return img, landmark, label

    def __len__(self):
        return len(self._data_list)

class DISFA_Landmark(Dataset):
    def __init__(self, root_path, train = True, info_nce = 'enhance', fold = 1, transform = None, target_transform = None, 
                 landmark_transform = None, landmark_target_transform = None, crop_size = 176, loader=default_loader):

        assert fold > 0 and fold <= 3, 'The fold num must be restricted from 1 to 3'
        self._root_path = root_path
        self._train = train
        self._info_nce = info_nce
        self._transform = transform
        self._target_transform = target_transform
        self._landmark_transform = landmark_transform
        self._landmark_target_transform = landmark_target_transform
        self._crop_size = crop_size
        self._loader = loader
        
        if self._train:
            self._img_folder_path = os.path.join(root_path,'img_200x200')
            train_image_list_path = os.path.join(root_path, 'list_200x200', 'DISFA_train_img_path_fold' + str(fold) +'.txt')
            train_image_list = open(train_image_list_path).readlines() 
            train_label_list_path = os.path.join(root_path, 'list_200x200', 'DISFA_train_label_fold' + str(fold) + '.txt')
            train_label_list = np.loadtxt(train_label_list_path)
            train_landmark_list_path = os.path.join(root_path, 'list_200x200', 'DISFA_train_landmark_fold' + str(fold) + '.txt')
            train_landmark_list = np.loadtxt(train_landmark_list_path)
            self._data_list = make_dataset_landmark(train_image_list, train_label_list, train_landmark_list)

        else:
            self._img_folder_path = os.path.join(root_path,'img_200x200')
            test_image_list_path = os.path.join(root_path, 'list_200x200', 'DISFA_test_img_path_fold' + str(fold) +'.txt')
            test_image_list = open(test_image_list_path).readlines() 
            test_label_list_path = os.path.join(root_path, 'list_200x200', 'DISFA_test_label_fold' + str(fold) + '.txt')
            test_label_list = np.loadtxt(test_label_list_path)
            test_landmark_list_path = os.path.join(root_path, 'list_200x200', 'DISFA_test_landmark_fold' + str(fold) + '.txt')
            test_landmark_list = np.loadtxt(test_landmark_list_path)
            self._data_list = make_dataset_landmark(test_image_list, test_label_list, test_landmark_list)

    def __getitem__(self, index):
        img, label, landmark = self._data_list[index]
        img = self._loader(os.path.join(self._img_folder_path, img))
        if self._train:
            if self._info_nce == 'enhance':
                img_target = img
                landmark_target = landmark
                w, h = img.size
                offset_y = random.randint(0, h - self._crop_size)
                offset_x = random.randint(0, w - self._crop_size)

                offset_target_y = random.randint(0, h - self._crop_size)
                offset_target_x = random.randint(0, w - self._crop_size)

                flip = random.randint(0, 1)
                flip_target = random.randint(0, 1)
                
                if self._transform and self._target_transform is not None:
                    img = self._transform(img, flip, offset_x, offset_y)
                    img_target = self._target_transform(img_target, flip_target, offset_target_x, offset_target_y)         

                if self._landmark_transform and self._landmark_target_transform is not None:
                    landmark = self._landmark_transform(landmark, flip, offset_x, offset_y)       
                    landmark_target = self._landmark_target_transform(landmark_target, flip_target, offset_target_x, offset_target_y)      

                return img, img_target, landmark, landmark_target, label
            
            else:
                w, h = img.size
                offset_y = random.randint(0, h - self._crop_size) 
                offset_x = random.randint(0, w - self._crop_size)
                flip = random.randint(0, 1)
                if self._transform is not None:
                    img = self._transform(img, flip, offset_x, offset_y)
                if self._landmark_transform is not None:
                    landmark = self._landmark_transform(landmark, flip, offset_x, offset_y)
                return img, landmark, label
            
        else:
            w, h = img.size
            offset_y = (h - self._crop_size)/2
            offset_x = (w - self._crop_size)/2
            if self._transform is not None:
                img = self._transform(img)
            if self._landmark_transform is not None:
                landmark = self._landmark_transform(landmark, 0, offset_x, offset_y)
            return img, landmark, label

    def __len__(self):
        return len(self._data_list)
