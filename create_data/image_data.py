import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import typing as tp

import dlib
import albumentations as A

class ONetDataset(Dataset):
    """
    """
    def __init__(self, files: list, mode: str):
        super().__init__()
        self.files = sorted(files)
        # train/test
        self.mode = mode
        self.rescale = 256
        self.len_ = len(self.files)
                      
    def __len__(self):
        return self.len_
      
    def load_sample(self, file):
        image = cv2.imread(file)
        return image
  
    def __getitem__(self, index):
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(self.rescale, self.rescale)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        x = self.load_sample(self.files[index]+'jpg')
        square = self.get_face_square(self.files[index]+'jpg')
        target,_ = self.read_pts(self.files[index]+'pts')
        _, scale = self.get_scale_pts(target, square)
        
        if self.mode == 'train':
            aug = A.Compose([
                A.RGBShift(),
                A.RandomContrast(p=0.2),
                A.RandomBrightnessContrast(p=0.2)                   
            ])
            x = aug(image=x[square])['image']

        x = transform(x)
        return x, target, scale, square[1].start, square[0].start

    def read_pts(sef, filename)->tp.Tuple[np.ndarray, int]:
        data = np.loadtxt(filename, comments=("version:", "n_points:", "{", "}"))
        return data, data.shape[0]

    def get_face_square(self, image_paths:str)->tp.Tuple[slice,slice]:
        detector = dlib.get_frontal_face_detector()
        img = dlib.load_rgb_image(image_paths)
        dets = detector(img, 0)
        pts_arr = None
        for d in dets:
            left=d.left()
            right=d.right()
            top=d.top()
            bot=d.bottom()
            pts_arr = [[top, bot], [left, right]]
        if pts_arr is None:
            pts_arr = self.get_empty_square(img.shape)
        else:
            pts_arr = self.check_neg_num(pts_arr)    
        return pts_arr

    def get_empty_square(self, dim_):
        return (slice(0,dim_[0]),slice(0,dim_[1]))
        

    def get_scale_pts(self, orig, square):
        sc_pts = orig.copy()
        scale = [(square[0].stop-square[0].start)/self.rescale, 
                 (square[1].stop-square[1].start)/self.rescale]
        sc_pts[:,0] -= square[1].start
        sc_pts[:,1] -= square[0].start
        return sc_pts/scale, scale

    def check_neg_num(self, pts):
        if pts[0][0] < 0:
            pts[0][0] = 0
        if pts[1][0] < 0:
            pts[1][0] = 0
        return (slice(pts[0][0], pts[0][1]),
                slice(pts[1][0], pts[1][1]))