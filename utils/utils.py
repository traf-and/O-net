from pathlib import Path
import tarfile
import typing as tp
import os
import numpy as np

import dlib
import gdown


def read_pts(filename)->tp.Tuple[np.ndarray, int]:
    data = np.loadtxt(filename, comments=("version:", "n_points:", "{", "}"))
    return data, data.shape[0]

def get_face_square(image_paths:tp.List[str], a)->tp.List[tp.Tuple[slice,slice]]:
    detector = dlib.get_frontal_face_detector()
    imgs = dlib.load_rgb_image(image_paths)
    print(imgs.shape)
    dets = detector(imgs, a)
    pts = None
    for d in dets:
        left=d.left()
        right=d.right()
        top=d.top()
        bot=d.bottom()
        pts=(slice(top,bot),slice(left,right))
    return pts

def del_list():
    file_list = np.loadtxt('to_del.txt', dtype=str)
    for i in file_list:
        img_file = Path(i+"jpg")
        if img_file.is_file():
            os.remove(i +'jpg')
            os.remove(i +'pts')

def first_run() -> bool:
    isExist = os.path.exists('/pl_ckpt')
    if not isExist:
        os.makedirs('/pl_ckpt/my_model/')
        os.makedirs('/tb_logs/my_model/train/loss')
        os.makedirs('/tb_logs/my_model/valid/loss')
        return True
    return False

def download_data():
    url = 'https://drive.google.com/uc?id=0B8okgV6zu3CCWlU3b3p4bmJSVUU'
    output = 'data/landmarks_task.tgz'
    gdown.download(url, output, quiet=False)
    file = tarfile.open('/data/landmarks_task.tgz')
    file.extractall('/data')
    file.close()
    os.remove('/data/landmarks_task.tgz')
