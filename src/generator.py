import os
import glob
import pandas as pd
import numpy as np


def train_df(path):
    image_dir= glob.glob(path + '/images/*.JPEG')
    image_dir= sorted(image_dir)
    image_dir= [img.replace('\\', '/') for img in image_dir]
    train_df= pd.DataFrame({
        'path': image_dir,
        'label': [os.path.basename(img).split('_')[0] for img in image_dir]
    })
    train_df.to_csv('train.csv', index=False)

def valid_df(path, root):
    df= pd.read_csv(path, sep='\t', header= None)
    df.drop([2, 3, 4, 5], axis= 1, inplace= True)
    df.columns= ['path', 'label']

    df['path']= root + df['path']

    df.to_csv('valid.csv', index=False)

def test_df(path):
    path= glob.glob(path + '/images/*.JPEG')
    path.sort()

    path= [p.replace('\\', '/') for p in path]

    df= pd.DataFrame({
        'path': path
    })

    df.to_csv('test.csv', index=False)
    
if __name__ == '__main__':
   
    train_df('tiny_imagenet/train/*')
    valid_df(path= 'tiny_imagenet/val/val_annotations.txt', root= 'tiny_imagenet/val/images/')
    test_df('tiny_imagenet/test')

