import albumentations as A
from albumentations.pytorch import ToTensorV2

def transforms(img_size):
    return A.Compose([
                      A.Resize(img_size, img_size),
                      A.VerticalFlip(p=0.5),
                      A.HorizontalFlip(p=0.5),
                      A.Normalize(p=1),
                      ToTensorV2(p=1)
                    ])