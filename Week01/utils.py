import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as T
import bcolz


def save_array(fname, arr): 
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

    
def load_array(fname):
    return bcolz.open(fname)[:]


# def to_var(x, volatile=False):
#     if torch.cuda.is_available():
#         x = x.cuda()
#     return Variable(x, volatile=volatile)


def create_img_dataloader(image_folder, transform=None, batch_size=25, shuffle=False, num_workers=2):
    if transform is None:
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    img_dataset = datasets.ImageFolder(image_folder, transform)
    img_dataloader = torch.utils.data.DataLoader(img_dataset, batch_size, shuffle, num_workers)
    return img_dataset, img_dataloader