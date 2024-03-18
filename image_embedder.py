import pickle

import torch
import torch.nn.functional as F
from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image


if torch.backends.mps.is_available():
    print('USING MPS')
    device = torch.device('mps')
elif torch.cuda.is_available():
    print('USING CUDA')
    device = torch.device('cuda')
else:
    print('USING CPU')
    device = torch.device('cpu')


def embed_image(img, preprocess, model):
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    batch = preprocess(img).unsqueeze(0)
    prediction = model.to(device)(batch.to(device))
    return prediction.squeeze(0).detach().cpu().numpy()