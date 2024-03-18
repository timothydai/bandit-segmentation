import pickle
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    vit_b_16,
    ViT_B_16_Weights,
    vgg11,
    VGG11_Weights,
    inception_v3,
    Inception_V3_Weights,
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights
)


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


if __name__ == '__main__':
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device)
    model.eval()
    preprocess = weights.transforms()

    with open('dataset_new.pkl', 'rb') as f:
        dataset = pickle.load(f)
    dataset = dataset[:5000]

    embeddings = []
    for i in tqdm(range(len(dataset))):
        img_path, img, gt_mask = dataset[i]
        np.save(f'embedding_{i}.npy', embed_image(img, preprocess, model))
        