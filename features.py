import numpy as np
from skimage.util import img_as_float
import torch
import torch.nn.functional as F
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from sklearn.decomposition import PCA

from deep import *


if torch.backends.mps.is_available():
    print('USING MPS')
    device = torch.device('mps')
elif torch.cuda.is_available():
    print('USING CUDA')
    device = torch.device('cuda')
else:
    print('USING CPU')
    device = torch.device('cpu')


def color_features(img):
    H, W, C = img.shape
    img = img_as_float(img)
    return img.reshape(H*W, C)


def color_pos_features(img):
    H, W, C = img.shape
    img = img_as_float(img)

    color = img.reshape(H*W, C)
    color = (color - color.mean(axis=0)) / color.std(axis=0)
    xs = np.arange(W)
    xs = (xs - xs.mean()) / xs.std()
    xs = np.tile(xs.reshape(1, len(xs)), (H, 1))
    ys = np.arange(H)
    ys = (ys - ys.mean()) / ys.std()
    ys = np.tile(ys.reshape(len(ys), 1), (1, W))

    xs = xs.reshape(H*W, 1)
    ys = ys.reshape(H*W, 1)

    features = np.concatenate([color, xs, ys], axis=1)
    return features


def mean_pool(img):
    color_pos = color_pos_features(img)
    color_pos = torch.from_numpy(color_pos)
    color_pos = color_pos.reshape(1, *img.shape[:2], 5).permute(0, -1, 1, 2)  # B, C, H, W
    mean_pooled = F.avg_pool2d(
        input=color_pos,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    mean_pooled = mean_pooled.squeeze()  # C, H, W
    mean_pooled = mean_pooled.permute(1, 2, 0)  # H, W, C
    mean_pooled = mean_pooled.numpy().reshape(img.shape[0] * img.shape[1], 5)
    return mean_pooled


def mean_pool_color_pos(img):
    mean_pooled = mean_pool(img)
    color_pos = color_pos_features(img)
    return np.concatenate([color_pos, mean_pooled], axis=1)


def filters_33(img):
    filters = torch.tensor([
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],  # Gradient East
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],  # Gradient North
        [[0, -1, -2], [1, 0, -1], [2, 1, 0]],  # Gradient North-East
        [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],  # Gradient North-West
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],  # Gradient South
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],  # Gradient West
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],  # Laplacian 3x3
        [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]],  # Line Detection horizontal
        [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]],  # Line Detection left diagonal
        [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]],  # Line Detection right diagonal
        [[-1, 0, -1], [-1, 2, -1], [-1, 2, -1]],  # Line Detection vertical
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],  # Sobel horizontal
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],  # Sobel vertical
        [[0, -0.25, 0], [-0.25, 2, -0.25], [0, -0.25, 0]],  # Sharpen
        [[-0.25, -0.25, -0.25], [-0.25, 3, -0.25], [-0.25, -0.25, -0.25]],  # Sharpen II
        [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]],  # Sharpen
        [[0.111, 0.111, 0.111], [0.111, 0.111, 0.111], [0.111, 0.111, 0.111]],  # Smooth arithmetic mean
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]],  # Smoothing 3x3
        [[-0.627, 0.352, -0.627], [0.352, 2.923, 0.352], [-0.627, 0.352, -0.627]],  # Point spread
    ]).unsqueeze(1).tile(1, 3, 1, 1)
    color = color_features(img)
    color = torch.from_numpy(color)
    color = color.reshape(1, *img.shape[:2], 3).permute(0, -1, 1, 2)  # B, C, H, W
    convolved = F.conv2d(
        input=color.float(),
        weight=filters.float(),
        stride=1,
        padding=1,
    )
    convolved = convolved.squeeze()  # C, H, W
    convolved = convolved.permute(1, 2, 0)  # H, W, C
    convolved = convolved.numpy().reshape(img.shape[0] * img.shape[1], -1)
    return convolved


def filters_33_color_pos(img):
    convolved = filters_33(img)
    color_pos = color_pos_features(img)
    return np.concatenate([color_pos, convolved], axis=1)


def deep_pretrained(img):
    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights, num_classes=21)
    model.eval()

    preprocess = weights.transforms()
    batch = preprocess(torch.from_numpy(img).permute(2, 0, 1)).unsqueeze(0)  # C, H, W
    prediction = model.backbone(batch)["out"]
    out = F.interpolate(prediction, size=img.shape[:2], mode="bilinear", align_corners=False)
    # out = torch.to_numpy(out.squeeze().permute(1, 2, 0).reshape(img.shape[0] * img.shape[1], -1))  # H, W, C; H * W, C
    out = out.squeeze().permute(1, 2, 0).reshape(img.shape[0] * img.shape[1], -1).detach().numpy()  # H, W, C; H * W, C
    pca = PCA(n_components=64)
    out = pca.fit_transform(out)
    return out


def deep_contrastive(img):
    model = BigModelUpsample()
    model.load_state_dict(torch.load('./contrastive_save_old_dataset/contrastive_weights_best.pt', map_location=torch.device('cpu')))
    model = model.to(device)

    img = img_as_float(img)
    img = torch.from_numpy(img).permute(2, 0, 1)  # C, H, W
    img = img.float().to(device)
    embeddings = model(img.unsqueeze(0))
    embeddings = embeddings.squeeze(0).flatten(1, -1).permute(1, 0).cpu().detach().numpy()
    return embeddings