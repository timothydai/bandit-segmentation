import argparse
import glob
import os
import pickle
import time
from tqdm import tqdm

import numpy as np
import pandas as pd
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

from algos import *
from features import *
from image_embedder import *
from viz import *


def main(args):
    with open('dataset_new.pkl', 'rb') as f:
        dataset = pickle.load(f)
    dataset = dataset[:5000]
    results_dir = f'results/{args.exp_name}'
    os.makedirs(results_dir, exist_ok=True)

    if args.img_embedder == 'resnet':
        weights = ResNet50_Weights.DEFAULT
        model = resnet50
    elif args.img_embedder == 'vit':
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16
    elif args.img_embedder == 'vgg':
        weights = VGG11_Weights.DEFAULT
        model = vgg11
    elif args.img_embedder == 'inception':
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3
    elif args.img_embedder == 'efficientnet':
        weights = EfficientNet_V2_S_Weights.DEFAULT
        model = efficientnet_v2_s

    model = model(weights=weights).to(device)
    model.eval()
    preprocess = weights.transforms()

    parser.add_argument('--image_embedder', required=True, choices=['resnet', 'vgg', 'vit', 'inception', 'efficientnet'])

    alpha = 0.1
    arms = {
        'color': color_features,
        'color_pos': color_pos_features,
        'mean_pool': mean_pool,
        'filters': filters_33, 
        'deep': deep_contrastive,
    }
    A_arms = {arm: np.identity(1000) for arm in arms}
    b_arms = {arm: np.zeros(1000) for arm in arms}

    pd.DataFrame({
        't': [],
        'img_index': [],
        'img_path': [],
        'arm': [],
        'reward': [],
        'color_acc': [],
        'color_pos_acc': [],
        'mean_pool_acc': [],
        'filters_acc': [],
        'deep_acc': [],
    }).to_csv(os.path.join(results_dir, 'results.csv'), index=False)

    image_indices = list(range(len(dataset)))
    if args.shuffle:
        np.random.shuffle(image_indices)
    for t in tqdm(range(5000)):
        image_index = image_indices[t]
        img_path, img, gt_mask = dataset[image_index]

        feature_fn_accs = {
            arm: evaluate_segmentation(
                gt_mask, compute_segmentation(
                    img=img, k=3, clustering_fn=kmeans_fast, feature_fn=feature_fn, scale=0.5
                )
            )
            for arm, feature_fn in arms.items()
        }

        # if args.algo == 'rl_context':
        # img_feature = np.load(f'embeddings/embedding_{image_index}.npy')  # embed_image(img, preprocess, model)
        # probs = []
        # for arm in arms:
        #     inv_A = np.linalg.inv(A_arms[arm])
        #     theta = inv_A @ b_arms[arm]
        #     p = theta.T @ img_feature + alpha * np.sqrt(img_feature.T @ inv_A @ img_feature)
        #     probs.append(p)
        # pulled_arm = np.array(probs).argmax()
        # pulled_arm_name = list(arms.keys())[pulled_arm]
        
        # # Observe reward.
        # reward = feature_fn_accs[pulled_arm_name]
        
        # # Update weights.
        # A_arms[pulled_arm_name] = A_arms[pulled_arm_name] + np.outer(img_feature, img_feature)
        # b_arms[pulled_arm_name] = b_arms[pulled_arm_name] + reward * img_feature

        pd.DataFrame({
            't': [t],
            'img_index': [image_index],
            'img_path': [img_path],
            'arm': [None],  # pulled_arm_name],
            'reward': [None],  # reward],
            'color_acc': [feature_fn_accs['color']],
            'color_pos_acc': [feature_fn_accs['color_pos']],
            'mean_pool': [feature_fn_accs['mean_pool']],
            'filters': [feature_fn_accs['filters']],
            'deep': [feature_fn_accs['deep']],
        }).to_csv(os.path.join(results_dir, 'results.csv'), index=False, mode='a', header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('--img_embedder', required=True, choices=['resnet', 'vgg', 'vit', 'inception', 'efficientnet'])
    parser.add_argument('--shuffle', default=False)
    # parser.add_argument('--algo', required=True, choices=['naive', 'best', 'majority', 'rl_context'])
    args = parser.parse_args()
    main(args)
