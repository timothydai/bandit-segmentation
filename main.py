import argparse
import glob
import os
import pickle
import time

import numpy as np
import pandas as pd

from algos import *
from features import *
from viz import *


def main(args):
    with open('dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    dataset = dataset[:150]
    results_dir = f'results/{args.exp_name}'
    os.makedirs(results_dir, exist_ok=True)

    if args.feature_fn == 'color':
        feature_fn = color_features
        d = 3
    elif args.feature_fn == 'color_pos':
        feature_fn = color_pos_features
        d = 5
    elif args.feature_fn == 'mean_pool':
        feature_fn = mean_pool
        d = 5
    elif args.feature_fn == 'mean_pool_color_pos':
        feature_fn = mean_pool_color_pos
        d = 10
    elif args.feature_fn == 'filters':
        feature_fn = filters_33
        d = 19
    elif args.feature_fn == 'filters_color_pos':
        feature_fn = filters_33_color_pos
        d = 24
    elif args.feature_fn == 'pretrained':
        feature_fn = deep_pretrained
        d = 64
    elif args.feature_fn == 'deep':
        feature_fn = deep_contrastive
        d = 64

    pd.DataFrame({
        'img_index': [],
        'img_path': [],
        'method': [],
        #'pixel_order': [],
        'preds': [],
        'time': [],
    }).to_csv(os.path.join(results_dir, 'results.csv'), index=False)
    for i in range(len(dataset)):
        img_path, img, gt_mask = dataset[i]

        begin = time.time()
        logreg_pred, pixel_order = logistic_reg(
           img=img,
           features=feature_fn(img),
           gt_mask=gt_mask,
           shuffle_pixels=args.shuffle_pixels,
        )
        # save_img_mask_pair(img, logreg_pred, gt_mask, os.path.join(results_dir, f'{i}_logreg_viz.png'))
        pd.DataFrame({
            'img_index': [i],
            'img_path': [img_path],
            'method': ['logreg'],
            'predictions': [str(logreg_pred.tolist())],
            #'pixel_order': [str(pixel_order.tolist())],
            'time': [time.time() - begin]
        }).to_csv(os.path.join(results_dir, 'results.csv'), index=False, mode='a', header=False)
        
        # begin = time.time()
        # sgd_pred, pixel_order = sgd_classifier(
        #     img=img,
        #     features=feature_fn(img),
        #     gt_mask=gt_mask,
        #     shuffle_pixels=args.shuffle_pixels,
        # )
        # # save_img_mask_pair(img, sgd_pred, gt_mask, os.path.join(results_dir, f'{i}_sgd_viz.png'))
        # pd.DataFrame({
        #     'img_index': [i],
        #     'img_path': [img_path],
        #     'method': ['sgd'],
        #     'predictions': [str(sgd_pred.tolist())],
        #     #'pixel_order': [str(pixel_order.tolist())],
        #     'time': [time.time() - begin]
        # }).to_csv(os.path.join(results_dir, 'results.csv'), index=False, mode='a', header=False)

        # begin = time.time()
        # linucb_pred, pixel_order = linucb_lite(
        #     img=img,
        #     features=feature_fn(img),
        #     gt_mask=gt_mask,
        #     d=d,
        #     shuffle_pixels=args.shuffle_pixels,
        # )
        # # save_img_mask_pair(img, linucb_pred, gt_mask, os.path.join(results_dir, f'{i}_linucb_viz.png'))
        # pd.DataFrame({
        #     'img_index': [i],
        #     'img_path': [img_path],
        #     'method': ['linucb'],
        #     'predictions': [str(linucb_pred.tolist())],
        #     #'pixel_order': [str(pixel_order.tolist())],
        #     'time': [time.time() - begin]
        # }).to_csv(os.path.join(results_dir, 'results.csv'), index=False, mode='a', header=False)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True)
    parser.add_argument(
        '--feature_fn',
        choices=['color', 'color_pos', 'mean_pool', 'filters', 'pretrained', 'deep'],
        required=True
    )
    parser.add_argument('--shuffle_pixels', type=bool, default=False)
    args = parser.parse_args()
    main(args)
