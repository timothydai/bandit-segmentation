import argparse
import glob
import numpy as np
import os
import pickle

from algos import linucb, naive
from features import *
from viz import save_img_mask_pair

def visualize_all_experiments_in(results_dir):
    for filename in glob.glob(f'{results_dir}/*'):
        if not filename.endswith('pkl'):
            continue
        with open(filename, 'rb') as f:
            img, pred, mask = pickle.load(f)
        save_img_mask_pair(img, pred, mask, (pred == mask).sum() / (img.shape[0] * img.shape[1]), f'{filename}_viz.png')
        
        pred = pred.reshape(-1)
        mask = mask.reshape(-1)
        print(
            (pred[:int(0.7 * len(pred))] == mask[:int(0.7 * len(pred))]).sum() / (int(0.7 * len(pred))), 
            (pred[int(0.7 * len(pred)):] == mask[int(0.7 * len(pred)):]).sum() / (int(0.3 * len(pred)))
            )


def main(args):
    with open('dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    dataset = dataset[:5]

    train = dataset[:int(0.7 * len(dataset))]
    test = dataset[int(0.7 * len(dataset)):]

    # for x in dataset:
    #     print(naive(x[0], x[1]))
    # assert not os.path.exists(save_dir)
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

    # print(dataset[-1][0])
    # assert False

    linucb(
        imgs=[dataset[-1][1]],
        features=[feature_fn(dataset[-1][1])],
        gt_masks=[dataset[-1][2]],
        save_dir=results_dir,
        foreground_A=np.identity(d),
        foreground_b=np.zeros(d),
        background_A=np.identity(d),
        background_b=np.zeros(d),
        train=True,
        shuffle_pixels=args.shuffle_pixels,
        internal_split=True,
    )
    visualize_all_experiments_in(results_dir)
    exit()
    # # Baby with shuffled pixels.
    for i in range(2):
        results_dir = f'results/baby_shuffle_{i}'
        # linucb(
        #     imgs=[dataset[-1][0]],
        #     features=[color_features(dataset[-1][0])],
        #     gt_masks=[dataset[-1][1]],
        #     save_dir=results_dir,
        #     foreground_A=np.identity(3),
        #     foreground_b=np.zeros(3),
        #     background_A=np.identity(3),
        #     background_b=np.zeros(3),
        #     train=True,
        #     shuffle_pixels=True,
        #     internal_split=True,
        # )
        visualize_all_experiments_in(results_dir)

    # # Run config 1: One model, with 70% train and 30% test.
    # results_dir = 'results/one_model'
    # preds, foreground_A, foreground_b, background_A, background_b = linucb(
    #     imgs=[x[0] for x in train],
    #     features=[color_features(x[0]) for x in train],
    #     gt_masks=[x[1] for x in train],
    #     save_dir=results_dir + '_train',
    #     foreground_A=np.identity(3),
    #     foreground_b=np.zeros(3),
    #     background_A=np.identity(3),
    #     background_b=np.zeros(3),
    #     train=True,
    #     shuffle_pixels=False,
    #     internal_split=False,
    # )
    # linucb(
    #     imgs=[x[0] for x in test],
    #     features=[color_features(x[0]) for x in test],
    #     gt_masks=[x[1] for x in test],
    #     save_dir=results_dir + '_test',
    #     foreground_A=foreground_A,
    #     foreground_b=foreground_b,
    #     background_A=background_A,
    #     background_b=background_b,
    #     train=False,
    #     shuffle_pixels=False,
    #     internal_split=False,
    # )
    # visualize_all_experiments_in(results_dir + '_train')
    # visualize_all_experiments_in(results_dir + '_test')
    
    # # Run config 2: One model per image, with internal split.
    # for i in range(len(dataset)):
    #     results_dir = f'results/one_model_per_image_{i}'
    #     linucb(
    #         imgs=[dataset[i][0]],
    #         features=[color_features(dataset[i][0])],
    #         gt_masks=[dataset[i][1]],
    #         save_dir=results_dir,
    #         foreground_A=np.identity(3),
    #         foreground_b=np.zeros(3),
    #         background_A=np.identity(3),
    #         background_b=np.zeros(3),
    #         train=True,
    #         shuffle_pixels=False,
    #         internal_split=True,
    #     )
    #     visualize_all_experiments_in(results_dir)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True)
    parser.add_argument(
        '--feature_fn',
        choices=['color', 'color_pos', 'mean_pool', 'filters', 'pretrained'],
        required=True
    )
    parser.add_argument('--shuffle_pixels', type=bool, default=False)
    args = parser.parse_args()
    main(args)