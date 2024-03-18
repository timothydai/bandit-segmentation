# parser.add_argument('--exp_name', required=True)
# parser.add_argument(
#     '--feature_fn',
#     choices=['color', 'color_pos', 'mean_pool', 'filters', 'pretrained', 'deep'],
#     required=True
# )
# parser.add_argument('--shuffle_pixels', type=bool, default=False)

python main.py --exp_name color --feature_fn color
python main.py --exp_name color_pos --feature_fn color_pos
python main.py --exp_name mean_pool --feature_fn mean_pool
python main.py --exp_name filters --feature_fn filters
python main.py --exp_name deep --feature_fn deep
python main.py --exp_name pretrained --feature_fn pretrained
#python main.py --exp_name color_shuffle --feature_fn color --shuffle_pixels 1
#python main.py --exp_name color_pos_shuffle --feature_fn color_pos --shuffle_pixels 1
#python main.py --exp_name mean_pool_shuffle --feature_fn mean_pool --shuffle_pixels 1
#python main.py --exp_name filters_shuffle --feature_fn filters --shuffle_pixels 1
#python main.py --exp_name deep_shuffle --feature_fn deep --shuffle_pixels 1
#python main.py --exp_name pretrained_shuffle --feature_fn pretrained --shuffle_pixels 1

python main.py --exp_name resnet --img_embedder resnet
python main.py --exp_name vgg --img_embedder vgg
python main.py --exp_name vit --img_embedder vit
python main.py --exp_name efficientnet --img_embedder efficientnet
python main.py --exp_name inception --img_embedder inception
