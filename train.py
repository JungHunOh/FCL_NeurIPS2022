import argparse
import importlib
from utils import *

MODEL_DIR=None
DATA_DIR = 'data/'
PROJECT='base'

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='cub200',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)
    parser.add_argument('--save', type=str, default=None)

    # about pre-training
    parser.add_argument('-epochs_base', type=int, default=100)
    parser.add_argument('-epochs_new', type=int, default=100)
    parser.add_argument('-lr_base', type=float, default=0.1)
    parser.add_argument('-lr_new', type=float, default=0.1)
    parser.add_argument('-schedule', type=str, default='Step',
                        choices=['Step', 'Milestone'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70])
    parser.add_argument('-step', type=int, default=40)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-temperature', type=int, default=16)
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')

    parser.add_argument('-batch_size_base', type=int, default=128)
    parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=100)
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                        choices=['ft_dot', 'ft_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos',
                        choices=['ft_dot', 'ft_cos', 'avg_cos', 'avg_cos_trans', 'ft_cos_trans', 'ft_cos_learnfeat']) # ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means using average data embedding and cosine classifier

    # for episode learning
    parser.add_argument('-train_episode', type=int, default=50)
    parser.add_argument('-episode_shot', type=int, default=5)
    parser.add_argument('-episode_way', type=int, default=10)
    parser.add_argument('-episode_query', type=int, default=40)

    # for cec
    parser.add_argument('-lrg', type=float, default=0.1) #lr for graph attention network
    parser.add_argument('-low_shot', type=int, default=5)
    parser.add_argument('-low_way', type=int, default=10)
    parser.add_argument('--inner_steps', type=int, default=5)
    parser.add_argument('--inner_lr', type=float, default=0.1)
    parser.add_argument('--outer_lr', type=float, default=0.1)
    parser.add_argument('--lamb_contrastive', type=float, default=1)
    parser.add_argument('--proto_lamb', type=float, default=0.1)
    parser.add_argument('--dropcontrastive_lamb', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--num_fullproto', type=int, default=100)
    parser.add_argument('--trans_lamb', type=float, default=1)
    parser.add_argument('--query_new_lamb', type=float, default=1)
    parser.add_argument('--fc_lamb', type=float, default=1)
    parser.add_argument('--num_heads', type=int, default=3)
    parser.add_argument('--freeze_epoch', type=int, default=200)
    parser.add_argument('--num_query_new', type=int, default=20)
    parser.add_argument('--num_query_base', type=int, default=10)
    parser.add_argument('--num_query_base_class', type=int, default=10)
    parser.add_argument('--num_plp', type=int, default=500)
    parser.add_argument('--kk', type=float, default=0.1)


    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')
    parser.add_argument('--meta', action='store_true', default=False)
    parser.add_argument('--second_order_maml', action='store_true', default=False)
    parser.add_argument('--meta_sgd', action='store_true', default=False)
    parser.add_argument('--data_dropout', action='store_true', default=False)
    parser.add_argument('--mixup', action='store_true', default=False)
    parser.add_argument('--new_loader', action='store_true', default=False)
    parser.add_argument('--feat_transform', action='store_true', default=False)
    parser.add_argument('--fullproto', action='store_true', default=False)

    # about training
    parser.add_argument('--save_fig', type=str, default='true')
    parser.add_argument('-gpu', default='0,1,2,3')
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-debug', action='store_true')

    return parser


if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)

    if args.save_fig == 'true':
        args.save_fig = True
    else:
        args.save_fig = False

    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()
