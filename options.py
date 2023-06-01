import argparse

# basic settings
parser = argparse.ArgumentParser(description='P-MIL for WTAL')
parser.add_argument('--exp_dir', type=str, default='outputs', help='the directory of experiments')
parser.add_argument('--run_type', type=str, default='train', help='train or test (default: train)')

# dataset patameters
parser.add_argument('--dataset_name', type=str, default='Thumos14reduced', help='dataset to train on')
parser.add_argument('--dataset_root', type=str, default='data/Thumos14reduced', help='dataset root path')
parser.add_argument('--base_method', type=str, default='base', help='baseline S-MIL method name')

# model parameters
parser.add_argument('--num_class', type=int, default=20, help='number of classes (default: 20)')
parser.add_argument('--feature_size', type=int, default=2048, help='size of feature (default: 2048)')
parser.add_argument('--roi_size', type=int, default=12, help='roi size for proposal features extraction (default: 12)')
parser.add_argument('--max_proposal', type=int, default=1000, help='maximum number of proposal during training (default: 1000)')
parser.add_argument('--pretrained_ckpt', type=str, default=None, help='ckpt for pretrained model')

# training paramaters
parser.add_argument('--batch_size', type=int, default=10, help='number of instances in a batch of data (default: 10)')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate (default: 0.0001)')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight deacy (default: 0.001)')
parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout value (default: 0.5)')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--max_epoch', type=int, default=200, help='maximum epoch to train (default: 200)')

parser.add_argument("--rampup_length", type=int, default=30, help='the rampup epoch (default: 30)')
parser.add_argument('--interval', type=int, default=10, help='epoch interval for performing the test (default: 10)')
parser.add_argument('--k', type=float, default=8, help='top-k for aggregating video-level classification (default: 8)')
parser.add_argument('--gamma', type=float, default=0.8, help='threshold for select pseudo instances (default: 0.8)')

parser.add_argument('--weight_loss_prop_mil_orig', type=float, default=2, help='weight of loss_prop_mil_orig')
parser.add_argument('--weight_loss_prop_mil_supp', type=float, default=1, help='weight of loss_prop_mil_supp')
parser.add_argument('--weight_loss_prop_irc', type=float, default=2, help='weight of loss_prop_irc')
parser.add_argument('--weight_loss_prop_comp', type=float, default=20, help='weight of loss_prop_comp')

# testing parameters
parser.add_argument('--threshold_cls', type=float, default=0.2, help='video-level classification threshold')
parser.add_argument('--gamma_vid', type=float, default=0.2, help='contribution of the video-level score to the final proposal score')

