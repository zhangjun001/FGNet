import argparse

parser = argparse.ArgumentParser(description='PyTorch Weakly Supervised Segmentation')
parser.add_argument('--nChannel', metavar='N', default=24, type=int,
                    help='number of channels of histogram')
parser.add_argument('--maxIter', metavar='T', default=10000, type=int,
                    help='number of maximum iterations')
parser.add_argument('--checkpoint', metavar='ckpt number', default=-1, type=int,
                    help='continue training from checkpoint #')
parser.add_argument('--num_superpixels', metavar='K', default=7000, type=int,
                    help='number of initial superpixels')
parser.add_argument('--compactness', metavar='C', default=3, type=float,
                    help='compactness of superpixels')
parser.add_argument('--visualize', metavar='1 or 0', default=0, type=int,
                    help='0:display image using OPENCV | 1:save images to train-path')
parser.add_argument('--color_channel_separation', metavar='True or False', default=False, type=bool,
                    help='Immunohistochemical staining colors separation toggle')
parser.add_argument('--half-precision', metavar='True or False', default=False, type=bool,
                    help='Half precision training, requires torch 0.4.0 and apex from nVidia')
parser.add_argument('--optimizer', default='Adam', help='optimizer(SGD|Adam)')
parser.add_argument('--gcn-lr', default=0.0005, type=float, help='gcn learning rate')
parser.add_argument('--fcn-lr', default=0.0001, type=float, help='fcn learning rate')
parser.add_argument('-b','--batch-size', default=2, type=int, help='batch size')
parser.add_argument('-t','--cpu-threads', default=8, type=int, help='number of threads for multiprocessing')
parser.add_argument('--switch-iter', default=13, type=int, help='switch GCN into small \
                    batch training after # of iterations')
parser.add_argument('--adjust-iter', default=0.1, type=float, help='each iteration, \
                    global_segments *= (1+/- slic_adjust_ratio)')
parser.add_argument('--weight-ratio', default=1.5, type=float, help='edge weight complementing \
                    (to 1) ratio, decreasing with training')
parser.add_argument('--warmup-threshold', default=2, type=float, help='when FCN warmup loss \
                    reaches #, terminate warmup and start training GCN')
parser.add_argument('--output-size', default=1024, type=int, help='The resolution along one axis\
                    of the image. input images will be scaled to the size defined.')
parser.add_argument('--fuse-thresh', default=0.001, type=float, help='the cutoff to use when \
                    outputting the final fused mask in inference')
parser.add_argument('--train-path', type=str,
                     default="./train_process_files",
                     help='a folder to save train progress visualization',
                     required=False)
parser.add_argument('--input-path', type=str,
                     default="./input_data",
                     help='training set containing the labeled images',
                     required=False)
parser.add_argument('--checkpoint-path', type=str,
                     default='./state_dict',
                     help='path where the checkpoints are saved to',
                     required=False)
parser.add_argument('--inference-path', type=str,
                     help='path where the inferenced masks are saved to',
                     required=False)
args = parser.parse_args()
