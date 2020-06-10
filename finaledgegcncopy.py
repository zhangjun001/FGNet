import glob
import torch
from multiprocessing import Pool
import math
from torch.nn import Sequential, Linear, ReLU
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import DataParallel as GeoParallel
from torch_geometric.data import Data, Batch
from torchvision import datasets, transforms
import torchvision
import torch.utils.data as data
from torch.autograd import Variable
from skimage.color import rgb2hed
from skimage.util import img_as_float
from skimage.segmentation import slic, mark_boundaries
from skimage.exposure import rescale_intensity
from skimage import io
import cv2
import sys
import numpy as np
import random
from skimage import segmentation
import matplotlib.pyplot as plt
import torch.nn.init
from PIL import ImageFile, Image
import os
from collections import defaultdict
from utilities import *
from model import *
from args_parser import *


#amp_handle = amp.init(enabled=True)

global_segments = 0
if args.half_precision: from apex import amp
ImageFile.LOAD_TRUNCATED_IMAGES = True
##CUDA_LAUNCH_BLOCKING=1 #if cuda fails to backpropagate use this toggle to debug
OUTPUT_SIZE = args.output_size
use_cuda = torch.cuda.is_available()

SAVE_PATH = args.train_path
SD_SAVE_PATH = args.checkpoint_path
DATA_PATH = args.input_path



class random_dataloader(torch.utils.data.Dataset):
    """
    Proposed Dataset with Random Matched Pairs
    This function is a dataset designed for small batch initialization.
    When initializing with small batches (< 8), less than ideal GCN convergence may occur.
    This dataset loads image patches from two subfolder: ./positives ./negatives
    -and combines positive and negative samples by adjustable ratio self.positive_negative_ratio
    -in one mini-batch.

    Keyword arguments:
    path --parent path which contains ./positives ./negatives subfolders
    """
    def __init__(self, path):
        super(random_dataloader, self).__init__()
        self.path = path
        self.positive_images = [names for names in os.listdir(os.path.join(path, 'positives'))]
        self.negative_images = [names for names in os.listdir(os.path.join(path, 'negatives'))]
        self.positive_image_files = []
        self.negative_image_files = []
        self.toTensor = transforms.ToTensor()
        self.positive_negative_ratio = 3  #3 positives and 1 negative in one mini-batch
        self.ratio_counter = 0
        self.negative_index = np.random.randint(low = 0, high = self.positive_negative_ratio + 1)
        for img in self.positive_images:
            if img[-4:] is not None and (img[-4:] == '.png' or img[-4:] == '.jpg'):
                img_file = os.path.join(os.path.join(path, 'positives'), "%s" % img)
                self.positive_image_files.append({
                    "img": img_file

                    })
        for img in self.negative_images:
            if img[-4:] is not None and (img[-4:] == '.png' or img[-4:] == '.jpg'):
                img_file = os.path.join(os.path.join(path, 'negatives'), "%s" % img)
                self.negative_image_files.append({
                    "img": img_file

                    })

    def __len__(self):
        return (len(self.positive_image_files) + len(self.negative_image_files))

    def __getitem__(self, index):
        if self.ratio_counter == self.negative_index:
            index = index % len(self.negative_images)
            data_file = self.negative_image_files[index]
        else:
            index = index % len(self.positive_images)
            data_file = self.positive_image_files[index]

        self.ratio_counter += 1
        if self.ratio_counter > self.positive_negative_ratio:
            self.negative_index = np.random.randint(low = 0, high = self.positive_negative_ratio + 1)
            self.ratio_counter = 0
        image = cv2.imread(data_file["img"])
        image = cv2.resize(image, (OUTPUT_SIZE, OUTPUT_SIZE))
        angle = np.random.randint(4)
        image = rotate(image, angle)
        image = cv2.flip(image, np.random.randint(2) - 1)
        if args.color_channel_separation:
            ihc_hed = rgb2hed(image)
            h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1))
            d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1))
            image = np.dstack((np.zeros_like(h), d, h))
            #image = zdh.transpose(2, 0, 1).astype('float32')/255
        name = data_file["img"]
        path, file = os.path.split(name)
        split_filename = file.split("_")
        gt_percent = float(split_filename[0])
        moe = float(split_filename[1])
        image = self.toTensor(image)
        return (image, name, gt_percent, moe)


class normal_dataloader(torch.utils.data.Dataset):
    """
    Typical Dataset
    This function loads images from path and returns:
    image:      image (resized, color channel separated(optional))
    name:       full image path(contains image name)
    gt_percent: Ground-Truth percent, an image-level weak annotation
    moe:        Margin of Error, an image-level weak annotation

    Keyword arguments:
    path --path which contains *.png or *.jpg image patches
    """
    def __init__(self, path):
        super(normal_dataloader, self).__init__()
        self.path = path
        self.images = [names for names in os.listdir(path)]
        self.image_files = []
        self.toTensor = transforms.ToTensor()

        for img in self.images:
            if img[-4:] is not None and (img[-4:] == '.png' or img[-4:] == '.jpg'):
                img_file = os.path.join(path, "%s" % img)
                self.image_files.append({
                    "img": img_file,
                    "label": "1"
                    })

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        index = index % len(self.image_files)
        data_file = self.image_files[index]
        image = cv2.imread(data_file["img"])
        image = cv2.resize(image, (OUTPUT_SIZE, OUTPUT_SIZE))
        angle = np.random.randint(4)
        image = rotate(image, angle)
        image = cv2.flip(image, np.random.randint(2) - 1)
        if args.color_channel_separation:
            ihc_hed = rgb2hed(image)
            h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1))
            d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1))
            image = np.dstack((np.zeros_like(h), d, h))
            #image = zdh.transpose(2, 0, 1).astype('float32')/255
        name = data_file["img"]
        path, file = os.path.split(name)
        split_filename = file.split("_")
        gt_percent = float(split_filename[0])
        moe = float(split_filename[1])
        image = self.toTensor(image)
        return (image, name, gt_percent, moe)


class inference_dataset(torch.utils.data.Dataset):
    """
    dataset loader used when inferencing
    """
    def __init__(self, path):
        super(inference_dataset, self).__init__()
        self.path = path
        self.images = [names for names in os.listdir(path)]
        self.image_files = []
        self.toTensor = transforms.ToTensor()

        for img in self.images:
            if img[-4:] is not None and (img[-4:] == '.jpg' or img[-4:] == '.png'):
                img_file = os.path.join(path, "%s" % img)
                #print(img)
                #print(img_file)
                self.image_files.append({
                    "img": img_file,
                    "label": "1"
                    })

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        index = index % len(self.image_files)
        data_file = self.image_files[index]
        image = cv2.imread(data_file["img"])
        image = cv2.resize(image, (OUTPUT_SIZE, OUTPUT_SIZE))
        name = data_file["img"]
        image = self.toTensor(image)
        return (image, name, 0,0)


def load_dataset(batch_size):
    """
    Typical Pytorch Dataloader
    This function loades image from Dataset

    Keyword arguments:
    path --please refer to class random_dataloader and class normal_dataloader
    """
    data_path = DATA_PATH
    train_dataset = normal_dataloader(path = data_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size= batch_size, # <8*1024 res @ P40 cards, <25*512 res @ 1 P40 card
        num_workers= 8, #depends on RAM and CPU
        shuffle=True
    )
    return train_loader

def inference_loader(batch_size = 1):
    """
    Typical Pytorch Dataloader
    This function loades image from Dataset

    Keyword arguments:
    path --please refer to class random_dataloader and class normal_dataloader
    """
    data_path = DATA_PATH
    train_dataset = inference_dataset(path = data_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size= batch_size, # <8*1024 res @ P40 cards, <25*512 res @ 1 P40 card
        num_workers= 8, #depends on RAM and CPU
        shuffle=False
    )
    return train_loader

def multithread_slic(multi_input):
    """
    Multi-Thread SLIC superpixel
    Since SLIC algorithm is implemented on CPU, we use Python Pool to accelerate
    -SLIC algorithm by taking full advantage of CPU's multi-threading capability.
    This function also calculate mutable edges' weight as one of GCN's input data.


    Keyword arguments:
    multi_input   --tuple lists containing:
                    (FCN output, Max Channel Response, Mutable Edge Weight Ratio)
    """
    (multi_output, max_channel_response, weight_ratio, i) = multi_input
    if i > 0:
        multi_output_slic = slic(multi_output, n_segments = i, compactness = args.compactness\
                                 , sigma = 0, multichannel = True)
    else:
        multi_output_slic = slic(multi_output, n_segments = global_segments, compactness = args.compactness\
                                 , sigma = 0, multichannel = True)

    num_segments = len(np.unique(multi_output_slic))
    multi_adj = adjacency3(multi_output_slic,num_segments)
##    true_euclidean_distance = []
##    f_norm = []
##    mses = []
    chisq = []
    classes_raw = np.zeros((num_segments, args.nChannel),dtype="float32")
    for y in range(OUTPUT_SIZE):
        for x in range(OUTPUT_SIZE):
            curr_index = multi_output_slic[x][y]
            max_channel = max_channel_response[x][y]
            classes_raw[curr_index][max_channel] += 1.0

    for x in range(len(classes_raw)):
        max_in_row = np.amax(classes_raw[x])
        classes_raw[x] = classes_raw[x] / max_in_row

    for (p1, p2) in multi_adj:
        p1_class = np.asarray(classes_raw[p1])
        p2_class = np.asarray(classes_raw[p2])
        #true_euclidean_distance.append( euclidean_dist(p1center, p2center) )
        #f_norm.append( fnorm(p1_class,p2_class) )
        chisq.append( np.absolute(chisq_dist(p1_class, p2_class)) )
        #mses.append( mse(p1_class, p2_class) )
    chisq_max_value = np.amax(chisq)
    chisq = chisq / chisq_max_value
    #complementary_weight = np.ones_like(chisq) - chisq
    #chisq = chisq + weight_ratio * complementary_weight
    edge_weight = torch.from_numpy(chisq)
    return multi_output_slic, multi_adj, edge_weight, num_segments


if __name__ == '__main__':
    # train
    model = FCN(3, args.nChannel)
    modelgcn = GCN(args.nChannel, 1)
    gcn_batch_iter = 1 #how many iteration on one GCN batch
    batch_counter = 0
    global_segments = args.num_superpixels #mutable superpixel quantity
    change_dataloader = args.switch_iter #switch GCN into small batch training after # of iterations
    model_loss = 99999 #variable keeping track of FCN loss during warmup phase
    in_GCN = False #True when FCN exiting warmup phase and feeding output into GCN
    inference_mode = False
    #slic_multiscale_descending = False #True when mutable superpixel quantity is decreasing per iteration
    slic_adjust_ratio = args.adjust_iter #each iteration, global_segments *= (1+/- slic_adjust_ratio)
    weight_ratio =  args.weight_ratio #edge weight complementing (to 1) ratio, decreasing with training
    warmup_threshold = args.warmup_threshold# 0.5 #when FCN warmup loss reaches #, terminate warmup and start training GCN
    half_precision = args.half_precision

    if args.checkpoint > 0:
        #if given a checkpoint to resume training
        model.load_state_dict(torch.load(os.path.join(SD_SAVE_PATH, "FCN" + str(args.checkpoint) + ".pt")))
        modelgcn.load_state_dict(torch.load(os.path.join(SD_SAVE_PATH, "GCN" + str(args.checkpoint) + ".pt")))
        batch_counter = int(args.checkpoint)
        change_dataloader = batch_counter - 1
        in_GCN = True

    if args.inference_path is not None:
        dataset_loader = inference_loader()
        inference_mode = True
        args.maxIter = 1
        if args.checkpoint < 1:
            print("please define a inference checkpoint using --checkpoint")
            exit(-1)
    else:
        dataset_loader = load_dataset(args.batch_size)

    if not inference_mode and args.cpu_threads > args.batch_size:
        args.cpu_threads = args.batch_size

    if use_cuda:
        model = model.to('cuda')
        modelgcn = modelgcn.to('cuda')
        if half_precision:
            model, optimizer = amp.initialize(model, optimizer)
            modelgcn, optimizergcn = amp.initialize(modelgcn, optimizergcn)
    else:
        print("model using CPU, please check CUDA settings")

    if use_cuda:
        if torch.cuda.device_count() > 1:
            print(str(torch.cuda.device_count()) + " GPUs visible")
            model = nn.DataParallel(model)
            modelgcn = GeoParallel(modelgcn)
    if inference_mode:
        model.train()
    else:
        model.train()
    modelgcn = modelgcn.float() #necessary for edge_weight initialized training
    loss_fn = torch.nn.CrossEntropyLoss()
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.fcn_lr, momentum=0)
        optimizergcn = optim.SGD(modelgcn.parameters(), lr=args.gcn_lr, momentum=0)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.fcn_lr)
        optimizergcn = optim.Adam(modelgcn.parameters(), lr=args.gcn_lr)
    else:
        print("please reselect optimizer, curr value:", args.optimizer)
        exit(-1)




    if args.checkpoint > 0:
        #if given a checkpoint to resume training
        optimizer.load_state_dict(torch.load(os.path.join(SD_SAVE_PATH, "FCNopt" + str(args.checkpoint) + ".pt")))
        optimizergcn.load_state_dict(torch.load(os.path.join(SD_SAVE_PATH, "GCNopt" + str(args.checkpoint) + ".pt")))
        print("training successfully resumed!")


    #an RGB colors array used to visualize channel response
    label_colours = np.array([[0,0,0], [255,255,255],[0,0,255], [0,255,0],
                              [255,0,0],[128,0,0],[0,128,0],[0,0,128],
                              [255,255,0], [255,128,0], [128,255,0],
                              [0,255,255],[255,0,255],[255,255,255],
                              [128,128,128],[255,0,128],[0,128,255],
                              [128,0,255],[0,255,128],[100,200,200],
                              [200,100,100],[200,255,0],[100,255,0],
                              [200,0,255],[30,99,212],[40,222,100],
                              [100,200,25],[30,199,20],[0,211,200],
                              [3,44,122],[23,44,100],[90,22,0],[233,111,222],
                              [122,122,150],[0,233,149],[3,111,23]])


    for epoch in range(args.maxIter):

        """switch large batch into small batch"""
        # if batch_counter < change_dataloader:
        #     dataset_loader = load_dataset(2)
        # else:
        #     gcn_batch_iter = 1
        #     dataset_loader = load_dataset(2)
        #

        for batch_idx, (data, name,  gt_percent, moe) in \
            enumerate(dataset_loader):

            if not inference_mode:
                print("iteration: " + str(batch_counter) + " epoch: " + str(epoch))
            else:
                print("---------------------------------------------------")
                print("inferencing ", str(os.path.basename(name[0])))
            if args.visualize:
                """visualize using opencv"""
                originalimg = cv2.imread(name[0])
                originalimg = cv2.resize(originalimg, (OUTPUT_SIZE,OUTPUT_SIZE))
                cv2.imshow("original", originalimg)
                cv2.waitKey(1)

            batch_counter += 1 #records in-epoch progress
            if use_cuda:
                data = data.to('cuda')
            optimizer.zero_grad()
            output = model(data)



            nLabels = -1

            if not inference_mode and ((not in_GCN and batch_counter % 50 == 0) or (in_GCN and batch_counter % 10 == 0)):
                """FCN output visualization, either save to SAVE_PATH or display using opencv"""
                #model.eval()
                #output = model(data)
                ignore, target = torch.max( output, 1 )
                im_target = target.data.cpu().numpy() #label map original
                num_in_minibatch = 0
                for i in im_target:
                    im_target = i.flatten()
                    nLabels = len(np.unique(im_target))
                    label_num = rank(im_target)
                    label_rank = [i[0] for i in label_num]
                    im_target_rgb = np.array([label_colours [label_rank.index(c)] for c in im_target])
                    im_target_rgb = im_target_rgb.reshape( OUTPUT_SIZE, OUTPUT_SIZE, 3 ).astype( np.uint8 )
                    curr_filename = name[num_in_minibatch][-16:-4]
                    if args.visualize:
                        cv2.imshow("pre", im_target_rgb)
                        cv2.waitKey(1)
                    else:
                        cv2.imwrite(os.path.join(SAVE_PATH, 'PRE' + str(batch_counter) \
                                                 + 'N' + str(num_in_minibatch) + "_" \
                                                 + str(curr_filename) + '.png'), \
                                    cv2.cvtColor(im_target_rgb, cv2.COLOR_RGB2BGR))
                    num_in_minibatch += 1
                torch.save(model.module.state_dict(), os.path.join(SD_SAVE_PATH,"FCN" + str(batch_counter) + ".pt"))



            if not inference_mode and (model_loss > warmup_threshold and not in_GCN): #stable 0.3 2000epoch
                """warning up FCN, loss is the cross entropy between pixel-wise
                   -max channel responses and FCN model's output"""
                ignore, target = torch.max( output, 1 )
                loss = loss_fn(output, target)
                if half_precision:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
                if model_loss > loss.data:
                    model_loss = loss.data
                print (epoch, '/', args.maxIter, ':', nLabels, loss.data)
                change_dataloader += 1
                continue
            else:
                """when FCN model_loss is below warmup_threshold, enter GCN traning"""
                #optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
                #change the learning rate of FCN to a less conservative number
                #optimizer = optim.Adam(model.parameters(), lr=0.00005)
                #dataset_loader = load_dataset(2)
                in_GCN = True

            if model_loss < warmup_threshold or in_GCN:
                """Proposed method bridging output of FCN and input to GCN"""

                """mutable superpixel count, changes every training iterations"""
                """count changes by a fixed pattern"""
                # if slic_multiscale_descending:
                #     global_segments = int(global_segments * (1 - slic_adjust_ratio))
                #     if global_segments < 2000:
                #         slic_adjust_ratio = slic_adjust_ratio * 0.7
                #         slic_multiscale_descending = False
                # else:
                #     global_segments = int(global_segments * (1 + slic_adjust_ratio))
                #     if global_segments > 8000:
                #         global_segments = global_segments + 17
                #         slic_multiscale_descending = True
                """count is randomized between [2000, 8000)"""
                global_segments = int(random.random() * 6000.0 + 2000.0)

                if not inference_mode: print("slic segments count:" + str(global_segments))

                gcn_batch_list = [] #a batch which later feed into GCN
                segments_list = [] #SLIC segmentations of each image in current GCN batch
                batch_node_num = [] #number of nodes(superpixel tiles) that each image have in\
                                    #current GCN batch

                print("computing multithread slic")
                """prepares FCN output for using in <def multithread_slic()>,
                    computes SLIC segmentations, adjacency edges, & edge weight using
                    multi-threads"""
                multi_input = []
                for multi_one_graph in output:
                    multi_one_graph = multi_one_graph.permute( 1 , 2 , 0 )
                    _, max_channel_response = torch.max(multi_one_graph, 2)
                    multi_one_graph = multi_one_graph.cpu().detach().numpy().astype(np.float64)
                    max_channel_response = max_channel_response.cpu().detach().numpy()
                    if not inference_mode:
                        multi_input.append((multi_one_graph, max_channel_response, weight_ratio, -1))
                    else:
                        for i in (2000,3000,4000,5000,6000,7000,8000):
                            multi_input.append((multi_one_graph, max_channel_response, weight_ratio, i))
                with Pool(args.cpu_threads) as p:
                    multi_slic_adj_list = p.map(multithread_slic, multi_input)
                print("multithread slic finished")
                if weight_ratio > 0.2:
                    weight_ratio = weight_ratio * 0.99 #reduce edge weight's complementary

                for one_graph, (segments, adj, edge_weight, num_segments) in zip(output, multi_slic_adj_list):
                    """Bridging FCN's output into GCN's input, initialize GCN batch"""
                    segments_list.append(segments)
                    one_graph = one_graph.permute( 1 , 2 , 0 )
                    #one_graph.shape: [img_size, img_size, channel_size]
                    original_one_graph = torch.flatten(one_graph, start_dim = 0, end_dim = 1)
                    #original_one_graph.shape: [channel_size, img_size*img_size]
                    one_graph = None

                    batch_node_num.append(num_segments)

                    slic_pixels = [[] for _ in range (num_segments)]
                    """slic_pixels stores x,y(flatten) corrdinates according to superpixel's index"""
                    for y in range (OUTPUT_SIZE):
                        for x in range (OUTPUT_SIZE):
                            curr_label = segments[x,y]
                            slic_pixels[curr_label].append(x * OUTPUT_SIZE + y)
                            #each slic seg's x y axis
                    classes = None

                    """
                    For each superpixel's tile, select the PyTorch Variable(FCN) inside.
                    These Variables(FCN) combines into new Nodes Variable(GCN) while
                    -carrying gradients.
                    """
                    for n in slic_pixels:
                        index_tensor = torch.LongTensor(n)
                        if use_cuda:
                            index_tensor = index_tensor.to('cuda')
                        one_class = torch.unsqueeze(torch.sum( \
                            torch.index_select(original_one_graph, dim = 0, index = index_tensor), \
                            0), dim = 0)
                        if classes is None:
                            classes = one_class
                        else:
                            classes = torch.cat((classes, one_class), 0)
                        one_class = None
                        index_tensor = None
                    original_one_graph = None
                    temp_ind = 0
                    adj = np.asarray(adj)
                    adj = torch.from_numpy(adj)
                    adj = adj.t().contiguous()
                    adj = Variable(adj).type(torch.LongTensor)

                    """datagcn: GCN-ready wrapped data for one image
                        gcn_batch_list: GCN-ready minibatch containings same images from
                        -previous FCN's minibatch."""
                    datagcn = Data(x = classes, edge_index = adj, \
                                   edge_weight = edge_weight.type(torch.FloatTensor))

                    gcn_batch_list.append(datagcn)
                    #print(gcn_batch_list)
                    classes = None #releases cached GPU memory immediately
                    adj = None
                    datagcn = None
                if torch.cuda.device_count() == 1:
                    gcn_batch_list =  Batch.from_data_list(gcn_batch_list)
                    if use_cuda:
                        gcn_batch_list = gcn_batch_list.to('cuda')
                """GCN training iterations"""
                if inference_mode:
                    modelgcn.train()
                else:
                    modelgcn.train()
                    print(gt_percent.data.cpu().numpy())
                    print(moe.data.cpu().numpy())
                for epochgcn in range(0, gcn_batch_iter):
                    optimizergcn.zero_grad()
                    outgcn = modelgcn(gcn_batch_list)

                    """visualize GCN output"""
                    if not inference_mode and epochgcn == gcn_batch_iter - 1 and batch_counter % 10 == 0:
                        start_index = 0
                        counter = 0
                        #modelgcn.eval()
                        #outgcn = modelgcn(gcn_batch_list)
                        for curr_batch_idx in range(len(name)):
                            outgcn_slice = torch.narrow(input = outgcn, dim = 0, \
                                                        start = start_index, \
                                                        length = batch_node_num[curr_batch_idx])
                            start_index += batch_node_num[curr_batch_idx]
                            outputgcn_np = outgcn_slice.detach().data.cpu().numpy()
                            segments_copy = segments_list[curr_batch_idx].copy()
                            segments_copy = segments_copy.astype(np.float64)
                            for segInd in range(len(outputgcn_np)):
                                segments_copy[segments_copy == segInd] = outputgcn_np[segInd]
                            gcn_target_rgb = np.array([[255*(c + 1) / 2, 255*(c + 1) / 2, 255*(c + 1) / 2] \
                                                       for c in segments_copy])
                            gcn_target_rgb = np.moveaxis(gcn_target_rgb, 1, 2)
                            gcn_target_rgb = gcn_target_rgb.reshape( (OUTPUT_SIZE,OUTPUT_SIZE,3) ).astype( np.uint8 )
                            if args.visualize:
                                cv2.imshow("gcn", gcn_target_rgb)
                                cv2.waitKey(1)
                            else:
                                cv2.imwrite(os.path.join(SAVE_PATH, 'GCN' + str(batch_counter) + "N" + str(counter)\
                                                         + "_" + str(name[curr_batch_idx][-16:-4]) + '.png'), \
                                            cv2.cvtColor(gcn_target_rgb, cv2.COLOR_RGB2BGR))
                            counter += 1
                            outgcn_slice = None

                    if not inference_mode:
                        """
                        loss_top    --cost, positive <gt_percent-moe> % of nodes responded correctly
                        loss_bottom --cost, negative <1-gt_percent-moe> % of nodes responded correctly
                        positive refers to desired region(cancer), negative refers to other regions(background)
                        """
                        loss_top, loss_bottom = one_label_loss(gt_percent = gt_percent.data.cpu().numpy(), \
                                                               predict = outgcn, \
                                                               moe = moe.data.cpu().numpy(), \
                                                               batch_node_num = batch_node_num)
                        if loss_top is None:
                            total_gcn_loss = loss_bottom
                        elif loss_bottom is None:
                            total_gcn_loss = loss_top
                        else:
                            total_gcn_loss = loss_top + loss_bottom
                        if half_precision:
                            with amp.scale_loss(total_gcn_loss, optimizergcn) as scaled_loss2:
                                scaled_loss2.backward(retain_graph=True)
                        else:
                            total_gcn_loss.backward(retain_graph=True)
                        #backward calculating GCN gradients according to combined loss
                        #print(total_gcn_loss)
                        print("GCN+FCN loss: " + str(total_gcn_loss.data.cpu().numpy()))
                if not inference_mode:
                    #backpropagate through GCN's layers
                    optimizergcn.step()
                    #backpropagate accumulated gradients through FCN's layers
                    optimizer.step()
                    #saving models & optimizers state_dict for later training and inference
                    #cpu = torch.device("cpu")
                    #model = model.to(cpu)
                    #modelgcn = modelgcn.to(cpu)
                    torch.save(model.module.state_dict(), os.path.join(SD_SAVE_PATH,"FCN" + str(batch_counter) + ".pt"))
                    torch.save(modelgcn.module.state_dict(), os.path.join(SD_SAVE_PATH,"GCN" + str(batch_counter) + ".pt"))
                    torch.save(optimizer.state_dict(), os.path.join(SD_SAVE_PATH,"FCNopt" + str(batch_counter) + ".pt"))
                    torch.save(optimizergcn.state_dict(), os.path.join(SD_SAVE_PATH,"GCNopt" + str(batch_counter) + ".pt"))
                    #model = model.to('cuda')
                    #modelgcn = modelgcn.to('cuda')
            if inference_mode:
                start_index = 0
                counter = 0
                final_map = np.zeros((args.output_size, args.output_size))
                multi_input = []
                print("fusing")
                for input_index in range(len(segments_list)):
                    outgcn_numpy = torch.narrow(input = outgcn, dim = 0, start = start_index, length = batch_node_num[input_index]).detach().data.cpu().numpy()
                    segments_copy = segments_list[input_index].astype(np.float64)
                    multi_input.append((outgcn_numpy, segments_copy))
                    start_index += batch_node_num[input_index]
                with Pool(args.cpu_threads) as p:
                    multi_graph = p.map(fuse_results, multi_input)
                for graph in multi_graph:
                    final_map += graph

                final_map = final_map / float(len(multi_graph))
                final_map += 1.0
                final_map = final_map / 2.0
                final_map[final_map < args.fuse_thresh] = 0

                gcn_target_rgb = np.array([[255 * c , 255* c , 255* c] for c in final_map])
                gcn_target_rgb = np.moveaxis(gcn_target_rgb, 1, 2)
                gcn_target_rgb = gcn_target_rgb.reshape( (args.output_size,args.output_size,3) ).astype( np.uint8 )
                if args.visualize:
                    cv2.imshow("gcn", gcn_target_rgb)
                    cv2.waitKey(1)
                else:
                    basename = os.path.basename(name[0])
                    cv2.imwrite(os.path.join(args.inference_path, basename), cv2.cvtColor(gcn_target_rgb, cv2.COLOR_RGB2BGR))
                    print("inference for", str(basename), "saved to", str(args.inference_path))
