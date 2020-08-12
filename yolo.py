from __future__ import division

#from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
# from test import evaluate

# from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

#import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

#######

import torch
import torch.nn as nn
import torchvision.utils as ut

import numpy as np
import cv2

class DummyLayer(nn.Module):
	def __init__(self):
		super(DummyLayer, self).__init__()

class DetectionLayer(nn.Module):
	def __init__(self, anchors):
		super(DetectionLayer, self).__init__()
		self.anchors = anchors

def conv_layer(i, in_c, out_c, k, s, p):
	seq = nn.Sequential()
	seq.add_module("conv_"+str(i), nn.Conv2d(in_c, out_c, kernel_size=(k, k), stride=(s, s), padding=(p, p), bias=False))
	seq.add_module("batch_norm_"+str(i), nn.BatchNorm2d(out_c, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
	seq.add_module("leaky_"+str(i), nn.LeakyReLU(negative_slope=0.1, inplace=True))
	return seq

def shortcut_layer(i):
	seq = nn.Sequential()
	seq.add_module("shortcut_"+str(i), DummyLayer())
	return seq

def single_conv_layer(i, in_c, out_c, k, s):
	seq = nn.Sequential()
	seq.add_module("conv_"+str(i), nn.Conv2d(in_c, out_c, kernel_size=(k, k), stride=(s, s)))
	return seq

def yolo_layer(i, anchors):
	seq = nn.Sequential()
	seq.add_module("Detection_"+str(i), DetectionLayer(anchors))
	return seq

def upsample_layer(i):
	seq = nn.Sequential()
	seq.add_module("upsample_"+str(i), nn.Upsample(scale_factor=2.0, mode="bilinear"))
	return seq

def route_layer(i):
	seq = nn.Sequential()
	seq.add_module("route"+str(i), DummyLayer())
	return seq

class Darknet(nn.Module):

	def __init__(self):
		super(Darknet, self).__init__()

		self.module_list = nn.ModuleList()

		# conv_layer(inputLayers, outputLayers, kernelSize, stride, padding)
		self.module_list.append(conv_layer(0, 3, 32, 3, 1, 1))
		self.module_list.append(conv_layer(1, 32, 64, 3, 2, 1))
		self.module_list.append(conv_layer(2, 64, 32, 1, 1, 0))
		self.module_list.append(conv_layer(3, 32, 64, 3, 1, 1))

		self.module_list.append(shortcut_layer(4))

		self.module_list.append(conv_layer(5, 64, 128, 3, 2, 1))
		self.module_list.append(conv_layer(6, 128, 64, 1, 1, 0))
		self.module_list.append(conv_layer(7, 64, 128, 3, 1, 1))

		self.module_list.append(shortcut_layer(8))

		self.module_list.append(conv_layer(9, 128, 64, 1, 1, 0))
		self.module_list.append(conv_layer(10, 64, 128, 3, 1, 1))

		self.module_list.append(shortcut_layer(11))

		self.module_list.append(conv_layer(12, 128, 256, 3, 2, 1))
		self.module_list.append(conv_layer(13, 256, 128, 1, 1, 0))
		self.module_list.append(conv_layer(14, 128, 256, 3, 1, 1))

		self.module_list.append(shortcut_layer(15))

		self.module_list.append(conv_layer(16, 256, 128, 1, 1, 0))
		self.module_list.append(conv_layer(17, 128, 256, 3, 1, 1))

		self.module_list.append(shortcut_layer(18))

		self.module_list.append(conv_layer(19, 256, 128, 1, 1, 0))
		self.module_list.append(conv_layer(20, 128, 256, 3, 1, 1))

		self.module_list.append(shortcut_layer(21))

		self.module_list.append(conv_layer(22, 256, 128, 1, 1, 0))
		self.module_list.append(conv_layer(23, 128, 256, 3, 1, 1))

		self.module_list.append(shortcut_layer(24))

		self.module_list.append(conv_layer(25, 256, 128, 1, 1, 0))
		self.module_list.append(conv_layer(26, 128, 256, 3, 1, 1))

		self.module_list.append(shortcut_layer(27))

		self.module_list.append(conv_layer(28, 256, 128, 1, 1, 0))
		self.module_list.append(conv_layer(29, 128, 256, 3, 1, 1))

		self.module_list.append(shortcut_layer(30))

		self.module_list.append(conv_layer(31, 256, 128, 1, 1, 0))
		self.module_list.append(conv_layer(32, 128, 256, 3, 1, 1))

		self.module_list.append(shortcut_layer(33))

		self.module_list.append(conv_layer(34, 256, 128, 1, 1, 0))
		self.module_list.append(conv_layer(35, 128, 256, 3, 1, 1))

		self.module_list.append(shortcut_layer(36))

		self.module_list.append(conv_layer(37, 256, 512, 3, 2, 1))
		self.module_list.append(conv_layer(38, 512, 256, 1, 1, 0))
		self.module_list.append(conv_layer(39, 256, 512, 3, 1, 1))

		self.module_list.append(shortcut_layer(40))

		self.module_list.append(conv_layer(41, 512, 256, 1, 1, 0))
		self.module_list.append(conv_layer(42, 256, 512, 3, 1, 1))

		self.module_list.append(shortcut_layer(43))

		self.module_list.append(conv_layer(44, 512, 256, 1, 1, 0))
		self.module_list.append(conv_layer(45, 256, 512, 3, 1, 1))

		self.module_list.append(shortcut_layer(46))

		self.module_list.append(conv_layer(47, 512, 256, 1, 1, 0))
		self.module_list.append(conv_layer(48, 256, 512, 3, 1, 1))

		self.module_list.append(shortcut_layer(49))

		self.module_list.append(conv_layer(50, 512, 256, 1, 1, 0))
		self.module_list.append(conv_layer(51, 256, 512, 3, 1, 1))

		self.module_list.append(shortcut_layer(52))

		self.module_list.append(conv_layer(53, 512, 256, 1, 1, 0))
		self.module_list.append(conv_layer(54, 256, 512, 3, 1, 1))

		self.module_list.append(shortcut_layer(55))

		self.module_list.append(conv_layer(56, 512, 256, 1, 1, 0))
		self.module_list.append(conv_layer(57, 256, 512, 3, 1, 1))

		self.module_list.append(shortcut_layer(58))

		self.module_list.append(conv_layer(59, 512, 256, 1, 1, 0))
		self.module_list.append(conv_layer(60, 256, 512, 3, 1, 1))

		self.module_list.append(shortcut_layer(61))

		self.module_list.append(conv_layer(62, 512, 1024, 3, 2, 1))
		self.module_list.append(conv_layer(63, 1024, 512, 1, 1, 0))
		self.module_list.append(conv_layer(64, 512, 1024, 3, 1, 1))

		self.module_list.append(shortcut_layer(65))

		self.module_list.append(conv_layer(66, 1024, 512, 1, 1, 0))
		self.module_list.append(conv_layer(67, 512, 1024, 3, 1, 1))

		self.module_list.append(shortcut_layer(68))

		self.module_list.append(conv_layer(69, 1024, 512, 1, 1, 0))
		self.module_list.append(conv_layer(70, 512, 1024, 3, 1, 1))

		self.module_list.append(shortcut_layer(71))

		self.module_list.append(conv_layer(72, 1024, 512, 1, 1, 0))
		self.module_list.append(conv_layer(73, 512, 1024, 3, 1, 1))

		self.module_list.append(shortcut_layer(74))

		self.module_list.append(conv_layer(75, 1024, 512, 1, 1, 0))
		self.module_list.append(conv_layer(76, 512, 1024, 3, 1, 1))
		self.module_list.append(conv_layer(77, 1024, 512, 1, 1, 0))
		self.module_list.append(conv_layer(78, 512, 1024, 3, 1, 1))
		self.module_list.append(conv_layer(79, 1024, 512, 1, 1, 0))
		self.module_list.append(conv_layer(80, 512, 1024, 3, 1, 1))

		self.module_list.append(single_conv_layer(81, 1024, 255, 1, 1))

		self.module_list.append(yolo_layer(82, [(116, 90), (156, 198), (373, 326)]))

		self.module_list.append(shortcut_layer(83))

		self.module_list.append(conv_layer(84, 512, 256, 1, 1, 0))

		self.module_list.append(upsample_layer(85))

		self.module_list.append(route_layer(86))

		self.module_list.append(conv_layer(87, 768, 256, 1, 1, 0))
		self.module_list.append(conv_layer(88, 256, 512, 3, 1, 1))
		self.module_list.append(conv_layer(89, 512, 256, 1, 1, 0))
		self.module_list.append(conv_layer(90, 256, 512, 3, 1, 1))
		self.module_list.append(conv_layer(91, 512, 256, 1, 1, 0))
		self.module_list.append(conv_layer(92, 256, 512, 3, 1, 1))

		self.module_list.append(single_conv_layer(93, 512, 255, 1, 1))

		self.module_list.append(yolo_layer(94, [(30, 61), (62, 45), (59, 119)]))

		self.module_list.append(route_layer(95))

		self.module_list.append(conv_layer(96, 256, 128, 1, 1, 0))

		self.module_list.append(upsample_layer(97))

		self.module_list.append(route_layer(98))

		self.module_list.append(conv_layer(99, 384, 128, 1, 1, 0))
		self.module_list.append(conv_layer(100, 128, 256, 3, 1, 1))
		self.module_list.append(conv_layer(101, 256, 128, 1, 1, 0))
		self.module_list.append(conv_layer(102, 128, 256, 3, 1, 1))
		self.module_list.append(conv_layer(103, 256, 128, 1, 1, 0))
		self.module_list.append(conv_layer(104, 128, 256, 3, 1, 1))

		self.module_list.append(single_conv_layer(105, 256, 255, 1, 1))

		self.module_list.append(yolo_layer(106, [(10, 13), (16, 30), (33, 23)]))
		#print(self.module_list)

	def forward(self, image, targets = None):

		loss = 0

		outputs = {}

		outputs[0] = self.module_list[0](image)
		outputs[1] = self.module_list[1](outputs[0])
		outputs[2] = self.module_list[2](outputs[1])
		outputs[3] = self.module_list[3](outputs[2])

		outputs[4] = outputs[4 - 1] + outputs[4 - 3]

		outputs[5] = self.module_list[5](outputs[4])
		outputs[6] = self.module_list[6](outputs[5])
		outputs[7] = self.module_list[7](outputs[6])

		outputs[8] = outputs[8 - 1] + outputs[8 - 3]

		outputs[9] = self.module_list[9](outputs[8])
		outputs[10] = self.module_list[10](outputs[9])

		outputs[11] = outputs[11 - 1] + outputs[11 - 3]

		outputs[12] = self.module_list[12](outputs[11])
		outputs[13] = self.module_list[13](outputs[12])
		outputs[14] = self.module_list[14](outputs[13])

		outputs[15] = outputs[15 - 1] + outputs[15 - 3]

		outputs[16] = self.module_list[16](outputs[15])
		outputs[17] = self.module_list[17](outputs[16])

		outputs[18] = outputs[18 - 1] + outputs[18 - 3]

		outputs[19] = self.module_list[19](outputs[18])
		outputs[20] = self.module_list[20](outputs[19])

		outputs[21] = outputs[21 - 1] + outputs[21 - 3]

		outputs[22] = self.module_list[22](outputs[21])
		outputs[23] = self.module_list[23](outputs[22])

		outputs[24] = outputs[24 - 1] + outputs[24 - 3]

		outputs[25] = self.module_list[25](outputs[24])
		outputs[26] = self.module_list[26](outputs[25])

		outputs[27] = outputs[27 - 1] + outputs[27 - 3]

		outputs[28] = self.module_list[28](outputs[27])
		outputs[29] = self.module_list[29](outputs[28])

		outputs[30] = outputs[30 - 1] + outputs[30 - 3]

		outputs[31] = self.module_list[31](outputs[30])
		outputs[32] = self.module_list[32](outputs[31])

		outputs[33] = outputs[33 - 1] + outputs[33 - 3]

		outputs[34] = self.module_list[34](outputs[33])
		outputs[35] = self.module_list[35](outputs[34])

		outputs[36] = outputs[36 - 1] + outputs[36 - 3]

		outputs[37] = self.module_list[37](outputs[36])
		outputs[38] = self.module_list[38](outputs[37])
		outputs[39] = self.module_list[39](outputs[38])

		outputs[40] = outputs[40 - 1] + outputs[40 - 3]

		outputs[41] = self.module_list[41](outputs[40])
		outputs[42] = self.module_list[42](outputs[41])

		outputs[43] = outputs[43 - 1] + outputs[43 - 3]

		outputs[44] = self.module_list[44](outputs[43])
		outputs[45] = self.module_list[45](outputs[44])

		outputs[46] = outputs[46 - 1] + outputs[46 - 3]

		outputs[47] = self.module_list[47](outputs[46])
		outputs[48] = self.module_list[48](outputs[47])

		outputs[49] = outputs[49 - 1] + outputs[49 - 3]

		outputs[50] = self.module_list[50](outputs[49])
		outputs[51] = self.module_list[51](outputs[50])

		outputs[52] = outputs[52 - 1] + outputs[52 - 3]

		outputs[53] = self.module_list[53](outputs[52])
		outputs[54] = self.module_list[54](outputs[53])

		outputs[55] = outputs[55 - 1] + outputs[55 - 3]

		outputs[56] = self.module_list[56](outputs[55])
		outputs[57] = self.module_list[57](outputs[56])

		outputs[58] = outputs[58 - 1] + outputs[58 - 3]

		outputs[59] = self.module_list[59](outputs[58])
		outputs[60] = self.module_list[60](outputs[59])

		outputs[61] = outputs[61 - 1] + outputs[61 - 3]

		outputs[62] = self.module_list[62](outputs[61])
		outputs[63] = self.module_list[63](outputs[62])
		outputs[64] = self.module_list[64](outputs[63])

		outputs[65] = outputs[65 - 1] + outputs[65 - 3]

		outputs[66] = self.module_list[66](outputs[65])
		outputs[67] = self.module_list[67](outputs[66])

		outputs[68] = outputs[68 - 1] + outputs[68 - 3]

		outputs[69] = self.module_list[69](outputs[68])
		outputs[70] = self.module_list[70](outputs[69])

		outputs[71] = outputs[71 - 1] + outputs[71 - 3]

		outputs[72] = self.module_list[72](outputs[71])
		outputs[73] = self.module_list[73](outputs[72])

		outputs[74] = outputs[74 - 1] + outputs[74 - 3]

		outputs[75] = self.module_list[75](outputs[74])
		outputs[76] = self.module_list[76](outputs[75])
		outputs[77] = self.module_list[77](outputs[76])
		outputs[78] = self.module_list[78](outputs[77])
		outputs[79] = self.module_list[79](outputs[78])
		outputs[80] = self.module_list[80](outputs[79])

		outputs[81] = self.module_list[81](outputs[80])

		# Yolo Layer
		x = outputs[81]
		x = x.data
		x = self.detection_preprocess(x, 608, [(116, 90), (156, 198), (373, 326)], 80)
		detections = x

		outputs[82] = outputs[82 - 1]

		# Route Layer
		layers = [79]
		outputs[83] = outputs[layers[0]]

		outputs[84] = self.module_list[84](outputs[83])

		# Upsample Layer
		outputs[85] = self.module_list[85](outputs[84])

		# Route Layer
		layers = [85, 61]
		outputs[86] = torch.cat((outputs[layers[0]], outputs[layers[1]]), 1)

		outputs[87] = self.module_list[87](outputs[86])
		outputs[88] = self.module_list[88](outputs[87])
		outputs[89] = self.module_list[89](outputs[88])
		outputs[90] = self.module_list[90](outputs[89])
		outputs[91] = self.module_list[91](outputs[90])
		outputs[92] = self.module_list[92](outputs[91])

		outputs[93] = self.module_list[93](outputs[92])

		# Yolo Layer
		x = outputs[93]
		x = x.data
		#x = self.detection_preprocess(x, 608, [(30, 61), (62, 45), (59, 119)], 80)
		x = self.detection_preprocess(x, 608, self.module_list[94][0].anchors, 80)

		detections = torch.cat((detections, x), 1)
		outputs[94] = outputs[94 - 1]

		# Route Layer
		layers = [91]
		outputs[95] = outputs[layers[0]]

		outputs[96] = self.module_list[96](outputs[95])

		# Upsample Layer
		outputs[97] = self.module_list[97](outputs[96])

		# Route Layer
		layers = [97, 36]
		outputs[98] = torch.cat((outputs[layers[0]], outputs[layers[1]]), 1)

		outputs[99] = self.module_list[99](outputs[98])
		outputs[100] = self.module_list[100](outputs[99])
		outputs[101] = self.module_list[101](outputs[100])
		outputs[102] = self.module_list[102](outputs[101])
		outputs[103] = self.module_list[103](outputs[102])
		outputs[104] = self.module_list[104](outputs[103])

		outputs[105] = self.module_list[105](outputs[104])

		# Yolo Layer
		x = outputs[105]
		x = x.data
		x = self.detection_preprocess(x, 608, self.module_list[106][0].anchors, 80)

		detections = torch.cat((detections, x), 1)
		outputs[106] = outputs[106 - 1]


		out = outputs[106]
		outputs = None

		return detections if targets is None else (loss, detections)

	@staticmethod
	def detection_preprocess(x,inp_dim,anchors,num_classes,CUDA=False):
		"""
		This function will take input_dimension_of_image,anchors and number of classes as input 
		"""
		# x --> 4D feature map
		batch_size = x.size(0)
		grid_size = x.size(2)
		stride =  inp_dim // x.size(2)   # factor by which current feature map reduced from input

	
		bbox_attrs = 5 + num_classes #5 + 80
		num_anchors = len(anchors) #3
	
		#[1, 255, 13, 13]
		prediction = x.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size) # 1x255x169     
		prediction = prediction.transpose(1,2).contiguous() #1x169x255
		prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs) #1x507x85

		# the dimension of anchors is wrt original image.We will make it corresponding to feature map
		anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

		#Sigmoid the  centre_X, centre_Y. and object confidencce
		prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
		prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
		prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
	
		#Add the center offsets
		grid = np.arange(grid_size)
		a,b = np.meshgrid(grid, grid)

		x_offset = torch.FloatTensor(a).view(-1,1) #(1,gridsize*gridsize,1)
		y_offset = torch.FloatTensor(b).view(-1,1)

		if CUDA:
			x_offset = x_offset.cuda()
			y_offset = y_offset.cuda()

		x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
	

		prediction[:,:,:2] += x_y_offset

		#log space transform height and the width
		anchors = torch.FloatTensor(anchors)

		if CUDA:
			anchors = anchors.cuda()

		anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
		prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors #width and height
		prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))    
		prediction[:,:,:4] *= stride    
		return prediction

	def load_weights(self, weightfile):
		
		#Open the weights file
		fp = open(weightfile, "rb")

		#The first 4 values are header information 
		# 1. Major version number
		# 2. Minor Version Number
		# 3. Subversion number 
		# 4. IMages seen 
		
		header = np.fromfile(fp, dtype = np.int32, count = 5)
		# header = torch.from_numpy(header)
		# self.seen = self.header[3]
		
		#The rest of the values are the weights
		# Let's load them up
		weights = np.fromfile(fp, dtype = np.float32)

		ptr = 0
		for i in range(len(self.module_list)):
			
			if i not in [4, 8, 11, 15, 18, 21, 24, 27, 30, 33, 36, 40, 43, 46, 49, 52, 55, 58, 61, 65, 68, 71, 74, 82, 83, 85, 86, 94, 95, 97, 98, 106]: #module_type == "convolutional":
				model = self.module_list[i]

				if i not in [81, 93, 105]:
					batch_normalize = True
				else:
					batch_normalize = False

				conv = model[0]
				
				if (batch_normalize):
					bn = model[1]
					
					#Get the number of weights of Batch Norm Layer
					num_bn_biases = bn.bias.numel()
					
					#Load the weights
					bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
					ptr += num_bn_biases
					
					bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
					ptr  += num_bn_biases
					
					bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
					ptr  += num_bn_biases
					
					bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
					ptr  += num_bn_biases
					
					#Cast the loaded weights into dims of model weights. 
					bn_biases = bn_biases.view_as(bn.bias.data)
					bn_weights = bn_weights.view_as(bn.weight.data)
					bn_running_mean = bn_running_mean.view_as(bn.running_mean)
					bn_running_var = bn_running_var.view_as(bn.running_var)

					#Copy the data to model
					bn.bias.data.copy_(bn_biases)
					bn.weight.data.copy_(bn_weights)
					bn.running_mean.copy_(bn_running_mean)
					bn.running_var.copy_(bn_running_var)
				
				else:
					#Number of biases
					num_biases = conv.bias.numel()
				
					#Load the weights
					conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
					ptr = ptr + num_biases
					
					#reshape the loaded weights according to the dims of the model weights
					conv_biases = conv_biases.view_as(conv.bias.data)
					
					#Finally copy the data
					conv.bias.data.copy_(conv_biases)
					
					
				#Let us load the weights for the Convolutional layers
				# we are loading weights as common beacuse when batchnormalization is present there is no bias for conv layer
				num_weights = conv.weight.numel()
				
				#Do the same as above for weights
				conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
				ptr = ptr + num_weights

				conv_weights = conv_weights.view_as(conv.weight.data)
				conv.weight.data.copy_(conv_weights)
				# Note: we dont have bias for conv when batch normalization is there

def bounding_box_iou(box1, box2):
	"""
	Returns the IoU of two bounding boxes 
	
	"""
	#Get the coordinates of bounding boxes
	b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
	b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
	
	#get the corrdinates of the intersection rectangle
	inter_rect_x1 =  torch.max(b1_x1, b2_x1)
	inter_rect_y1 =  torch.max(b1_y1, b2_y1)
	inter_rect_x2 =  torch.min(b1_x2, b2_x2)
	inter_rect_y2 =  torch.min(b1_y2, b2_y2)
	
	#Intersection area

	intersection_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
 
	#Union Area
	b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
	b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
	
	iou = intersection_area / (b1_area + b2_area - intersection_area)
	
	return iou

def final_detection(prediction, confidence_threshold, num_classes, nms_conf = 0.4):
	# taking only values above a particular threshold and set rest everything to zero
	mask = (prediction[:,:,4] > confidence_threshold).float().unsqueeze(2)
	prediction = prediction*mask
	
	
	#(center x, center y, height, width) attributes of our boxes, 
	#to (top-left corner x, top-left corner y, right-bottom corner x, right-bottom corner y)
	box_corner = prediction.new(prediction.shape)
	box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
	box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
	box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
	box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
	prediction[:,:,:4] = box_corner[:,:,:4]
	
	batch_size = prediction.size(0)
	write = False
	
	# we can do non max suppression only on individual images so we will loop through images
	for ind in range(batch_size):  
		image_pred = prediction[ind] 
		# we will take only those rows with maximm class probability
		# and corresponding index
		max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
		max_conf = max_conf.float().unsqueeze(1)
		max_conf_score = max_conf_score.float().unsqueeze(1)
		combined = (image_pred[:,:5], max_conf, max_conf_score)
		# concatinating index values and max probability with box cordinates as columns
		image_pred = torch.cat(combined, 1) 
		#Remember we had set the bounding box rows having a object confidence
		# less than the threshold to zero? Let's get rid of them.
		non_zero_index =  (torch.nonzero(image_pred[:,4])) # non_zero_ind will give the indexes 
		image_pred_ = image_pred[non_zero_index.squeeze(),:].view(-1,7)
		try:
			#Get the various classes detected in the image
			img_classes = torch.unique(image_pred_[:,-1]) # -1 index holds the class index
		except:
			 continue
	   
		for cls in img_classes:
			#perform NMS
			#get the detections with one particular class
			cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
			# taking the non zero indexes
			class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
			image_pred_class = image_pred_[class_mask_ind].view(-1,7)
			
			# sort them based on probability #getting index
			conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
			image_pred_class = image_pred_class[conf_sort_index]
			idx = image_pred_class.size(0)
			
			for i in range(idx):
				#Get the IOUs of all boxes that come after the one we are looking at 
				 #in the loop
				try:
					ious = bounding_box_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
				except ValueError:
					break
				except IndexError:
					break
				
				#Zero out all the detections that have IoU > treshhold
				iou_mask = (ious < nms_conf).float().unsqueeze(1)
				image_pred_class[i+1:] *= iou_mask
				
				#Remove the non-zero entries
				non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
				image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
		  
			#Concatenate the batch_id of the image to the detection
			#this helps us identify which image does the detection correspond to 
			#We use a linear straucture to hold ALL the detections from the batch
			#the batch_dim is flattened
			#batch is identified by extra batch column
			
			#creating a row with index of images
			batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
			seq = batch_ind, image_pred_class
			if not write:
				output = torch.cat(seq,1)
				write = True
			else:
				out = torch.cat(seq,1)
				output = torch.cat((output,out))
	return output

# function to load the classes
def load_classes(class_file):
	fp = open(class_file, "r")
	names = fp.read().split("\n")[:-1]
	return names

# function converting images from opencv format to torch format
def preprocess_image(img, inp_dim):
	"""
	Prepare image for inputting to the neural network. 
	
	Returns processed image, original image and original image dimension  
	"""

	orig_im = cv2.imread(img)
	dim = orig_im.shape[1], orig_im.shape[0]
	img = (canvas_image(orig_im, (inp_dim, inp_dim)))
	img = img[:,:,::-1]
	img_ = img.transpose((2,0,1)).copy()
	img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
	return img_, orig_im, dim

#function letterbox_image that resizes our image, keeping the 
# aspect ratio consistent, and padding the left out areas with the color (128,128,128)
def canvas_image(img, conf_inp_dim):
	'''resize image with unchanged aspect ratio using padding'''
	img_w, img_h = img.shape[1], img.shape[0]
	w, h = conf_inp_dim  # dimension from configuration file

	ratio = min(w/img_w, h/img_h)

	new_w = int(img_w * ratio)
	new_h = int(img_h * ratio)
	resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
	
	# we fill the extra pixels with 128
	canvas = np.full((conf_inp_dim[1], conf_inp_dim[0], 3), 128)

	canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,:] = resized_image
	
	return canvas

def draw_boxes(x, img):
	c1 = tuple(x[1:3].int())
	c2 = tuple(x[3:5].int())
	cls = int(x[-1])
	label = "{0}".format(classes[cls])
	color = (0,0,255)
	cv2.rectangle(img, c1, c2,color, 2)
	t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
	c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
	cv2.rectangle(img, c1, c2,color, -1)
	cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
	return img


if __name__ == "__main__":

	weightsfile = "./yolov3.weights"
	classfile = "./coco.names"

	CUDA = False
	image_name = "bedroom.jpg"
	nms_thesh = 0.5

	model = Darknet()
	model.load_weights(weightsfile)
	classes = load_classes(classfile)
	print('Classes and weights loaded.')

	conf_inp_dim = 608

	processed_image, original_image, original_img_dim = preprocess_image(image_name, conf_inp_dim)

	im_dim = original_img_dim[0], original_img_dim[1]
	im_dim = torch.FloatTensor(im_dim).repeat(1,2)

	if CUDA:
		im_dim = im_dim_list.cuda()
		model.cuda()

	#Set the model in evaluation mode
	model.eval()
	with torch.no_grad():
		prediction = model(processed_image)

	output = final_detection(prediction, confidence_threshold=0.5, num_classes=80, nms_conf = nms_thesh)
	im_dim_list = torch.index_select(im_dim, 0, output[:,0].long())

	scaling_factor = torch.min(conf_inp_dim/im_dim_list,1)[0].view(-1,1)
	output[:,[1,3]] -= (conf_inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
	output[:,[2,4]] -= (conf_inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
	output[:,1:5] /= scaling_factor

	# adjusting bounding box size between 0 and configuration image size
	output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(conf_inp_dim))

	list(map(lambda x: draw_boxes(x, original_image), output))
	cv2.imwrite("out.png", original_image)


if __name__ == "__main__":
	# parser = argparse.ArgumentParser()
	# parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
	# parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
	# parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
	# parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
	# parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
	# parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
	# parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
	# parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
	# parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
	# parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
	# parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
	# parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
	# opt = parser.parse_args()
	# print(opt)

	logger = Logger("logs")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	os.makedirs("output", exist_ok=True)
	os.makedirs("checkpoints", exist_ok=True)

	# Get data configuration
	#data_config = "config/coco.data"
	train_path = "data/custom/train.txt"
	valid_path = "data/custom/valid.txt"
	class_names = load_classes("data/custom/classes.names")
	weightsfile = "./yolov3.weights"
	img_size = 416

	numEpochs = 10
	evaluationInterval = 1
	checkpointInterval = 1

	# Initiate model
	model = Darknet()#.to(device)
	model.load_weights(weightsfile)

	# If specified we start from checkpoint
	#if opt.pretrained_weights:
	#	if opt.pretrained_weights.endswith(".pth"):
	#		model.load_state_dict(torch.load(opt.pretrained_weights))
	#	else:
	#		model.load_darknet_weights(opt.pretrained_weights)

	# Get dataloader
	dataset = ListDataset(train_path, augment=True, multiscale=True)
	dataloader = torch.utils.data.DataLoader(
		dataset,
		batch_size=8,
		shuffle=True,
		num_workers=8,
		pin_memory=True,
		collate_fn=dataset.collate_fn,
	)

	optimizer = torch.optim.Adam(model.parameters())

	metrics = [
		"grid_size",
		"loss",
		"x",
		"y",
		"w",
		"h",
		"conf",
		"cls",
		"cls_acc",
		"recall50",
		"recall75",
		"precision",
		"conf_obj",
		"conf_noobj",
	]

	for epoch in range(numEpochs):
		model.train()
		start_time = time.time()
		for batch_i, (_, imgs, targets) in enumerate(dataloader):
			batches_done = len(dataloader) * epoch + batch_i

			imgs = Variable(imgs.to(device))
			targets = Variable(targets.to(device), requires_grad=False)

			loss, outputs = model(imgs, targets)
			loss.backward()

			if batches_done % 2:
				# Accumulates gradient before each step
				optimizer.step()
				optimizer.zero_grad()

			# ----------------
			#   Log progress
			# ----------------

			log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, numEpochs, batch_i, len(dataloader))

			metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

			# Log metrics at each YOLO layer
			for i, metric in enumerate(metrics):
				formats = {m: "%.6f" for m in metrics}
				formats["grid_size"] = "%2d"
				formats["cls_acc"] = "%.2f%%"
				row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
				metric_table += [[metric, *row_metrics]]

				# Tensorboard logging
				tensorboard_log = []
				for j, yolo in enumerate(model.yolo_layers):
					for name, metric in yolo.metrics.items():
						if name != "grid_size":
							tensorboard_log += [(f"{name}_{j+1}", metric)]
				tensorboard_log += [("loss", loss.item())]
				logger.list_of_scalars_summary(tensorboard_log, batches_done)

			log_str += AsciiTable(metric_table).table
			log_str += f"\nTotal loss {loss.item()}"

			# Determine approximate time left for epoch
			epoch_batches_left = len(dataloader) - (batch_i + 1)
			time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
			log_str += f"\n---- ETA {time_left}"

			print(log_str)

			model.seen += imgs.size(0)

		if epoch % evaluationInterval == 0:
			print("\n---- Evaluating Model ----")
			# Evaluate the model on the validation set
			precision, recall, AP, f1, ap_class = evaluate(
				model,
				path=valid_path,
				iou_thres=0.5,
				conf_thres=0.5,
				nms_thres=0.5,
				img_size=img_size,
				batch_size=8,
			)
			evaluation_metrics = [
				("val_precision", precision.mean()),
				("val_recall", recall.mean()),
				("val_mAP", AP.mean()),
				("val_f1", f1.mean()),
			]
			logger.list_of_scalars_summary(evaluation_metrics, epoch)

			# Print class APs and mAP
			ap_table = [["Index", "Class name", "AP"]]
			for i, c in enumerate(ap_class):
				ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
			print(AsciiTable(ap_table).table)
			print(f"---- mAP {AP.mean()}")

		if epoch % checkpointInterval == 0:
			torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
