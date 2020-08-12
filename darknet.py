import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import cv2
import os
import time
from torch.autograd import Variable
import matplotlib.pyplot as plt

def parse_cfg(config_file):

	file = open(config_file,'r')
	file = file.read().split('\n')
	file =  [line for line in file if len(line)>0 and line[0] != '#']
	file = [line.lstrip().rstrip() for line in file]

	final_list = []
	element_dict = {}

	for line in file:

		if line[0] == '[':

			if len(element_dict) != 0:     # appending the dict stored on previous iteration

					final_list.append(element_dict)
					element_dict = {} # again emtying dict

			element_dict['type'] = ''.join([i for i in line if i != '[' and i != ']'])
			
		else:

			val = line.split('=')
			element_dict[val[0].rstrip()] = val[1].lstrip()  #removing spaces on left and right side
		
	final_list.append(element_dict) # appending the values stored for last set

	#print(final_list)

	return final_list

class DummyLayer(nn.Module):
	def __init__(self):
		super(DummyLayer, self).__init__()

class DetectionLayer(nn.Module):
	def __init__(self, anchors):
		super(DetectionLayer, self).__init__()
		self.anchors = anchors

def model_initialization(blocks):
	darknet_details = blocks[0]
	channels = 3 
	output_filters = []  #list of filter numbers in each layer.It is useful while defining number of filters in routing layer
	modulelist = nn.ModuleList()
	
	for i,block in enumerate(blocks[1:]):

		seq = nn.Sequential()
		if (block["type"] == "convolutional"):
			activation = block["activation"]
			filters = int(block["filters"])
			kernel_size = int(block["size"])
			strides = int(block["stride"])
			use_bias= False if ("batch_normalize" in block) else True
			pad = (kernel_size - 1) // 2
			conv = nn.Conv2d(in_channels=channels, out_channels=filters, kernel_size=kernel_size, 
							 stride=strides, padding=pad, bias = use_bias)

			seq.add_module("conv_{0}".format(i), conv)
			
			if "batch_normalize" in block:
				bn = nn.BatchNorm2d(filters)
				seq.add_module("batch_norm_{0}".format(i), bn)

			if activation == "leaky":
				activn = nn.LeakyReLU(0.1, inplace = True)
				seq.add_module("leaky_{0}".format(i), activn)
			
		elif (block["type"] == "upsample"):
			upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
			seq.add_module("upsample_{}".format(i), upsample)

		elif (block["type"] == 'route'):
			# start and end is given in format (eg:-1 36 so we will find layer number from it.
			# we will find layer number in negative format
			# so that we can get the number of filters in that layer
			block['layers'] = block['layers'].split(',')
			block['layers'][0] = int(block['layers'][0])
			start = block['layers'][0]
			if len(block['layers']) == 1:               
				#ie if -1 given and present layer is 20 . we have to sum filters in 19th and 20th layer 
				block['layers'][0] = int(i + start)             
				filters = output_filters[block['layers'][0]]  #start layer number
					   
			
			elif len(block['layers']) > 1:
				# suppose we have -1,28 and present layer is 20 we have sum filters in 19th and 28th layer
				block['layers'][0] = int(i + start) 
				# block['layers'][1] = int(block['layers'][1]) - i # end layer number  
				block['layers'][1] = int(block['layers'][1])
				filters = output_filters[block['layers'][0]] + output_filters[block['layers'][1]]

			# that means this layer don't have any forward operation
			route = DummyLayer()
			seq.add_module("route_{0}".format(i),route)

		elif block["type"] == "shortcut":
			from_ = int(block["from"])
			shortcut = DummyLayer()
			seq.add_module("shortcut_{0}".format(i),shortcut)
			
			
		elif block["type"] == "yolo":
			mask = block["mask"].split(",")
			mask = [int(m) for m in mask]
			anchors = block["anchors"].split(",")
			anchors = [(int(anchors[i]), int(anchors[i + 1])) for i in range(0, len(anchors), 2)]
			anchors = [anchors[i] for i in mask]
			block["anchors"] = anchors
			
			detectorLayer = DetectionLayer(anchors)
			seq.add_module("Detection_{0}".format(i),detectorLayer)
				
		modulelist.append(seq)
		output_filters.append(filters)     
		channels = filters
	
	return darknet_details, modulelist

class Darknet(nn.Module):
	def __init__(self, cfgfile):
		super(Darknet, self).__init__()
		self.blocks = parse_cfg(cfgfile)
		self.net_info, self.module_list = model_initialization(self.blocks)
		
	def forward(self, x, CUDA=False):
		modules = self.blocks[1:]
		#We cache the outputs for the route layer
		outputs = {}
		write = 0     
		for i, module in enumerate(modules):        
			module_type = (module["type"])
			if module_type == "convolutional" or module_type == "upsample":
				x = self.module_list[i](x)
				#print(i, module_type)
				outputs[i] = x
				
			elif module_type == "route":
				layers = module["layers"]
				#print(i, "route", layers)
				layers = [int(a) for a in layers]
				if len(layers) == 1:
					x = outputs[layers[0]]
				if len(layers) > 1:
					map1 = outputs[layers[0]]
					map2 = outputs[layers[1]]
					x = torch.cat((map1,map2),1)
					# print(map1.shape,map2.shape,x.shape)
				outputs[i] = x
				
			elif  module_type == "shortcut":
				from_ = int(module["from"])
				#print(i, "shortcut", from_)

				# just adding outputs for residual network
				x = outputs[i-1] + outputs[i+from_]  
				outputs[i] = x
				
			elif module_type == 'yolo':
				anchors = self.module_list[i][0].anchors
				
				#Get the input dimensions
				inp_dim = int(self.net_info["height"])
				#Get the number of classes
				num_classes = int(module["classes"])
				#print(i, "yolo", write, inp_dim, anchors, num_classes)
			
				#Transform 
				x = x.data   # get the data at that point
				x = self.detection_preprocess(x, inp_dim, anchors, num_classes)
				
				if not write:              #if no collector has been intialised.
					#print(i) 
					detections = x
					write = 1
				else:
					#print(i)
					detections = torch.cat((detections, x), 1)

				outputs[i] = outputs[i-1]
				
		try:
			return detections   #return detections if present
		except:
			return 0

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
			module_type = self.blocks[i + 1]["type"]

			# if module_type == "convolutional":
			# 	try:
			# 		batch_normalize = int(self.blocks[i+1]["batch_normalize"])
			# 	except:
			# 		batch_normalize = 0
			# 	print(i, "CONV", batch_normalize, ptr)
			# else:
			# 	print(i, "NONCONV", ptr)


			
			if module_type == "convolutional":
				model = self.module_list[i]
				try:
					batch_normalize = int(self.blocks[i+1]["batch_normalize"])
				except:
					batch_normalize = 0
				
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
				if i == 19: print(conv)
				if i == 19: print(num_weights)
				
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

# Utility functions

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




##############################################################################################
if __name__ == "__main__":

	cfgfile = "./cfg/yolov3.cfg"
	weightsfile = "./yolov3.weights"
	classfile = "./coco.names"

	CUDA = False
	image_name = "biking.jpg"
	nms_thesh = 0.5
	#Set up the neural network
	print("Loading network.....")
	model = Darknet(cfgfile)
	model.load_weights(weightsfile)
	print("Network successfully loaded")
	classes = load_classes(classfile)
	print('Classes loaded')

	conf_inp_dim = int(model.net_info["height"])#608

	# treading and resizing image
	processed_image, original_image, original_img_dim = preprocess_image(image_name,conf_inp_dim)
	#print(processed_image.shape)

	im_dim = original_img_dim[0], original_img_dim[1]
	im_dim = torch.FloatTensor(im_dim).repeat(1,2)

	#If there's a GPU availible, put the model on GPU
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