import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

import program

#Command Line Arguments

ap = argparse.ArgumentParser(
    description='predict-file')
ap.add_argument('ip_image', default='/home/workspace/aipnd-project/flowers/test/28/image_05230.jpg', nargs='*', action="store", type = str)
ap.add_argument('checkpoint', default='/home/workspace/aipnd-project/classifier.pth', nargs='*', action="store",type = str)
ap.add_argument('--top_num', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
#ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")
parse_args = ap.parse_args()
path = parse_args.ip_image
num_outputs = parse_args.top_k
#power = pa.gpu
#input_img = parse_args.input_img
checkpoint = parse_args.checkpoint
image_datasets, validation_datasets, test_datasets, training_loader, testing_loader, validation_loader = program.load_data()
print(checkpoint)
model=program.load_model(checkpoint)
print(model)
with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)
probabilities,labels,flowers_top = program.predict(path, model, num_outputs)
#labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
#probability = np.array(probabilities[0])
i=0
#print('probability')
#print(probability)
print('labels')
print(labels)
while i < 3:
    print("{} with a probability of {}".format(labels[i], probabilities[i]))
    i += 1
