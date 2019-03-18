import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
def load_data():
    data_transforms = transforms.Compose([transforms.RandomRotation(50),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
    validation_datasets = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
    vloaders = torch.utils.data.DataLoader(image_datasets, batch_size=32, shuffle=True)
    tloaders = torch.utils.data.DataLoader(image_datasets, batch_size=32, shuffle=True)
    return image_datasets, validation_datasets, test_datasets, dataloaders, vloaders, tloaders
	
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# TODO: Build and train your network
import torchvision.models as models
vgg16 = models.vgg16(pretrained=True)
alexnet = models.alexnet(pretrained=True)
model = models.densenet121(pretrained=True)

# Hyperparameters for our network
def nn_build(lr=0.001,hidden_units=512,dropout=0.5):
    from collections import OrderedDict
    hidden_sizes = [256, 128]
    #output_size=number of classes
    output_size = 102
    # Build a feed-forward network
    #Input size is 1024 for dense net
    classifier = nn.Sequential(OrderedDict([
                      ('dropout',nn.Dropout(dropout)),
                      ('fc1', nn.Linear(1024, hidden_units)),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_units, hidden_sizes[0])),
                      ('relu2', nn.ReLU()),
                      ('fc3', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                      ('relu3', nn.ReLU()),
                      ('logits', nn.Linear(hidden_sizes[1], output_size)),
                      ('output', nn.LogSoftmax(dim=1))
                 ]))
    model.classifier=classifier
    #Defining Loss Function
    criterion = nn.CrossEntropyLoss()
    #Adam Function, specify learning rate
    optimizer = optim.Adam(model.parameters(), float(lr))
    model.to('cuda')
    return model , optimizer ,criterion 

	# TODO: Save the checkpoint 
def save_checkpoint(image_datasets, path='classifier.pth',hidden_units=120,dropout=0.5,lr=0.1,epochs=2):
    model.class_to_idx = image_datasets.class_to_idx
    model.cpu
    torch.save({'structure' :'densenet121',
            'hidden_units':hidden_units,
            'dropout': dropout,
             'lr':lr,
             'nb_of_epochs':epochs,
             'state_dict':model.state_dict(),
             'class_to_idx':model.class_to_idx},
             path)

def load_model(path='classifier.pth'):
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_units']
    model,_,_ = nn_build(lr=0.1,hidden_units=hidden_layer1,dropout=0.5)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model
   
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
	
def process_image(path):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch       
    model, returns an Numpy array
    '''
    # Open the image
    from PIL import Image
    img = Image.open(path)
    print(img)
    # Resize the image
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Cropping the image 
    r_margin = (img.width+224)/2 
    l_margin = (img.width-224)/2
    t_margin = (img.height+224)/2 
    b_margin = (img.height-224)/2
    img = img.crop((l_margin, b_margin, r_margin,   
                      t_margin))
    # Normalize the omage array
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406]) 
    image = np.array(img)/255
    image = (image - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    image = image.transpose((2, 0, 1))
    
    return image

def predict(path, model, top_num=3):
    # Processing the image
    print('model')
    print(model)
    print('path')
    print(path[0])
    image = process_image(path[0])
    # Converting from numpy to tensor
    #image_tensor = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    image_tensor = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    model_input = image_tensor.unsqueeze_(0)
    # This gives the probability of labels
    probabilities = torch.exp(model.forward(model_input))
    # Getting the topmost probability out of all the predicted probabilities
    probabilities_top, labels_top = probabilities.topk(top_num)
    probabilities_top = probabilities_top.detach().cuda()
    probabilities_top=probabilities_top.cpu().numpy().tolist()[0] 
    labels_top = labels_top.detach().cuda()
    labels_top = labels_top.cpu().numpy().tolist()[0] 
    # Convert indices to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    labels = [idx_to_class[label] for label in labels_top]
    flowers_top = [cat_to_name[idx_to_class[label]] for label in labels_top]
    print(probabilities_top)
    return probabilities_top, labels, flowers_top

def train_nn(dataloaders,vloaders,model,optimizer,criterion,epochs=6,print_every=20):
    steps = 0
    loss=[]
    running_loss=0
    # change to cuda
    model.to('cuda')
    for e in range(epochs):
        for ii, (inputs, labels) in enumerate(dataloaders):
            steps += 1
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
        
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()      
        
            if steps % print_every == 0:
                model.eval()
                vlost = 0
                accuracy=0
        
                for ii, (inputs2,labels2) in enumerate(vloaders):
                    optimizer.zero_grad()
                
                    inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
                    model.to('cuda:0')
                    with torch.no_grad():    
                        outputs = model.forward(inputs2)
                        vlost = criterion(outputs,labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
                validation_loss= vlost / len(vloaders)
                accuracy = accuracy /len(vloaders)
            
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss {:.4f}".format(validation_loss),
                       "Accuracy: {:.4f}".format(accuracy))
                running_loss=0