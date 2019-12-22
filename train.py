# importing libraries 
import argparse
from time import time, sleep
from os import listdir
import ast
import torch 
import torchvision
from torchvision import datasets, transforms, models 
from torch import nn 
from torch import optim 
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from time import time, sleep

def main():
    # start time for model 
    start_time_ = time()
    
    # getting arguments
    args = get_input_args() 
    
    # setting directories passed by user.
    data_dir=  args.data_dir
    saving_dir = args.saving_dir
    
    # loading datasets
    trainloader, validloader, testloader, train_datasets, valid_datasets, test_datasets = load_data_preparation(data_dir)
    
    
    
    # getting model 
    model, model_name = get_model(hidden_units=args.hidden_units,drop_p = args.drop_out, model_name=args.arch)
    
    #device = gpu_check(args.gpu)
    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    
    model.to(device)
    
   
    # defining loss and optimizer 
    optimizer = optim.Adam(model.classifier.parameters(), lr= args.learning_rate)
    criterion = nn.NLLLoss()
    
    

    
    # training model 
    train_network(model, trainloader, validloader, device,criterion, optimizer, args.epochs)
    
    # validate model 
    accuracy(model, testloader,trainloader, device)
    
    #saving model
    save_model(model, train_datasets, saving_dir)
    
    # time model finished compuation 
    end_time_ = time()
    
    # this code was by https://github.com/udacity/AIPND/tree/master/intropylab-classifying-images
    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time_ = end_time_ - start_time_
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time_/3600)))+":"+str(int((tot_time_%3600)/60))+":"
          +str(int((tot_time_%3600)%60)) )
    

 # function for arguments 
def get_input_args():
    parser = argparse.ArgumentParser(description="Train a Neural Network to Predict Images")
    parser.add_argument('--arch', type=str, help="choose archecture")
    parser.add_argument('--data_dir', type=str, required=True, help="directory of data to train network")
    parser.add_argument('--hidden_units',type=int, help="number of hidden units in the neural network")
    parser.add_argument('--gpu', action = "store_true",help= "GPU")
    parser.add_argument('--learning_rate', type=float,  default=0.001, help="learning rate for the model")
    parser.add_argument('--drop_out', type=float,  default=0.5, help="learning rate for the model")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training network")
    parser.add_argument('--saving_dir', type=str, default= "./", help="directory to save trained model and weights")
    
    args = parser.parse_args()
    
    return args

# function to load and preprocess data.
def load_data_preparation(data_dir):
    # Define means and standard deviation for transforms
    means = [0.485, 0.456, 0.406]
    deviations = [0.229, 0.224, 0.225]
    
    # Define path 
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Define transformers for train, testing and validation datasets. 
    test_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224), 
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(), 
                                     transforms.Normalize(means, deviations)])

    valid_transforms = transforms.Compose([transforms.Resize(256), 
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(means, deviations)])

    train_transforms = transforms.Compose([transforms.Resize(256), 
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(means, deviations)])
    
    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    # Using the image datasets and the trainforms, define the dataloaders 
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True) 
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64)
    
    return trainloader, validloader, testloader, train_datasets, valid_datasets, test_datasets


# function to load model using PIL image and passing it for test 
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # open imgae using PIL.Image
    img = Image.open(image)
    
    # define means and std
    means = [0.485, 0.456, 0.406]
    deviations = [0.229, 0.224, 0.225]
    # define transforms
    transform = transforms.Compose([transforms.Resize(256), 
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(means, deviations)])
    # apply transform to the image
    image = transform(img) 
    image.unsqueeze_(0)
    image = img.cuda().float()
    
    # return image 
    return image

# function for get and load model
def get_model(hidden_units,drop_p, model_name = 'vgg16'):
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        # freezing parameters to avoid backward propagation through them 
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units:
            classifier = nn.Sequential(nn.Linear(25088,4096),
                                             nn.ReLU(),
                                             nn.Dropout(p= drop_p), 
                                             nn.Linear(4096,1024),
                                             nn.ReLU(),
                                             nn.Dropout(p=drop_p),
                                             nn.Linear(1024,hidden_units),
                                             nn.ReLU(),
                                             nn.Dropout(p=drop_p),
                                             nn.Linear(hidden_units,128),
                                             nn.ReLU(),
                                             nn.Dropout(p=drop_p),
                                             nn.Linear(128,120),
                                             nn.LogSoftmax(dim=1))
        else:
            classifier = nn.Sequential(nn.Linear(25088,4096),
                                             nn.ReLU(),
                                             nn.Dropout(p= drop_p), 
                                             nn.Linear(4096,1024),
                                             nn.ReLU(),
                                             nn.Dropout(p=drop_p),
                                             nn.Linear(1024,784),
                                             nn.ReLU(),
                                             nn.Dropout(p=drop_p),
                                             nn.Linear(784,128),
                                             nn.ReLU(),
                                             nn.Dropout(p=drop_p),
                                             nn.Linear(128,120),
                                             nn.LogSoftmax(dim=1))
            
        
    else:
        exec("model = models.{}(pretrained=True)".format(model_name))
        # freezing parameters to avoid backward propagation through them 
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units:
            classifier = nn.Sequential(nn.Linear(25088,4096),
                                             nn.ReLU(),
                                             nn.Dropout(p= drop_p), 
                                             nn.Linear(4096,1024),
                                             nn.ReLU(),
                                             nn.Dropout(p=drop_p),
                                             nn.Linear(1024,hidden_units),
                                             nn.ReLU(),
                                             nn.Dropout(p=drop_p),
                                             nn.Linear(hidden_units,128),
                                             nn.ReLU(),
                                             nn.Dropout(p=drop_p),
                                             nn.Linear(128,120),
                                             nn.LogSoftmax(dim=1))
        else:
            classifier = nn.Sequential(nn.Linear(25088,4096),
                                       nn.ReLU(),
                                       nn.Dropout(p= drop_p), 
                                       nn.Linear(4096,1024),
                                       nn.ReLU(),
                                       nn.Dropout(p=drop_p),
                                       nn.Linear(1024,784),
                                       nn.ReLU(),
                                       nn.Dropout(p=drop_p),
                                       nn.Linear(784,128),
                                       nn.ReLU(),
                                       nn.Dropout(p=drop_p),
                                       nn.Linear(128,120),
                                       nn.LogSoftmax(dim=1))
            
    model.classifier = classifier
        
    return model, model_name

    
    
    
# function for validation process 
def validation(model, validloader, criterion, device):
    model.to(device)
    test_loss = 0 
    accuracy = 0 
    
    for images, labels in validloader:
        
        images, labels = images.to(device), labels.to(device)

        outputs = model.forward(images)
        test_loss += criterion(outputs, labels).item()

        # Calculate accuracy
        ps = torch.exp(outputs)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy   

# function to train the netowork    
def train_network(model, trainloader, validloader, device,criterion, optimizer, epochs):
    print("Starting Training ..... \n")
    start_time = time()
    running_loss = 0
    print_every = 40
    steps = 0
    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            steps += 1
            
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step() 

            running_loss += loss.item()
            
            
            if steps % print_every == 0:
                model.eval()  # eval mode for interference 

                #turn off gradient for validation to save memory and computation 
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion, device)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Valiation Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
                
                running_loss = 0
        
                # Make sure training is back on
                model.train()
                
    end_time = time()
    tot_time = end_time - start_time
    print("\n** Total Time Model Trained:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    
    
    
            

            
            
# function to return accruracy of the model      
def accuracy(model,testloader,trainloader, device):
    model.to(device)
    correct = 0 
    total = 0 
    with torch.no_grad():
        model.eval()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            outputs= model(images)
            _,predicted = torch.max(outputs.data,1)
            total += labels.size(0) 
            correct += (predicted == labels).sum().item()
        
    print("Accuracy of the network test images: %d %%" %(100*correct / total))
    
# function to save the model 
def save_model(model, train_datasets, saving_dir):
    model.class_to_idx = train_datasets.class_to_idx
    checkpoint = {'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, 'my_checkpoint.pth')
    print("Model & weights saved")

    


if __name__ == '__main__':main()

