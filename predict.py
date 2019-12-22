# importing models. 
import argparse
import torch 
import json
import torchvision
from torchvision import datasets, transforms, models 
from torch import nn 
from torch import optim 
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def main():
    # getting arguments 
    args = get_input_args()
    
    
    # loading model
    model = load_checkpoint(args.model_dir)
    
    # device = gpu_check(args.gpu)
    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    # converting model to device type
    model.to(device)
    
    with open(args.cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
    
    # predict image 
    probs, names = predict(args.image_path,args.on_gpu, model,device,cat_to_name,topk=args.top_k)
    

        
    
    # Print name of predicted flower with highest probability
    print("This flower is most likely to be a: {} with a probability of {}% ".format(names[0],np.round(probs[0][0]*100,2)))

 # function for arguments   
def get_input_args():
    parser = argparse.ArgumentParser(description="Train a Neural Network to Predict Images")
    parser.add_argument('--model_dir', type=str, default= "my_checkpoint.pth", help="directory of the model")
    parser.add_argument('--image_path',type=str, default= "flowers/test/1/image_06743.jpg",help="Enter path to image for prediction")
    parser.add_argument('--cat_to_name', type=str, default="cat_to_name.json", help="Enter path to cat_to_name file")
    parser.add_argument('--gpu', action = "store_true",help= "GPU")
    parser.add_argument('--top_k', type=int,  default=5, help="Top K")
    parser.add_argument('--on_gpu', type=str, default= "gpu", help="use gpu or cpu, default is gpu. Make sure gpu is turned on")
    
    args = parser.parse_args()
    
    return args

# function to load model 
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])    
    model.class_to_idx = checkpoint['class_to_idx']
    
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model

# function to load model using PIL image and passing it to the model
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
    
    # return image 
    return image   

# function to predict the image.
def predict(image_path,on_gpu, model,device,cat_to_name,topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
        
    model.to(device)
    # process image
    img = process_image(image_path)
    img.unsqueeze_(0)
    # convert image to float
    img = img.float()
    
    model.eval() 

    # check if gpu exist. if gpu exits  transformed image is converted to cuda, else pass transformed image to model 
    if on_gpu == "gpu":
        with torch.no_grad():
            output = model(img.cuda()) 
    else:
        with torch.no_grad():
            output = model(img)
        
    # calculating probabilites 
    prob = torch.exp(output)
    probs = np.array(prob.topk(topk)[0])
    idxs = np.array(prob.topk(topk)[1])
                
    # convert indicies to classes 
    idxs = np.array(idxs)
    idx_to_class = {value:key for key, value in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in idxs[0]]
        
    #map the class name with collect topk classes 
    names = [] 
    for class_ in classes:
        names.append(cat_to_name[str(class_)])
            
    return probs, names
    
    
    
    
        
if __name__ == '__main__':main()
