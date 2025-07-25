from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

#detects if cuda is available for gpu training otherwise uses CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#desired size of output image
imsize=512 if torch.cuda.is_available() else 128
print(imsize)

#scale and transform imported image
loader=transforms.Compose([transforms.Resize(imsize),transforms.ToTensor()])

#helper function
def image_loader(image_name):
    image=Image.open(image_name)
    #fake batch dimensions required to fit network's input dimensions
    image=loader(image).unsqueeze(0)
    return image.to(device,torch.float)

#loading of images
image_directory="/home/mak/Pictures/imgs/"
style_img=image_loader(image_directory + "picasso.jpg")
content_img=image_loader(image_directory + "dancing.jpg")

assert style_img.size()==content_img.size(),"we need to import style and content images of same size"

unloader=transforms.ToPILImage()#reconvert into PIL image
plt.ion()

#helper function to show the tensor as a PIL image
def imshow(tensor,title=None):
    image=tensor.cpu().clone()
    image=image.squeeze(0)
    image=unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)#pause a bit so the plots are updated

plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')

#custom content loss

class ContentLoss(nn.Module):
    def __init__(self,target,):
        super(ContentLoss,self).__init__()
        '''we detach the target content to dynamically compute the 
        gradient: this is a stated value, not a variable.
        Otherwise the forward method of the criterion will throw
        an error.'''
        self.target= target.detach()
    def forward(self,input):
        self.loss=F.mse_loss(input,self.target)
        return input

#this is for style loss
def gram_matrix(input):
    a,b,c,d=input.size() #a=batchsize(=1)
    #b=number of feature maps
    #c,d=dimensions of a f.map(N=c*d)

    features=input.view(a*b,c*d)
    G=torch.mm(features,features.t())#comput the gram product
    return G.div(a*b*c*d)#normalizing values of gram matrix

#same structure as the content loss
class StyleLoss(nn.Module):

    def __init__(self,target_feature):
        super(StyleLoss,self).__init__()
        self.target=gram_matrix(target_feature).detach()

    def forward(self,input):
        G=gram_matrix(input)
        self.loss=F.mse_loss(G, self.target)
        return input
#importing VGG19 model
cnn=models.vgg19(pretrained=True).features.to(device).eval()

#VGG network are normalized with special values for the mean and std
cnn_normalization_mean= torch.tensor([0.485,0.456,0.406]).to(device)
cnn_normalization_std=torch.tensor([0.229,0.224,0.225]).to(device)

#create a module to normalize input image so we can easily put it in nn.Sequential
class Normalization(nn.Module):
    def __init__(self,mean,std):
        super(Normalization,self).__init__()
        self.mean=torch.tensor(mean).view(-1,1,1).to(device)
        self.std=torch.tensor(std).view(-1,1,1).to(device)

    def forward(self,img):
        #normalise img
        return(img-self.mean)/self.std
#here we insert the loss layer at the right stop
#desired depth layers to compute  style/content losses:
content_layers_default=['conv_4']
style_layers_default=['conv_1','conv_2','conv_3','conv_4','conv_5']

def get_style_model_and_losses(cnn, normalization_mean,normalization_std,
                               style_img,content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn=copy.deepcopy(cnn)


    #normaliztion module
    normalization=Normalization(normalization_mean,normalization_std).to(device)

    #just in order to have an iterableaccess to or list of content/style errors
    content_losses=[]
    style_losses=[]

    model=nn.Sequential(normalization)

    i=0#increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer,nn.Conv2d):
            i+=1
            name='conv_{}'.format(i)
        elif isinstance(layer,nn.ReLU):
            name='relu_{}'.format(i)
            layer=nn.ReLU(inplace=False)
        elif isinstance(layer,nn.MaxPool2d):
            name='pool_{}'.format(i)
        elif isinstance(layer,nn.BatchNorm2d):
            name='bn_{}'.format(i)
        else:
            raise RuntimeError("Unrecognized layer: {}".format(layer.__class__.__name__))
        model.add_module(name,layer)

        if name in content_layers:
            #add content loss:
            target=model(content_img).detach()
            content_loss=ContentLoss(target)
            model.add_module("content_loss_{}".format(i),content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            #add style loss
            target_feature=model(style_img).detach()
            style_loss=StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i),style_loss)
            style_losses.append(style_loss)

    #now we trim off layers after the last content and style losses
    for i in range(len(model)-1,-1,-1):
        if isinstance(model[i],ContentLoss) or isinstance(model[i],StyleLoss):
            break
    model=model[:(i+1)]

    return model,style_losses,content_losses

input_img=content_img.clone()
#add the original input image to the figure
plt.figure()
imshow(input_img,title='Input Image')

def get_input_optimizer(input_img):
    optimizer=optim.LBFGS([input_img.requires_grad_()])
    return optimizer
#this will run the nerual style transfer
#it will create image that goes above 1 or below 0, however it will be normalized
def run_style_transfer(cnn,normalization_mean,normalization_std,
                       content_img,style_img,input_img,num_steps=300,
                       style_weight=100000, content_weight=1):
    """run the style transfer"""
    print("building the style transfer model...")
    model,style_losses,content_losses=get_style_model_and_losses(cnn,normalization_mean,normalization_std,style_img,content_img)
    optimizer=get_input_optimizer(input_img)

    print("optimizing..")
    run=[0]
    while run[0]<=num_steps:
        def closure():
            #correct the values of updated input image
            input_img.data.clamp_(0,1)

            optimizer.zero_grad()
            model(input_img)
            style_score=0
            content_score=0

            for sl in style_losses:
                style_score+=sl.loss
            for cl in content_losses:
                content_score+=cl.loss
            
            style_score *=style_weight
            content_score*=content_weight

            loss=style_score+content_score
            loss.backward()

            run[0]+=1
            if run[0]%50==0:
                print("run{}:".format(run))
                print('Style loss:{:4f}Content loss:{:4f}'.format(
                    style_score.item(),content_score.item()
                ))
                print()
            return style_score+content_score
        optimizer.step(closure)

    #a last correction to have the  tensors between 0 and 1
    input_img.data.clamp_(0,1)
    return input_img

output=run_style_transfer(cnn,cnn_normalization_mean, cnn_normalization_std, content_img,style_img,input_img, num_steps=3000)
plt.figure()
imshow(output,title='output image')

plt.ioff()
plt.show()
plt.pause(0.001)