
import numpy as np
import pandas as pd
import torchvision.models as models
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from PIL import Image
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader
import os

batch_size = 5


class TrainData():
    """
    Train data with the required function for data loader in pytorch 
    """
    def __init__(
            self, class_file, img_file,
            transform=None, target_transform=None):
        self.get_labels()
        self.labels = pd.read_table(
            class_file, sep=" ",
            names=['img','label'],
            header=None
            )
        self.img_file = img_file
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        img_path = os.path.join(self.img_file, self.labels.iloc[idx, 0])
        image = Image.open(img_path)

        label = self.labels.iloc[idx,1]
        label_index = int((self.label_list[self.label_list['label'] == label]).index.values)

        if self.transform is not None:
            image = self.transform(image)
            
        return image.to(device),torch.tensor(label_index).to(device)

    def get_labels(self):
        self.label_list =  pd.read_table('classes.txt',names=['label'])
        

class TestData():
    """
    Test data with the required function for data loader in pytorch 
    """
    def __init__(
            self, class_file, img_file,
            transform=None, target_transform=None):
        self.get_labels()
        self.labels = pd.read_table(
            class_file, sep=" ",
            names = ['img'], header = None
            )
        self.img_file = img_file
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        img_path = os.path.join(self.img_file, self.labels.iloc[idx, 0])
        image = Image.open(img_path)
        
        if self.transform is not None:
            image = self.transform(image)
     
        return image.to(device),self.labels.iloc[idx, 0]

    def get_labels(self):
        self.label_list =  pd.read_table('classes.txt',names = ['label'])


class Attention(torch.nn.Module):
    """
    Attention block for CNN model.
    """
    def __init__(
            self, in_channels, out_channels, 
            kernel_size, padding):
        super(Attention, self).__init__()
        self.conv_depth = torch.nn.Conv2d(
            in_channels, out_channels, 
            kernel_size, padding = padding, 
            groups = in_channels
            )
        self.conv_point = torch.nn.Conv2d(
            out_channels, out_channels, 
            kernel_size = (1, 1)
            )
        self.bn = torch.nn.BatchNorm2d(
            out_channels, eps=1e-5, 
            momentum = 0.1, affine = True, 
            track_running_stats = True
            )
        self.activation = torch.nn.Tanh()

    def forward(self, inputs):
        x, output_size = inputs
        x = nn.functional.adaptive_max_pool2d(x, output_size=output_size)
        x = self.conv_depth(x)
        x = self.conv_point(x)
        x = self.bn(x)
        x = self.activation(x) + 1.0
        return x


class ResNet50Attention(torch.nn.Module):
    """
    Attention-enhanced ResNet-50 model.
    """
    weights_loader = staticmethod(models.resnet152)

    def __init__(
            self, num_classes=200, 
            pretrained=True, use_attention=True):
        super(ResNet50Attention, self).__init__()
        net = self.weights_loader(pretrained=pretrained)

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.use_attention = use_attention


        net.fc = torch.nn.Linear(
            in_features=net.fc.in_features,
            out_features=num_classes,
            bias=net.fc.bias is not None
            )
        self.net = net
        
        if self.use_attention:
            self.att1 = Attention(
                in_channels = 64,
                out_channels = 256,
                kernel_size = (3, 5),
                padding = (1, 2)
                )
            self.att2 = Attention(
                in_channels = 256,
                out_channels = 512,
                kernel_size = (5, 3),
                padding = (2, 1)
                )
            self.att3 = Attention(
                in_channels = 512, 
                out_channels = 1024, 
                kernel_size = (3, 5), 
                padding = (1, 2)
                )
            self.att4 = Attention(
                in_channels=1024, 
                out_channels=2048, 
                kernel_size=(5, 3), 
                padding=(2, 1)
                )

            if pretrained:
                self.att1.bn.weight.data.zero_()
                self.att1.bn.bias.data.zero_()
                self.att2.bn.weight.data.zero_()
                self.att2.bn.bias.data.zero_()
                self.att3.bn.weight.data.zero_()
                self.att3.bn.bias.data.zero_()
                self.att4.bn.weight.data.zero_()
                self.att4.bn.bias.data.zero_()

    def _forward(self, x):
        return self.net(x)
    
    def _forward_att(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x_a = torch.clone(x)
        x = self.net.layer1(x)
        x = x.mul(self.att1((x_a,x.shape[-2:])))

        x_a = x.clone()
        x = self.net.layer2(x)
        x = x * self.att2((x_a, x.shape[-2:]))

        x_a = x.clone()
        x = self.net.layer3(x)
        x = x * self.att3((x_a, x.shape[-2:]))

        x_a = x.clone()
        x = self.net.layer4(x)
        x = x * self.att4((x_a, x.shape[-2:]))

        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net.fc(x)

        return x
    
    def forward(self, x):
        if self.use_attention :
            return self._forward_att(x) 
        else :
            return self._forward(x)
        


def pad(img, size_max=500):
    """
    Pads images to the specified size (height x width). 
    """
    pad_height = max(0, size_max - img.height)
    pad_width = max(0, size_max - img.width)
    
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    
    return transforms.functional.pad(
        img,
        (pad_left, pad_top, pad_right, pad_bottom),
        fill=tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406))))


def train_loop(dataloader, model, loss_fn, optimizer):

    def print_info(loss,train_loss,num_of_img,correct):
        loss, current = loss.item(), batch * len(X)+batch_size
        
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        train_loss /=  num_of_img
        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            train_loss, correct, num_of_img,
            100. * correct / num_of_img))
            
        

    model.train()

    size = len(dataloader.dataset)
    train_loss = 0
    correct = 0
    num_of_img = 0
    

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss.
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        train_loss += loss

        # Get the prediction and add one to correct if the prediction 
        # is correct.
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(y.view_as(pred)).sum().item() 
        num_of_img += batch_size

        # Backpropagation
        loss.backward()
        optimizer.step()

        
        
        if batch % 100 == 99:
            # Save the model weight and print training information
            # for every 100 loop.
            torch.save(model.state_dict(), 'model_weights.pth')
            print_info(loss,train_loss)

            train_loss = 0
            correct = 0
            num_of_img = 0

    if batch % 100 != 99:
        torch.save(model.state_dict(), 'model_weights.pth')
        print_info(loss,train_loss,num_of_img,correct)
        

def val_loop(model, test_loader):
    # The loop for the model validation

    model.eval()  # Set the model to evaluate mode

    test_loss = 0
    correct = 0

    with torch.no_grad(): 
        # disable gradient calculation for efficiency
        for X, y in test_loader:
            
            output = model(X)  # Get model prediction output

        
            test_loss += loss_fn(output, y)  # Add current loss to total loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the prediction

            # Add one to correct if the prediction is correct
            correct += pred.eq(y.view_as(pred)).sum().item()  

    test_loss /= len(test_loader.dataset)  # Calculate average loss

    # Print testing information
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    model.train()  # Set the model to training mode


def test(model,test_loader):
    model.eval()

    file = open('answer.txt','w')
    label_list = pd.read_table('classes.txt',names=['label'])
   
    with torch.no_grad():
        for data, img_name in test_loader:
            # Prediction
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            
            for idx,name in enumerate(img_name):
                file.write(name)
                file.write(' ')
                file.write(label_list.loc[int(pred[idx]),'label'])
                file.write('\n')
            
    file.close()
    
    model.train()


# Initialize the model and transform used in model
train_transform = transforms.Compose([
   transforms.Lambda(pad),
   transforms.RandomOrder([
       transforms.RandomCrop((375, 375)),
       transforms.RandomHorizontalFlip(),
       transforms.RandomVerticalFlip()
       ]),
   transforms.ToTensor(),
   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ])

val_transform = transforms.Compose([
   transforms.Lambda(pad),
   transforms.CenterCrop((300, 300)),
   transforms.ToTensor(),
   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = val_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ResNet50Attention()
model.to(device)

# Ask user if they want to load the weight trained before
print("load?(y/n)")
load=input()
if load=='y':
    try:
        model.load_state_dict(torch.load('model_weights.pth'))
        print("load weight")
    except:
        print("load failed")

# Get the mode from user's input
print("Choose mode: (1) train (2) test (3) train and test")
mode = int(input())
while not (mode>=1 and mode<=3):
    print("wrong input")
    mode = input()

# Training process
if mode==1 or mode==3:
    # Ask user the initial learning rate
    print("input init learning rate")
    learning_rate=float(input())

    # Ask user how much epoch they want to train
    print("how many epoch?")
    epoch = int(input())

    # Load data and set optimizer, scheduler and loss function
    Data = TrainData("training_labels.txt",'training_images',train_transform)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
    loss_fn = nn.CrossEntropyLoss()

    # Set random seed to ensure the training process reproducible
    torch.manual_seed(0)

    for i in range(epoch):
        # Loop for 'epoch' times to train the model

        # Split the data into training data and validation data randomly
        train_dataset,validation_dataset = random_split(Data,[2500,500])
        train_dataloader = DataLoader( 
            train_dataset, 
            batch_size = batch_size, 
            shuffle = True)
        val_dataloader = DataLoader(
            validation_dataset,
            batch_size = batch_size,
            shuffle = True)         

        # Go to train loop function to train the model
        train_loop(train_dataloader,model,loss_fn,optimizer)
        # Go to validation loop function to validate the model
        val_loop(model,val_dataloader)
        
        # Decrease learning rate by scheduler and print current learning rate
        scheduler.step()
        print(i, scheduler.get_last_lr()[0])

# Testing Process     
if mode==2 or mode==3:
    file_name = 'testing_img_order.txt'
    folder_name = 'testing_images'

    # Load test data from file
    test_dataset = TestData(file_name, folder_name, test_transform)
    test_dataloader = DataLoader(
        test_datase, 
        batch_size = batch_size, 
        shuffle = False)     

    # Go to test function to test and output predict result 
    test(model,test_dataloader)







