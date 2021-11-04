
import numpy as np
import pandas as pd
import torchvision.models as models
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from PIL import Image
import os
import matplotlib.pyplot as plt



from torch.utils.data import DataLoader


batch_size = 5
Mixup = True
Progressive_Resizing = True
image_size = 375
log_file = open('log_info.txt','a')

class TrainData():
    """
    Train data with the required function for data loader in pytorch 
    """
    def __init__(
            self, class_file, img_file,
            transform=None, target_transform=None,
            is_train=False):
        self.get_labels()
        self.labels = pd.read_table(
            class_file, sep=" ",
            names=['img','label'],
            header=None
            )
        self.img_file = img_file
        self.transform = transform
        self.target_transform = target_transform
        self.is_train=is_train

    def __len__(self):
        return len(self.labels)

    def set_is_train(self,is_train):
        self.is_train = is_train

    def __getitem__(self,idx):
        img_path = os.path.join(self.img_file, self.labels.iloc[idx, 0])
        image = Image.open(img_path)

        label_name = self.labels.iloc[idx,1]
        label_index = int((self.label_list[self.label_list['label'] == label_name]).index.values)

        label = torch.zeros(200)
        label[label_index] = 1.

        if self.transform is not None:
            image = self.transform(image)

        #print(self.is_train)
        # If data is for training, perform mixup, only perform mixup roughly on 1 for every 5 images
        if self.is_train and idx > 0 and idx%5 == 0:
            #print('mix')
            # Choose another image/label randomly

            mixup_idx = np.random.randint(0, len(self.labels)-1)
            img_path = os.path.join(self.img_file, self.labels.iloc[mixup_idx, 0])
            mixup_image  = Image.open(img_path)
            mixup_label_name = self.labels.iloc[idx,1]
            mixup_label_index = int((self.label_list[self.label_list['label'] == mixup_label_name]).index.values)
            mixup_label = torch.zeros(200)
            mixup_label[mixup_label_index] = 1.
            
            if self.transform is not None:
                mixup_image = self.transform(mixup_image)


            # Select a random number from the given beta distribution
            # Mixup the images accordingly
            alpha = 0.2
            lam = np.random.beta(alpha, alpha)
            image = lam * image + (1 - lam) * mixup_image
            label = lam * label + (1 - lam) * mixup_label


        #plt.imshow(  image.permute(1, 2, 0)  )
        #plt.show()
            
        return image.to(device),label.to(device)

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
    '''
    The loop for the model training
    '''

    def print_info(loss,train_loss,num_of_img,correct):
        loss, current = loss.item(), batch * len(X)+batch_size
        
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        log_file.write(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]\n")
        train_loss /=  num_of_img
        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            train_loss, correct, num_of_img,
            100. * correct / num_of_img))
        log_file.write('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
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
        pred = torch.argmax(output, dim=1)
        correct += (pred == torch.argmax(y, dim=1)).float().sum() 
        num_of_img += batch_size

        # Backpropagation
        loss.backward()
        optimizer.step()

        
        
        if batch % 100 == 99:
            # Save the model weight and print training information
            # for every 100 loop.
            torch.save(model.state_dict(), 'model_weights.pth')
            print_info(loss,train_loss,num_of_img,correct)

            train_loss = 0
            correct = 0
            num_of_img = 0

    if batch % 100 != 99:
        torch.save(model.state_dict(), 'model_weights.pth')
        print_info(loss,train_loss,num_of_img,correct)
        

def val_loop(model, test_loader, is_test=False):
    '''
    The loop for the model validation
    '''

    model.eval()  # Set the model to evaluate mode

    test_loss = 0
    correct = 0

    with torch.no_grad(): 
        # disable gradient calculation for efficiency
        for X, y in test_loader:
            
            output = model(X)  # Get model prediction output

        
            test_loss += loss_fn(output, y)  # Add current loss to total loss
            pred = torch.argmax(output, dim=1)  # Get the prediction

            # Add one to correct if the prediction is correct
            correct += (pred == torch.argmax(y, dim=1)).float().sum()

    test_loss /= len(test_loader.dataset)  # Calculate average loss

    # Print testing information
    if is_test:
        print('Test set:')
        log_file.write('Test set:\n')
    else:
        print('Validation set:')
        log_file.write('Validation set:\n')
    print(' Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    log_file.write(' Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    model.train()  # Set the model to training mode


def test(model,test_loader):
    '''
    The loop for the model testing
    '''

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


def log_info():
    
    log_file.write('batch_size:'+str(batch_size)+'\n')
    log_file.write('MixUp:'+str(Mixup)+'\n')
    log_file.write('Progressive_Resizing:'+str(Progressive_Resizing)+'\n')
    if Progressive_Resizing:
        log_file.write('img_size:'+str(image_size)+'\n')

log_info()

# Initialize the model and transform used in model
train_transform = transforms.Compose([
   transforms.Lambda(pad),
   transforms.Resize([image_size], antialias=True),
   
   transforms.RandomOrder([
       transforms.RandomCrop((int(image_size),int(image_size))),
       transforms.RandomHorizontalFlip(),
       transforms.RandomVerticalFlip()
   ]),
   
   transforms.ToTensor(),
   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ])

test_transform = transforms.Compose([
   
   transforms.Lambda(pad),
   transforms.CenterCrop((375,375)),
   transforms.ToTensor(),
   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet152(pretrained=True)
model.fc = torch.nn.Linear(
            in_features=model.fc.in_features,
            out_features=200
            )
mode = ResNet50Attention()
model.to(device)
# Ask user if they want to load the weight trained before
print("load?(y/n)")
load=input()
if load=='y':
    #try:
        model.load_state_dict(torch.load('model_weights.pth'))
        print("load weight")
    #except:
     #   print("load failed")

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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
    loss_fn = nn.CrossEntropyLoss()

    # Set random seed to ensure the training process reproducible
    torch.manual_seed(0)
    Data = TrainData("training_labels.txt",'training_images',test_transform)
    _,test_dataset = random_split(Data,[2700,300])

    Data = TrainData("training_labels.txt",'training_images',train_transform)
    Data.is_train = Mixup
    torch.manual_seed(0)
    Data,_ = random_split(Data,[2700,300])
    test_dataloader = DataLoader(
            test_dataset,
            batch_size = batch_size, 
            shuffle = True)  
    
   
    
    for i in range(epoch):
        # Loop for 'epoch' times to train the model

        epoch_info = 'epoch:'+ str(i)+ ' learning rate:'+ str(scheduler.get_last_lr()[0])
        print(epoch_info)  
        log_file.write(epoch_info+'\n')

        # Split the data into training data and validation data randomly
        train_dataset,validation_dataset = random_split(Data,[2400,300])
        
        
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
        # Go to validation loop function to test the model
        val_loop(model,test_dataloader,True)
        print()
        log_file.write('\n')
        
        scheduler.step() # Decrease learning rate by scheduler 
        

# Testing Process     
if mode==2 or mode==3:
    file_name = 'testing_img_order.txt'
    folder_name = 'testing_images'

    # Load test data from file
    test_dataset = TestData(file_name, folder_name, test_transform)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size = batch_size, 
        shuffle = False)     

    # Go to test function to test and output predict result 
    test(model,test_dataloader)

log_file.close()





