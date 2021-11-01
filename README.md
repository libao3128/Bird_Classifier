# Bird_Classifier
## Introduction
The project is used for **2021 VRDL HW1** competition on CodaLab. It can be used to classify the birds into 200 specises by providing the image of them.
You can get more information and download the dataset and related file [here](https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07#participate-get_starting_kit).
## Environment
The project is built with **Python 3.9** on **Visual Studio 2019** on **Windows 10**.
You can directly download the project and run it on **Visual Studio 2019** or you can download the code and including following dependencies on your own environment.
## Dependency
1. **CUDA Toolkit 11.5**
    - Is needed for boosting training process by GPU. 
    - See more information here:
        - https://developer.nvidia.com/cuda-downloads
2. **Neural net related**
    - torch==1.10.0+cu113
    - torchvision==0.11.1+cu113
    - torchaudio===0.10.0+cu113
3. **Others**
    - numpy
    - pandas
    - Pillow

You can install all dependencies above **except 'CUDA Toolkit'** by **'pip -r requirment.txt'**. 
For CUDA Tookit, you can install it [here](https://developer.nvidia.com/cuda-downloads).
## Dataset
You can download the dataset on [2021 VRDL HW1 competition on CodaLab](https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07#participate-get_starting_kit).
The dataset includes:
- **training_images.zip**
    - used for training the model
- **testing_images.zip**
    - only used for the competition
- **classes.txt**
    - the name of 200 species of birds
- **training_labels.txt**
    - the file specify the image name in training_images.zip and its species
- **testing_img_order.txt**
    - only used for the competition


You should download them and unzip the .zip file into **testing_images/** and **training_images/**, and put all file mentioned above into the same directory of **'Bird_Classifier.py'**.

## Usage
There is only one .py file named **'Bird_Classifier.py'**. You can run it simply on **Visual Studio 2019** with **'Bird_Classifier.pyproj'** or on your own environment. Once you run the code, there are some input you need to specify.
### 1. load?(y/n)
- This means that whether you want to load the trained weight for the model.
- If 'y':
        1. Notice that there should be a weight file named **'model_weights.pth'** in the same directory of code.
        2. You can get my trained weight here and reprodouce my result.
### Choose mode: (1) train (2) test (3) train and test
- This means which mode do you want to set.
- If **'1'** :
    - The program will start training the model with the images in **'training_img/'**.
- If **'2'** :
    - The program will make a prediction of the images in **'testing_img/'** with the order specified in **'testing_img_order.txt'** and produce the result file named **'answer.txt'**.
- If **'3'** :
    - The program will train first and then predict.
### input init learning rate
Input the initial learning rate of the model and the rate will decrease 10% every epoch by scheduler. The recommended rate for training start over in **0.0001**.
:::warning
Learning rate is important for the training model. Since the program use the pretrained model provided by torchvision, you should **not** set learning rate greater than **0.001**, which will ruin the model.
Also, if you load the trained weight, it is better to set learning rate as small as possible (say 0.000001 for example) due to the reasons I mentioned above.
:::
### how many epoch?
This means that how many epochs do you want for training. Recommended epochs for training start over is 10~15. Notice that it may be a overfitting issue if you train too much.

After you specify the input above, the program will start training or predicting. 

In training process, you would see the training log on console like the image below.
![](https://i.imgur.com/XVlOR0W.png)
The first line shows what epoch it is now and its learning rate now.
The following lines shows the process of the epoch and the total loss, average loss and accuracy of the prediction.
The last line shows the performance of model on validation data.

In testing process, the program will not show any information on console and it will shut down the program once it finish the prediction.
