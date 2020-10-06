## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        
        # Model 1
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        # after one pool layer, this becomes (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.batch1 = nn.BatchNorm2d(32)
        
        # Model 1
        #maxpool layer pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2,2)
        
        # Model 1
        ## output size = (W-F)/S +1 = (110-5)/1 +1 = 106
        # the output tensor will have dimensions: (64, 106, 106)
        # after another pool layer this becomes (64, 53, 53)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.batch2 = nn.BatchNorm2d(64)
        
        # Model 1
        ## output size = (W-F)/S +1 = (53-3)/1 +1 = 51
        # the output tensor will have dimensions: (128, 51, 51)
        # after another pool layer this becomes (128, 25, 25)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.batch3 = nn.BatchNorm2d(128)
        
        # Model 1
        ## output size = (W-F)/S +1 = (25-3)/1 +1 = 23
        # the output tensor will have dimensions: (256, 23, 23)
        # after another pool layer this becomes (256, 11, 11)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.batch4 = nn.BatchNorm2d(256)
        
        
        # Model 1
        ## output size = (W-F)/S +1 = (11-3)/1 +1 = 9
        # the output tensor will have dimensions: (512, 9, 9)
        # after another pool layer this becomes (512, 4, 4)
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.batch5 = nn.BatchNorm2d(512)
        
        # Model 1
        ## output size = (W-F)/S +1 = (4-1)/1 +1 = 4
        # the output tensor will have dimensions: (512, 4, 4)
        # after another pool layer this becomes (512, 2, 2)
        self.conv6 = nn.Conv2d(512, 512, 1)
        self.batch6 = nn.BatchNorm2d(512)
        

        
        # Model 1
        # 512 outputs * the 2*2 filtered/pooled map size
        self.fc1 = nn.Linear(512*2*2,512)
        self.batch_fc1 = nn.BatchNorm1d(512)
        
        # Model 1
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.5)
        self.drop6 = nn.Dropout(p=0.4)
        self.drop7 = nn.Dropout(p=0.4)

        #Model 1
        # finally, create 136 output channels (2 for each of the 68 keypoints(x,y) pairs)
        self.fc2 = nn.Linear(512, 136)
        

        
    def forward(self, x):
        ## Model 1
        x = self.pool(F.relu(self.drop1(self.batch1(self.conv1(x)))))
        x = self.pool(F.relu(self.drop2(self.batch2(self.conv2(x)))))
        x = self.pool(F.relu(self.drop3(self.batch3(self.conv3(x)))))
        x = self.pool(F.relu(self.drop4(self.batch4(self.conv4(x)))))
        x = self.pool(F.relu(self.drop5(self.batch5(self.conv5(x)))))
        x = self.pool(F.relu(self.drop6(self.batch6(self.conv6(x)))))
        # prep for linear layer
        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)        
        # two linear layers with dropout in between
        x = F.relu(self.drop7(self.batch_fc1(self.fc1(x))))
        x = self.fc2(x)             
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
