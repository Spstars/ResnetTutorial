import numpy as np
from  typing import Any

#import operators from folder
import operators as nn


#torchvison to download datasets
from torchvision import datasets
#torch to load pth file
import torch

class resnet50():
    def __init__(self,weights=True, progress=True, **kwargs :Any ) -> None:
        self.weight =  {}

        #initial layer
        self.relu = nn.relu(inplace=True)

        #initial layer
        self.layer0_0_conv1 = nn.conv2D(3,64,kernel_size=(7,7),stride=(2,2),padding=(3,3))
        self.layer0_0_bn1 = nn.batchNorm2D(64)

        #first layer
        self.layer1_0_conv1 = nn.conv2D(64,64,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer1_0_bn1 = nn.batchNorm2D(64)
        self.layer1_0_conv2 = nn.conv2D(64,64,kernel_size=(3,3),stride=(1,1),padding=1,bias=False)
        self.layer1_0_bn2 = nn.batchNorm2D(64)
        self.layer1_0_conv3 = nn.conv2D(64,256,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer1_0_bn3 = nn.batchNorm2D(256)

        self.layer1_0_downsample_conv1 = nn.conv2D(64,256,kernel_size=(1,1),stride=(1,1),padding=0)
        self.layer1_0_donwsample_bn1 = nn.batchNorm2D(256)

        self.layer1_1_conv1 = nn.conv2D(256,64,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer1_1_bn1 = nn.batchNorm2D(64)
        self.layer1_1_conv2 = nn.conv2D(64,64,kernel_size=(3,3),stride=(1,1),padding=1,bias=False)
        self.layer1_1_bn2 = nn.batchNorm2D(64)
        self.layer1_1_conv3 = nn.conv2D(64,256,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer1_1_bn3 = nn.batchNorm2D(256)


        self.layer1_2_conv1 = nn.conv2D(256,64,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer1_2_bn1 = nn.batchNorm2D(64)
        self.layer1_2_conv2 = nn.conv2D(64,64,kernel_size=(3,3),stride=(1,1),padding=1,bias=False)
        self.layer1_2_bn2 = nn.batchNorm2D(64)
        self.layer1_2_conv3 = nn.conv2D(64,256,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer1_2_bn3 = nn.batchNorm2D(256)

        #second layer 
        self.layer2_0_conv1 = nn.conv2D(256,128,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer2_0_bn1 = nn.batchNorm2D(128)
        self.layer2_0_conv2 = nn.conv2D(128,128,kernel_size=(3,3),stride=(2,2),padding=1,bias=False)
        self.layer2_0_bn2 = nn.batchNorm2D(128)
        self.layer2_0_conv3 = nn.conv2D(128,512,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer2_0_bn3 = nn.batchNorm2D(512)

        self.layer2_0_downsample_conv1 = nn.conv2D(256,512,kernel_size=(1,1),stride=(2,2),padding=0)
        self.layer2_0_donwsample_bn1 = nn.batchNorm2D(512)
        
        self.layer2_1_conv1 = nn.conv2D(512,128,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer2_1_bn1 = nn.batchNorm2D(128)
        self.layer2_1_conv2 = nn.conv2D(128,128,kernel_size=(3,3),stride=(1,1),padding=1,bias=False)
        self.layer2_1_bn2 = nn.batchNorm2D(128)
        self.layer2_1_conv3 = nn.conv2D(128,512,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer2_1_bn3 = nn.batchNorm2D(512)

        self.layer2_2_conv1 = nn.conv2D(512,128,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer2_2_bn1 = nn.batchNorm2D(128)
        self.layer2_2_conv2 = nn.conv2D(128,128,kernel_size=(3,3),stride=(1,1),padding=1,bias=False)
        self.layer2_2_bn2 = nn.batchNorm2D(128)
        self.layer2_2_conv3 = nn.conv2D(128,512,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer2_2_bn3 = nn.batchNorm2D(512)

        self.layer2_3_conv1 = nn.conv2D(512,128,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer2_3_bn1 = nn.batchNorm2D(128)
        self.layer2_3_conv2 = nn.conv2D(128,128,kernel_size=(3,3),stride=(1,1),padding=1,bias=False)
        self.layer2_3_bn2 = nn.batchNorm2D(128)
        self.layer2_3_conv3 = nn.conv2D(128,512,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer2_3_bn3 = nn.batchNorm2D(512)

        #third layer 
        self.layer3_0_conv1 = nn.conv2D(512,256,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer3_0_bn1 = nn.batchNorm2D(128)
        self.layer3_0_conv2 = nn.conv2D(256,256,kernel_size=(3,3),stride=(2,2),padding=1,bias=False)
        self.layer3_0_bn2 = nn.batchNorm2D(128)
        self.layer3_0_conv3 = nn.conv2D(256,1024,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer3_0_bn3 = nn.batchNorm2D(1024)

        self.layer3_1_conv1 = nn.conv2D(1024,256,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer3_1_bn1 = nn.batchNorm2D(256)
        self.layer3_1_conv2 = nn.conv2D(256,256,kernel_size=(3,3),stride=(1,1),padding=1,bias=False)
        self.layer3_1_bn2 = nn.batchNorm2D(256)
        self.layer3_1_conv3 = nn.conv2D(256,1024,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer3_1_bn3 = nn.batchNorm2D(1024)

        self.layer3_2_conv1 = nn.conv2D(1024,256,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer3_2_bn1 = nn.batchNorm2D(256)
        self.layer3_2_conv2 = nn.conv2D(256,256,kernel_size=(3,3),stride=(1,1),padding=1,bias=False)
        self.layer3_2_bn2 = nn.batchNorm2D(256)
        self.layer3_2_conv3 = nn.conv2D(256,1024,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer3_2_bn3 = nn.batchNorm2D(1024)

        self.layer3_3_conv1 = nn.conv2D(1024,256,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer3_3_bn1 = nn.batchNorm2D(256)
        self.layer3_3_conv2 = nn.conv2D(256,256,kernel_size=(3,3),stride=(1,1),padding=1,bias=False)
        self.layer3_3_bn2 = nn.batchNorm2D(256)
        self.layer3_3_conv3 = nn.conv2D(256,1024,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer3_3_bn3 = nn.batchNorm2D(1024)

        self.layer3_4_conv1 = nn.conv2D(1024,256,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer3_4_bn1 = nn.batchNorm2D(256)
        self.layer3_4_conv2 = nn.conv2D(256,256,kernel_size=(3,3),stride=(1,1),padding=1,bias=False)
        self.layer3_4_bn2 = nn.batchNorm2D(256)
        self.layer3_4_conv3 = nn.conv2D(256,1024,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer3_4_bn3 = nn.batchNorm2D(1024)

        self.layer3_5_conv1 = nn.conv2D(1024,256,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer3_5_bn1 = nn.batchNorm2D(256)
        self.layer3_5_conv2 = nn.conv2D(256,256,kernel_size=(3,3),stride=(1,1),padding=1,bias=False)
        self.layer3_5_bn2 = nn.batchNorm2D(256)
        self.layer3_5_conv3 = nn.conv2D(256,1024,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer3_5_bn3 = nn.batchNorm2D(1024)

        self.layer3_0_downsample_conv1 = nn.conv2D(512,1024,kernel_size=(1,1),stride=(2,2),padding=0)
        self.layer3_0_donwsample_bn1 = nn.batchNorm2D(1024)

        #fourth layer
        self.layer4_0_conv1 = nn.conv2D(1024,512,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer4_0_bn1 = nn.batchNorm2D(512)
        self.layer4_0_conv2 = nn.conv2D(512,512,kernel_size=(3,3),stride=(2,2),padding=1,bias=False)
        self.layer4_0_bn2 = nn.batchNorm2D(512)
        self.layer4_0_conv3 = nn.conv2D(512,2048,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer4_0_bn3 = nn.batchNorm2D(2048)    

        self.layer4_1_conv1 = nn.conv2D(2048,512,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer4_1_bn1 = nn.batchNorm2D(512)
        self.layer4_1_conv2 = nn.conv2D(512,512,kernel_size=(3,3),stride=(1,1),padding=1,bias=False)
        self.layer4_1_bn2 = nn.batchNorm2D(512)
        self.layer4_1_conv3 = nn.conv2D(512,2048,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer4_1_bn3 = nn.batchNorm2D(2048)

        self.layer4_2_conv1 = nn.conv2D(2048,512,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer4_2_bn1 = nn.batchNorm2D(512)
        self.layer4_2_conv2 = nn.conv2D(512,512,kernel_size=(3,3),stride=(1,1),padding=1,bias=False)
        self.layer4_2_bn2 = nn.batchNorm2D(512)
        self.layer4_2_conv3 = nn.conv2D(512,2048,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer4_2_bn3 = nn.batchNorm2D(2048)

        self.layer4_0_downsample_conv1 = nn.conv2D(1024,2048,kernel_size=(1,1),stride=(2,2),padding=0)
        self.layer4_0_donwsample_bn1 = nn.batchNorm2D(2048)
        #output

        self.avgpool2d = nn.avgpool2((1,1))
        self.linear= nn.fc(2024,1000)
        
    def __str__(self) -> str:
        return "this is resnet50"
    
    def load_state_dict(self):
        print("load state dict")
        weight = torch.load("./pretrained_weight.pth")
        #'conv1.weight', 'bn1.running_mean', 'bn1.running_var', 'bn1.weight', 'bn1.bias'
        self.layer0_0_conv1.init_weight(weight['conv1.weight'])
        self.layer0_0_bn1.init_mean_weight(weight['bn1.running_mean'],weight['bn1.running_var'],weight['bn1.weight'],weight['bn1.bias'])
        #'layer1.0.conv1.weight', 'layer1.0.bn1.running_mean', 'layer1.0.bn1.running_var', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias', 'layer1.0.conv2.weight',
        for i in range(3):
            for j in range(1,4):
                getattr(self,f"layer1_{i}_conv{j}").init_weight(weight[f'layer1.{i}.conv{j}.weight'])
                getattr(self,f"layer1_{i}_bn{j}").init_mean_weight(weight[f'layer1.{i}.bn{j}.running_mean'],weight[f'layer1.{i}.bn{j}.running_var'],weight[f'layer1.{i}.bn{j}.weight'],weight[f'layer1.{i}.bn{j}.bias'])

        for i in range(5):
            for j in range(1,4):
                getattr(self,f"layer2_{i}_conv{j}").init_weight(weight[f'layer1.{i}.conv{j}.weight'])
                getattr(self,f"layer2_{i}_bn{j}").init_mean_weight(weight[f'layer1.{i}.bn{j}.running_mean'],weight[f'layer1.{i}.bn{j}.running_var'],weight[f'layer1.{i}.bn{j}.weight'],weight[f'layer1.{i}.bn{j}.bias'])


        for i in range(6):
            for j in range(1,4):
                getattr(self,f"layer3_{i}_conv{j}").init_weight(weight[f'layer1.{i}.conv{j}.weight'])
                getattr(self,f"layer3_{i}_bn{j}").init_mean_weight(weight[f'layer1.{i}.bn{j}.running_mean'],weight[f'layer1.{i}.bn{j}.running_var'],weight[f'layer1.{i}.bn{j}.weight'],weight[f'layer1.{i}.bn{j}.bias'])

        for i in range(3):
            for j in range(1,4):
                getattr(self,f"layer4_{i}_conv{j}").init_weight(weight[f'layer1.{i}.conv{j}.weight'])
                getattr(self,f"layer4_{i}_bn{j}").init_mean_weight(weight[f'layer1.{i}.bn{j}.running_mean'],weight[f'layer1.{i}.bn{j}.running_var'],weight[f'layer1.{i}.bn{j}.weight'],weight[f'layer1.{i}.bn{j}.bias'])
  
        for i in range(1,5):
            getattr(self,f"layer{i}_0_downsample_conv1").init_weight(f"layer{i}.0.downsample.0.weight")
            getattr(self,f"layer{i}_0_downsample_bn1").init_mean_weight(weight[f'layer{i}.0.downsample.1.running_mean'],weight[f'layer{i}.0.downsample.1.running_var'],weight[f'layer{i}.0.downsample.1.weight'],weight[f'layer{i}.0.downsample.1.bias'])
resnet= resnet50({123:12})

print(resnet)
