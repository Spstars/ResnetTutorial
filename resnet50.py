import numpy as np
from  typing import Any

#import operators from folder
import operators as nn
import copy

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
        self.layer0_0_conv1 = nn.conv2D(3,64,kernel_size=(7,7),stride=(2,2),padding=3)
        self.layer0_0_bn1 = nn.batchNorm2D(64)
        self.layer0_0_maxpool1 = nn.maxpool(kernel_size=(3,3),stride=(2,2),padding=1)
        #first layer... 나중에 압축하자.

        self.layer1_0_conv1 = nn.conv2D(64,64,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer1_0_bn1 = nn.batchNorm2D(64)
        self.layer1_0_conv2 = nn.conv2D(64,64,kernel_size=(3,3),stride=(1,1),padding=1,bias=False)
        self.layer1_0_bn2 = nn.batchNorm2D(64)
        self.layer1_0_conv3 = nn.conv2D(64,256,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.layer1_0_bn3 = nn.batchNorm2D(256)

        self.layer1_0_downsample_conv1 = nn.conv2D(64,256,kernel_size=(1,1),stride=(1,1),padding=0)
        self.layer1_0_downsample_bn1 = nn.batchNorm2D(256)
             
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
        self.layer2_0_downsample_bn1 = nn.batchNorm2D(512)
        
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
        self.layer3_0_downsample_bn1 = nn.batchNorm2D(1024)

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
        self.layer4_0_downsample_bn1 = nn.batchNorm2D(2048)
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

        for i in range(4):
            for j in range(1,4):
                getattr(self,f"layer2_{i}_conv{j}").init_weight(weight[f'layer2.{i}.conv{j}.weight'])
                getattr(self,f"layer2_{i}_bn{j}").init_mean_weight(weight[f'layer2.{i}.bn{j}.running_mean'],weight[f'layer2.{i}.bn{j}.running_var'],weight[f'layer2.{i}.bn{j}.weight'],weight[f'layer2.{i}.bn{j}.bias'])


        for i in range(6):
            for j in range(1,4):
                getattr(self,f"layer3_{i}_conv{j}").init_weight(weight[f'layer3.{i}.conv{j}.weight'])
                getattr(self,f"layer3_{i}_bn{j}").init_mean_weight(weight[f'layer3.{i}.bn{j}.running_mean'],weight[f'layer3.{i}.bn{j}.running_var'],weight[f'layer3.{i}.bn{j}.weight'],weight[f'layer3.{i}.bn{j}.bias'])

        for i in range(3):
            for j in range(1,4):
                getattr(self,f"layer4_{i}_conv{j}").init_weight(weight[f'layer4.{i}.conv{j}.weight'])
                getattr(self,f"layer4_{i}_bn{j}").init_mean_weight(weight[f'layer4.{i}.bn{j}.running_mean'],weight[f'layer4.{i}.bn{j}.running_var'],weight[f'layer4.{i}.bn{j}.weight'],weight[f'layer4.{i}.bn{j}.bias'])
  
        for i in range(1,5):
            print("i : ",i)
            getattr(self,f"layer{i}_0_downsample_conv1").init_weight(weight[f"layer{i}.0.downsample.0.weight"])
            getattr(self,f"layer{i}_0_downsample_bn1").init_mean_weight(weight[f'layer{i}.0.downsample.1.running_mean'],weight[f'layer{i}.0.downsample.1.running_var'],weight[f'layer{i}.0.downsample.1.weight'],weight[f'layer{i}.0.downsample.1.bias'])

    def add_block(self,x,y,):
        """
            (batches, channels, h,w) 인 x,y를 서로 더한다. 
        """
        batch, channel, height = range(len(x)),range(len(x[0])),range(len(x[0][0]))
        return [[[[x[b][c][h][w] +y[b][c][h][w]  for w in height] for h in height] for c in channel] for b in batch ]
    def deepcopy(self,x):
        batch, channel, height = range(len(x)),range(len(x[0])),range(len(x[0][0]))
        return [[[[x[b][c][h][w]  for w in height] for h in height] for c in channel] for b in batch ]
    
    def residual_block (self,input_feature,layer_num,layer_idx, downsample=False):
        identity  = self.deepcopy(input_feature)
        out = input_feature
        for j in range(1,4):
            out = getattr(self,f"layer{layer_num}_{layer_idx}_conv{j}")(out)
            out = getattr(self,f"layer{layer_num}_{layer_idx}_bn{j}")(out)        
        #identity mapping 이면
        if not downsample:
            return self.relu(self.add_block(identity, out))
        else :
            identity = getattr(self,f"layer{layer_num}_0_downsample_conv1")(identity)
            identity = getattr(self,f"layer{layer_num}_0_downsample_bn1")(identity)
            return self.relu(self.add_block(identity, out))




        # add 함수 구현필요
        

    def forward(self,x):
        print("shape, ",len(x),len(x[0]),len(x[0][0]))
        x = self.layer0_0_maxpool1(self.layer0_0_bn1(self.layer0_0_conv1(x)))

        #true면 convolution block이지만, 함수 한번에 구현하는게 편하니까..
        x = self.residual_block(x,1,0,True)
        x = self.residual_block(x,1,1,False)
        x = self.residual_block(x,1,2,False)

        x = self.residual_block(x,2,0,True)
        x = self.residual_block(x,2,1,False)
        x = self.residual_block(x,2,2,False)
        x = self.residual_block(x,2,3,False)

        x = self.residual_block(x,3,0,True)
        x = self.residual_block(x,3,1,False)
        x = self.residual_block(x,3,2,False)
        x = self.residual_block(x,3,3,False)
        x = self.residual_block(x,3,4,False)
        x = self.residual_block(x,3,5,False)

        x = self.residual_block(x,4,0,True)
        x = self.residual_block(x,4,1,False)
        x = self.residual_block(x,4,2,False)

        x= self.avgpool2d(x)
        return self.linear(x)

        
        # #두개 합 구현...
        # layer1_0 = self.layer1_0_downsample_bn1(self.layer1_0_downsample_conv1(layer0)) + self.relu(self.layer1_0_bn3(self.layer1_0_conv3(self.layer1_0_bn2(self.layer1_0_conv2(self.layer1_0_bn1(self.layer1_0_conv1(layer0)))))))

        # layer2 = self.layer2_0_donwsample_bn1(self.layer2_0_downsample_conv1(layer0)) + \
        #     self.layer2_0_bn3(self.layer2_0_conv3(self.layer2_0_bn2(self.layer2_0_conv2(self.layer1_0_bn1(self.layer2_0_conv1())))))
        
        # layer3 = self.layer3_0_donwsample_bn1(self.layer3_0_downsample_conv1(layer0)) + \
        #     self.layer1_0_bn3(self.layer1_0_conv3(self.layer1_0_bn2(self.layer1_0_conv2(self.layer1_0_bn1(self.layer1_0_conv1(layer2))))))
        
        # layer4 = self.layer4_0_donwsample_bn1(self.layer4_0_downsample_conv1(layer0)) + \
        #     self.layer1_0_bn3(self.layer1_0_conv3(self.layer1_0_bn2(self.layer1_0_conv2(self.layer1_0_bn1(self.layer1_0_conv1(layer3))))))
       
if __name__ =="__main__":
    print(123)