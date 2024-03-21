import torch
class conv2D():
    """
        Resnet에서 사용되는 conv2D 구현.
        diliation은 하지 않을 것 같고,
        3x3 conv 연산시에 zero-padding이 있다.
        1x1 conv에는 zero-padding이 없다.
        맨처음 conv 연산은 7x7에 stride 2 padding 3이다.
        bias=True가 보이지 않으므로  bias는 쓰지 않을것
    """

    def __init__(self,input_channel, output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False,dlilation=None):
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.dliation= dlilation

        self.weight = []
    


    #곱 구현
    def elementwise(self, input_feature,output_channel,output_width):
        """
            여기서는 input_feature,weight,sp,hp를 받아서,
            sp와 hp를 stride와 padding을 통해 원래 위치를 유추해서
            elementwise 연산을 구현
        """
    
        sum_element = [[0 for _ in range(output_width)] for _ in range(output_width)]
        cur_weight = self.weight[output_channel]

        for m in range(self.kernel_size[1]):
            for n in range(self.kernel_size[0]):
                input_row = self.padding + m - self.kernel_size[0] // 2
                input_col = self.padding + n - self.kernel_size[1] // 2
                for hp in range(output_width):
                    for sp in range(output_width):
                        for num_c in range(len(input_feature)):
                            sum_element[hp][sp] += (cur_weight[num_c][m][n] *
                                input_feature[num_c][input_row + hp * self.stride[0]][input_col + sp * self.stride[1]])

       
       
        return sum_element


    def conv(self,input_feature):
        num_batch =len(input_feature)
        num_channel = len(input_feature[0])
        width= len(input_feature[0][0])
       
        padding = self.padding
        kernel_size = self.kernel_size
        stride = self.stride
        output_channel =self.output_channel
        print("cnn : batch, channel, width,kernel :",num_batch,num_channel,width,kernel_size)


        #만약 padding >0 이면
        if self.padding >0:
            for b in range(num_batch):
                for c in range(num_channel):
                    input_feature[b][c] =[[0]*(width+2*padding)]*padding \
                    + [[0]*padding+feature+[0]*padding for feature in input_feature[b][c]]+ [[0]*(width+2*padding)]*padding 
        
        
        output_width = int((width - kernel_size[0] + 2*padding +1)/stride[0] + 0.5)

        
        output_feature = [[[[ 0 for _ in range(output_width)] for _ in range(output_width)] for _ in range(output_channel)]  for _ in range(num_batch) ]
        for b in range(num_batch):
            for c in range(output_channel):
                    output_feature[b][c] = self.elementwise(input_feature[b],c,output_width)


        return output_feature
    
    def init_weight(self,weight,bias=0):
        self.weight=weight.tolist()
 

    def __call__(self, input_feature) :
        return self.conv(input_feature)
    

if __name__ == "__main__":
    conv = conv2D(3,10,kernel_size=(7,7),stride=(2,2),padding=3)
    conv2 = torch.nn.Conv2d(3,10,kernel_size=(7,7),stride=(2,2),padding=3,bias=False)
    
    X_t = torch.randn(size=(1,3,56,56))
    X = X_t.tolist()
    W = torch.randn(size=(3,10,3,3))
    W_t = torch.nn.Parameter(W)
    W_l = W
    conv.init_weight(W_l)
    conv2.weight=W_t
    y2= conv2(X_t)    

    
    print(y2[0][0][0][:])
    y= conv(X)

    print(y[0][0][0][:])



    # conv = conv2D(256,128,kernel_size=(1,1),stride=(1,1),padding=0)
    # X2= torch.randn(size = (1,256,28,28)).tolist()
    # print(len(X2),len(X2[0]),len(X2[0][0]))
    # y= conv(X2)
    # print(len(y),len(y[0]),len(y[0][0]))