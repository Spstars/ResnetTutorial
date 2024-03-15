import numpy as np



def conv2D(input_channel, output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False,dlilation=None):
    """
        Resnet에서 사용되는 conv2D 구현.
        diliation은 하지 않을 것 같고,
        3x3 conv 연산시에 zero-padding이 있다.
        1x1 conv에는 zero-padding이 없다.
        맨처음 conv 연산은 7x7에 stride 2 padding 3이다.
        bias=True가 보이지 않으므로  bias는 쓰지 않을것
    """
    #여기서 shape 크기 비교해서 다르면 error assert
    def elementwise(input_feature,filter,sp,hp):
        """
            여기서는 input_feature,weight,sp,hp를 받아서,
            sp와 hp를 stride와 padding을 통해 원래 위치를 유추해서
            elementwise 연산을 구현
        """
        w,h = kernel_size
        sum_element =0

        for j in range(hp):
            for i in range(sp):
                # 좌표 = x +padding +x*stride와 filter사이에 연산
                for m in range(kernel_size[0]):
                    for n in range(kernel_size[0]):
                        # -1 
                        sum_element =kernel_size[m][n] * input_feature[j+padding+j*stride[0]][i+padding+i*stride[0]]  

        print(213)


    def conv(input_feature):
        #... why not numpy...?
        num_batch =len(input_feature)
        num_channel = len(input_feature[0])
        
        width= len(input_feature[0][0])
        print(num_batch,num_channel)
        #만약 padding >0 이면
        if padding >0:
            for b in range(num_batch):
                for c in range(num_channel):
                    input_feature[b][c] =[[0]*(width+2*padding)] + [[0]*padding+feature+[0]*padding for feature in input_feature[b][c]]+ [[0]*(width+2*padding)]
        
        #다음주에는, weights 가져와서 implementation 해야.
        #일단 커스텀 이니까.
        filter = [[1,2,1],[0,0,0],[-1,-2,-1]]
        output_width = (width -kernel_size[0] + 2*padding[0] +1)/stride[0]


        #np.zeros (batch,channel ,width, height)
        output_feature = [[[[ 0 for _ in range(output_width)] for _ in range(output_width)] for _ in range(output_channel)]  for _ in range(num_batch) ]
        for b in range(num_batch):
            for c in range(num_channel):
                #시작점을 찾아서, 무조건 정사각형 가정하겠음.
                for hp in range(0,output_width):
                    for sp in range(0,output_width):
                        #conv 연산
                        output_feature[b][c][hp][sp] = elementwise(input_feature[b][c],filter,sp,hp)

        return output_feature
    
    return conv
