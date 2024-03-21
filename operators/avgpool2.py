# import fc
class avgpool2:
    """
        CNN backbone이 지난뒤에 한 channel의 feature 평균내고, 나열하여 avgpool을 만든다.
    """
    def __init__(self,output_size=(1,1)) :
        self.output_size = output_size

    def avgpool(self,input_feature):

        batch,channel,length = len(input_feature),len(input_feature[0]),len(input_feature[0][0])
        output_list = [[[0 for _ in range(self.output_size[0])] for _ in range(channel)] for _ in range(batch) ]
        for b in range(batch):
            for c in range(channel):
                sum = 0
                for h in range(length):
                    for w in range(length):
                        sum+= input_feature[b][c][h][w]
                output_list[b][c] = sum / (length * length)
        return output_list
    
    def forward(self,input_feature):
        return self.avgpool(input_feature)
    
    def __call__(self,input_feature):
        return self.forward(input_feature)
    
if __name__ =="__main__":
    arr=[[[[ (i+j) *k for i in range(4)] for j in range(4)] for k in range(7)]  for _ in range(1) ]
    avgpool = avgpool2()
    print(arr)
    k= avgpool(arr)
    print(k)
    # fc = fc.fc(4,1)
    # weight = [[0.5,0.5,0.5,0.5]]
    # bias = [0]
    # fc.init_weight(weight,bias)
    # print(k)
    # print(fc(k))

