def avgpool(output_size=(1,1)):
    """
        CNN backbone이 지난뒤에 한 channel의 feature 평균내고, 나열하여 avgpool을 만든다.
    """

    #보통... batch * 2048 * h* w 
    # 결과 값은... batch * 2048 * 1 * 1 (output이 1이니까)

    def avgpool(input_feature):

        batch,channel,length = len(input_feature),len(input_feature[0]),len(input_feature[0][0])
        output_list = [[[0 for _ in range(output_size[0])] for _ in range(channel)] for _ in range(batch) ]
        for b in range(batch):
            for c in range(channel):
                sum = 0
                for h in range(length):
                    for w in range(length):
                        sum+= input_feature[b][c][h][w]
                output_list[b][c] = sum / (length * length)
        return output_list
    return avgpool
