from torchvision import datasets



if __name__== "__main__":
    # resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # training_data= datasets.FashionMNIST(
    #     # 데이터가 저장될것
    #     root="data",
    #     # train할 주제
    #     train=True,
    #     #root에서 정보 못찾으면 다운로드
    #     download=True,
    # )

    training_data= datasets.ImageNet(
    # 데이터가 저장될것
    root="data",
    # train할 주제
    train=False,
    #root에서 정보 못찾으면 다운로드
    download=True)



