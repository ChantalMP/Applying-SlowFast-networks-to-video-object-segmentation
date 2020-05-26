import torch
from efficientnet_pytorch import EfficientNet
from torchvision import models
import geffnet
from time import time


def get_resnet_features(net, x):
    x = net.conv1(x)
    x = net.bn1(x)
    x = net.relu(x)
    x = net.maxpool(x)

    x = net.layer1(x)
    x = net.layer2(x)
    x = net.layer3(x)
    x = net.layer4(x)

    return x


def run_model_config(name):
    x = torch.rand((40, 3, 144, 256), dtype=torch.float32)
    device = torch.device('cuda')
    if name == 'efficientnet_pytorch':
        efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        efficient_net.to(device)  # Storing model takes 525mib
        x = x.to(device)
        start = time()
        efficient_net.extract_features(x)  # takes about 5213 mib
        efficient_net.extract_features(x)  # takes about 5213 mib
        efficient_net.extract_features(x)  # takes about 5213 mib
        efficient_net.extract_features(x)  # takes about 5213 mib
        result = efficient_net.extract_features(x)  # takes about 5213 mib


    elif name == 'geffnet':
        efficient_net = geffnet.efficientnet_b0(pretrained=True)
        efficient_net.to(device)  # 525 mib
        x = x.to(device)
        start = time()
        efficient_net.features(x)  # 5031 mib (little better)
        efficient_net.features(x)  # 5031 mib (little better)
        efficient_net.features(x)  # 5031 mib (little better)
        efficient_net.features(x)  # 5031 mib (little better)
        result = efficient_net.features(x)  # 5031 mib (little better)

    elif name == 'resnet_18':
        resnet = models.resnet18(pretrained=True)
        resnet.to(device)  # Storing takes about 555mib
        x = x.to(device)
        start = time()
        get_resnet_features(resnet, x)  # takes about 1759 mib
        get_resnet_features(resnet, x)  # takes about 1759 mib
        get_resnet_features(resnet, x)  # takes about 1759 mib
        get_resnet_features(resnet, x)  # takes about 1759 mib
        result = get_resnet_features(resnet, x)  # takes about 1759 mib



    elif name == 'resnet_50':
        resnet = models.resnet50(pretrained=True)
        resnet.layer3._modules['0'].conv2.stride = (1, 1)
        resnet.layer3._modules['0'].downsample._modules['0'].stride = (1, 1)
        resnet.layer4._modules['0'].conv2.stride = (1, 1)
        resnet.layer4._modules['0'].downsample._modules['0'].stride = (1, 1)
        # self.resnet.layer3._modules['0'].conv1.stride = (1, 1)  # TODO decide if this fix is actually helping or hurting
        # self.resnet.layer3._modules['0'].downsample._modules['0'].stride = (1, 1)  # Fixing too much pooling
        # self.resnet.layer4._modules['0'].conv1.stride = (1, 1)
        # self.resnet.layer4._modules['0'].downsample._modules['0'].stride = (1, 1)  # Fixing too much pooling
        resnet.to(device)  # Storing takes about 555mib
        x = x.to(device)
        start = time()  # takes about 1759 mib
        get_resnet_features(resnet, x)
        get_resnet_features(resnet, x)
        get_resnet_features(resnet, x)
        get_resnet_features(resnet, x)
        result = get_resnet_features(resnet, x)

    print(f'{name} took {time() - start}')

    a = 1


if __name__ == '__main__':
    run_model_config('resnet_50')
    # TODO resnet both takes less space and is faster than efficient net

    a = 1
