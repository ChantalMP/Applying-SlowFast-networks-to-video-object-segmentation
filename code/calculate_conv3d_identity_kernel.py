import torch
from torch import nn
from torch.optim import Adam


def identity_kernel_for_conv3d(input_channel_size, identity_index, time_size=8, kernel_size=(3, 3, 3), padding=(0, 1, 1)):
    '''
    :param identity_index: As we are merging over time, only one of the original inputs can remain the same, this index is needed for that
    :return:
    '''
    conv_layer = nn.Conv3d(input_channel_size, input_channel_size, kernel_size=kernel_size, padding=padding)
    optimizer = Adam(conv_layer.parameters())
    loss_fn = nn.MSELoss()

    for i in range(3000):
        x = torch.randn(32, input_channel_size, time_size, 3, 3)
        output = conv_layer(x)
        loss = loss_fn(output[:, :, 0, :, :], x[:, :, identity_index, :, :])
        print(f'Loss: {loss.item():.5f}')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    identity_kernel_weight = (conv_layer.weight >= 0.5).float()
    return identity_kernel_weight


if __name__ == '__main__':
    # Uses pytorch optimization to find the identity kernel for any 3D conv
    identity_kernel_weight = identity_kernel_for_conv3d(input_channel_size=3, identity_index=1)
    torch.save(identity_kernel_weight, 'data/identity_kernel_weight.pth')
