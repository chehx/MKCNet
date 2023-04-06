import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, in_channel, avgpool = True):
        super(VGG16, self).__init__()
        filter = [64, 128, 256, 512, 512]

        # define convolution block in VGG-16
        self.block1 = self.conv_layer(in_channel, filter[0], 1)
        self.block2 = self.conv_layer(filter[0], filter[1], 2)
        self.block3 = self.conv_layer(filter[1], filter[2], 3)
        self.block4 = self.conv_layer(filter[2], filter[3], 4)
        self.block5 = self.conv_layer(filter[3], filter[4], 5)
        
        self.flag_avgpool = avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_size = 512
        
        # apply weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def conv_layer(self, in_channel, out_channel, index):
        if index < 3:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        return conv_block

    def forward(self, x):
        g_block1 = self.block1(x)
        g_block2 = self.block2(g_block1)
        g_block3 = self.block3(g_block2)
        g_block4 = self.block4(g_block3)
        feature = self.block5(g_block4)
        
        if self.flag_avgpool:
            feature = self.avgpool(feature)

        return feature

def vgg16(pretrained=False, **kwargs):
    model = VGG16(**kwargs)
    return model
