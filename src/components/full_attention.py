import torch.nn as nn
import torch


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class full_attention(nn.Module):
    def __init__(self, dim, reduction=16):
        super(full_attention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.dim = dim 

        mip = max(8, dim // reduction)

        self.conv1 = nn.Conv2d(dim, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, dim, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, dim, kernel_size=1, stride=1, padding=0)

        self.squeeze = nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, padding=0)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        identity = x
        
        _, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        se = torch.cat([identity, out], dim=1)
        se = self.squeeze(se)
        b, c, _, _ = se.size()
        tmp = self.avg_pool(se).view(b, c)
        tmp = self.fc(tmp).view(b, c, 1, 1)
        res = se * tmp.expand_as(se)
        return res
    

if __name__ == '__main__':
    model = full_attention(256, 256)

    x = torch.randn(1, 256, 64, 64)
    output = model(x)
    print(output.shape)