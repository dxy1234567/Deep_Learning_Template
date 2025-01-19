"""
Author:     Ang Li
Date:       2020-6-14
licensed under the Apache License 2.0
"""

"""
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, layers, kernel_size=3):
        super(ResBlock, self).__init__()
        self.kernel = kernel_size
        self.layers = layers
        self.block = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(layers, layers, self.kernel, 1, 1),
                                   nn.ReLU(),
                                   nn.Conv2d(layers, layers, self.kernel, 1, 1))
    
    def forward(self, input):
        input_res = self.block(input)
        output = input + input_res

        return output


# 稀疏不变卷积
class SparseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SparseConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.mask_conv.weight.data.fill_(1.0)  # 初始化为1
        for param in self.mask_conv.parameters():
            param.requires_grad = False  # 禁止梯度计算

    def forward(self, x):
        # 生成稀疏掩码
        mask = (x > 0).float()
        # 将输入与掩码相乘，得到稀疏输入
        x = x * mask
        # 对稀疏输入进行卷积操作
        x_out = self.conv(x)
        # 对掩码进行卷积操作
        mask_out = self.mask_conv(mask)
        # 确保掩码的卷积结果最小值为1e-5，防止除数为零
        mask_out = torch.clamp(mask_out, min=1e-5)
        # 用卷积结果除以掩码的卷积结果，进行归一化
        x_out = x_out / mask_out
        return x_out


class DepthEncoder(nn.Module):
    def __init__(self, in_layers, layers, filter_size):
        super(DepthEncoder, self).__init__()

        padding = int((filter_size - 1) / 2)

        self.init = nn.Sequential(nn.Conv2d(in_layers, layers, filter_size, stride=1, padding=padding),)
        self.init_ = ResBlock(layers)

        self.enc1 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  )
        self.enc1_ = ResBlock(layers)

        self.enc2 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  )
        self.enc2_ = ResBlock(layers)

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)
                    

    def forward(self, input, scale=2, pre_x2=None, pre_x3=None, pre_x4=None):
        """
        Params:
            input:  当前大小的单张深度图
            pre_x2: 1/8 Input Size
            pre_x3: 1/4 Input Size
            pre_x4: 1/2 Input Size

        Return:
            x0: 1/1 Input Size
            x1: 1/2 Input Size
            x2: 1/4 Input Size

        """
        ### input
        ##### 修改了！！！
        x0 = self.init(input)   # channels = 32
        x0 = self.init_(x0)

        if pre_x4 is not None:
            x0 += F.interpolate(pre_x4, scale_factor=scale, mode='bilinear', align_corners=True)

        x1 = self.enc1(x0)  # 1/2 input size
        x1 = self.enc1_(x1)

        if pre_x3 is not None:  # newly added skip connection
            x1 += F.interpolate(pre_x3, scale_factor=scale, mode='bilinear', align_corners=True)

        x2 = self.enc2(x1)  # 1/4 input size
        x2 = self.enc2_(x2)

        if pre_x2 is not None:  # newly added skip connection
            x2 += F.interpolate(pre_x2, scale_factor=scale, mode='bilinear', align_corners=True)
            
        # No.1 Hourglass    
        # X0:   1/4  Res
        # x1:   1/8  Res
        # x2:   1/16 Res
        return x0, x1, x2   # channels = 32


class RGBEncoder(nn.Module):
    def __init__(self, in_layers, layers, filter_size):
        super(RGBEncoder, self).__init__()

        padding = int((filter_size - 1) / 2)

        self.rgb_conv = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.rgb_conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)



        # # 初始变换，分辨率不变
        self.init = nn.Sequential(nn.Conv2d(16, layers, filter_size, stride=1, padding=padding))
        self.init_ = ResBlock(layers)
        
        
        # 分辨率缩小一半(1/2)
        self.enc1 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  )
        self.enc1_ = ResBlock(layers)

        # 分辨率缩小一半(1/4)
        self.enc2 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  )
        self.enc2_ = ResBlock(layers)

        # 分辨率缩小一半(1/8)
        self.enc3 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  )
        self.enc3_ = ResBlock(layers)

        # 分辨率缩小一半(1/16)
        self.enc4 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  )
        self.enc4_ = ResBlock(layers)
        
        # 分辨率缩小一半(1/32) ###add
        self.enc5 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  )
        self.enc5_ = ResBlock(layers)

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)


    def forward(self, input, input_d, scale=2, pre_x=None):
        ### input

        # 通道数扩大到16
        rgb = self.rgb_conv(input)
        rgb = self.rgb_conv1(rgb)

        x0 = self.init(rgb)
        if pre_x is not None:
            x0 = x0 + F.interpolate(pre_x, scale_factor=scale, mode='bilinear', align_corners=True)

        x1 = self.enc1(x0)  # 1/2 input size

        x2 = self.enc2(x1)  # 1/4 input size

        x3 = self.enc3(x2)  # 1/8 input size

        x4 = self.enc4(x3)  # 1/16 input size
        
        x5 = self.enc5(x4)  # 1/32 input size

        return x0, x1, x2, x3, x4, x5
    
    
class DepthDecoder(nn.Module):
    """
    正常
    """
    def __init__(self, layers, filter_size):
        super(DepthDecoder, self).__init__()
        padding = int((filter_size - 1) / 2)
        

        self.dec2 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers // 2, layers // 2, filter_size, stride=2, padding=padding,
                                                     output_padding=padding),
                                  )
        self.dec2_ = ResBlock(layers // 2)
        self.shortcut2 = nn.Conv2d(layers, layers // 2, 3, 1, 1)
 

        self.dec1 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers // 2, layers // 2, filter_size, stride=2, padding=padding,
                                                     output_padding=padding),
                                  )
        self.dec1_ = ResBlock(layers // 2)


        self.prdct = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(layers // 2, layers // 2, filter_size, stride=1, padding=padding),
                                   nn.ReLU(),
                                   nn.Conv2d(layers // 2, 1, filter_size, stride=1, padding=padding),
                                   )
        
    
        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, pre_dx, pre_cx):

        """
        Params：
            两个参数分别为3、4个元素构成的元组，序号越小分辨率越大，pre_dx[0]为当前沙漏网络分辨率的大小
            pre_dx: 前半个沙漏网络（编码器）的输出
            pre_cx: RGB编码器的输出
        """

        x2 = pre_dx[2] + pre_cx[2]
        x1 = pre_dx[1] + pre_cx[1]
        x0 = pre_dx[0] + pre_cx[0]

        x3 = self.dec2(x2)          # 1/4 -> 1/2 
        x3 = self.dec2_(x3)


        x4 = self.dec1(x1 + x3)     # 1/1 input size
        x4 = self.dec1_(x4)

        ### prediction
        output_d = self.prdct(x4 + x0)

        return x2, x3, x4, output_d

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()

        denc_layers = 32
        cenc_layers = 32
        ddcd_layers = denc_layers + cenc_layers



        self.rgb_encoder = RGBEncoder(16, cenc_layers, 3)

        self.depth_encoder1 = DepthEncoder(1, denc_layers, 3)
        self.depth_decoder1 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder2 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder2 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder3 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder3 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder4 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder4 = DepthDecoder(ddcd_layers, 3)


    def forward(self, input_d: torch.Tensor, input_rgb):
        C = (input_d > 0).float()

        enc_c = self.rgb_encoder(input_rgb, input_d)

        ## for the 1/8 res(scale 3)
        input_d18 = F.avg_pool2d(input_d, 8, 8) / (F.avg_pool2d(C, 8, 8) + 0.0001)
        enc_d18 = self.depth_encoder1(input_d18, 2)

        dcd_d18 = self.depth_decoder1(enc_d18, enc_c[3:6])

        ## for the 1/4 res
        input_d14 = F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
        predict_d14 = F.interpolate(dcd_d18[3], scale_factor=2, mode='bilinear', align_corners=True)
        input_14 = torch.cat((input_d14, predict_d14), 1)

        enc_d14 = self.depth_encoder2(input_14, 2, dcd_d18[0], dcd_d18[1], dcd_d18[2])
        dcd_d14 = self.depth_decoder2(enc_d14, enc_c[2:5])

        ## for the 1/2 res
        input_d12 = F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
        predict_d12 = F.interpolate(predict_d14 + dcd_d14[3], scale_factor=2, mode='bilinear', align_corners=True)
        input_12 = torch.cat((input_d12, predict_d12), 1)

        enc_d12 = self.depth_encoder3(input_12, 2, dcd_d14[0], dcd_d14[1], dcd_d14[2])
        dcd_d12 = self.depth_decoder3(enc_d12, enc_c[1:4])

        ## for the 1/1 res
        predict_d11 = F.interpolate(predict_d12 + dcd_d12[3], scale_factor=2, mode='bilinear', align_corners=True)
        input_11 = torch.cat((input_d, predict_d11), 1)

        enc_d11 = self.depth_encoder4(input_11, 2, dcd_d12[0], dcd_d12[1], dcd_d12[2])
        dcd_d11 = self.depth_decoder4(enc_d11, enc_c[0:3])

        ## output
        output_d11 = dcd_d11[3] + predict_d11
        output_d12 = predict_d11
        output_d14 = F.interpolate(predict_d12, scale_factor=2, mode='bilinear', align_corners=True)
        output_d18 = F.interpolate(dcd_d18[3], scale_factor=8, mode='bilinear', align_corners=True)


        return output_d11, output_d12, output_d14, output_d18, 







