import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    Depth-wise separable convolution uses less parameters to generate output by convolution.
    :Examples:
        >>> m = DepthwiseSeparableConv(300, 200, 5, dim=1)
        >>> input_tensor = torch.randn(32, 300, 20)
        >>> output = m(input_tensor)
    """

    def __init__(self, in_ch, out_ch, k, dim=1, relu=True):
        """
        :param in_ch: input hidden dimension size
        :param out_ch: output hidden dimension size
        :param k: kernel size
        :param dim: default 1. 1D conv or 2D conv
        """
        super(DepthwiseSeparableConv, self).__init__()
        self.relu = relu
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch,
                                            kernel_size=k, groups=in_ch, padding=k//2)
            self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch,
                                            kernel_size=1, padding=0)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch,
                                            kernel_size=k, groups=in_ch, padding=k//2)
            self.pointwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                                            kernel_size=1, padding=0)
        else:
            raise Exception("Incorrect dimension!")

    def forward(self, x):
        """
        :Input: (N, L_in, D)
        :Output: (N, L_out, D)
        """
        x = x.transpose(1, 2)
        if self.relu:
            out = F.relu(self.pointwise_conv(self.depthwise_conv(x)), inplace=True)
        else:
            out = self.pointwise_conv(self.depthwise_conv(x))
        return out.transpose(1, 2)  # (N, L, D)


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim=1, stride=1, padding=0, relu=True, dropout=0.1):
        """
        :param in_channels: input hidden dimension size
        :param out_channels: output hidden dimension size
        :param kernel_size: kernel size
        :param dim: default 1. 1D conv or 2D conv
        """
        super(ConvRelu, self).__init__()
        self.relu = relu
        self.dropout = dropout
        if dim == 1:
            self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding)
        elif dim == 2:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            raise Exception("Incorrect dimension!")

    def forward(self, x):
        """
        :Input: (batch_num, in_ch, seq_length)
        :Output: (batch_num, out_ch, seq_length)
        """
        x = F.dropout(x, training=self.training, p=self.dropout)
        if self.relu:
            return F.relu(self.conv(x), inplace=True)
        else:
            return self.conv(x)


# deprecated
class DepthwiseSeparableConv1d(nn.Module):

    def __init__(self, n_filters=128, kernel_size=7, padding=3):
        super(DepthwiseSeparableConv1d, self).__init__()

        self.depthwise = nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=padding, groups=n_filters)
        self.separable = nn.Conv1d(n_filters, n_filters, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.separable(x)

        return x
