import torch


x_dim, y_dim = 3, 5
xx_ones = torch.ones([1,1,1,y_dim], dtype=torch.int32)
yy_ones = torch.ones([1,1,1,x_dim], dtype=torch.int32)

xx_range = torch.arange(x_dim, dtype=torch.int32)
yy_range = torch.arange(y_dim, dtype=torch.int32)
print(xx_ones, xx_ones.shape)
print(yy_ones, yy_ones.shape)
print(xx_range)
print(yy_range)
xx_range = xx_range[None, None, :, None]
yy_range = yy_range[None, None, :, None]

print(xx_range, xx_range.shape)
print(yy_range, yy_range.shape)

xx_channel = torch.matmul(xx_range, xx_ones)
yy_channel = torch.matmul(yy_range, yy_ones)
print(xx_channel, xx_channel.shape)
print(yy_channel, yy_channel.shape)

yy_channel = yy_channel.permute(0, 1, 3, 2)
print(yy_channel, yy_channel.shape)

xx_channel = xx_channel.repeat(10, 1, 1, 1)
print(xx_channel)