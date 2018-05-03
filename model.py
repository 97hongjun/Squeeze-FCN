import torch
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import math

class Fire(nn.Module):

	def __init__(self, inplanes, squeeze_planes,
		expand1x1_planes, expand3x3_planes):
		super(Fire, self).__init__()
		self.inplanes = inplanes
		self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
		self.squeeze_activation = nn.ReLU(inplace=True)
		self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
		self.expand1x1_activation = nn.ReLU(inplace=True)
		self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
		self.expand3x3_activation = nn.ReLU(inplace=True)

	def forward(self, x):
		print("Fire Input Shape: ", x.shape)
		x = self.squeeze_activation(self.squeeze(x))
		output = torch.cat([
			self.expand1x1_activation(self.expand1x1(x)),
			self.expand3x3_activation(self.expand3x3(x))
		], 1)
		print("Fire Output Shape: ", output.shape)
		return output

class BackFire(nn.Module):

	def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
		super(BackFire, self).__init__()
		self.inplanes = inplanes
		self.expand3x3 = nn.ConvTranspose2d(expand3x3_planes, squeeze_planes, kernel_size=3)
		self.expand3x3_activation = nn.ReLU(inplace=True)
		self.expand1x1 = nn.ConvTranspose2d(expand1x1_planes, squeeze_planes, kernel_size=1)
		self.expand1x1_activation = nn.ReLU(inplace=True)
		self.squeeze = nn.ConvTranspose2d(squeeze_planes*2, inplanes, kernel_size=1)			
		self.squeeze_activation = nn.ReLU(inplace=True)

	def forward(self, x):
		x1 = self.expand1x1(x)
		match_size = x1.shape[-1]
		x2 = self.expand3x3(x)
		x2 = x2[:,:,1:-1, 1:-1]
		x = torch.cat([
			self.expand1x1_activation(x1),
			self.expand3x3_activation(x2)
		], 1)
		output = self.squeeze_activation(self.squeeze(x))
		# print("output Shape: ", output.shape)
		return output


class Squeeze_FCN(nn.Module):

	def __init__(self, num_classes=1000):
		super(Squeeze_FCN, self).__init__()
		self.num_classes = num_classes

		self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, stride=2)
		self.relu_1 = nn.ReLU(inplace=True)
		self.pool_1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True, return_indices=True)
		
		self.fire_1 = Fire(64, 16, 64, 64)
		self.fire_2 = Fire(128, 16, 64, 64)
		self.pool_2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True, return_indices=True)
		
		self.fire_3 = Fire(128, 32, 128, 128)
		self.fire_4 = Fire(256, 32, 128, 128)
		self.pool_3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True, return_indices=True)

		self.fire_5 = Fire(256, 48, 192, 192)
		self.fire_6 = Fire(384, 48, 192, 192)
		self.fire_7 = Fire(384, 64, 256, 256)
		self.fire_8 = Fire(512, 64, 256, 256)
		self.conv_2 = nn.Conv2d(512, self.num_classes, kernel_size=1)



		self.conv_3 = nn.Conv2d(self.num_classes, 512, kernel_size=1)
		self.bfire_8 = BackFire(512, 64, 512, 512)
		self.bfire_7 = BackFire(384, 64, 512, 512)
		self.bfire_6 = BackFire(384, 48, 384, 384)
		self.bfire_5 = BackFire(256, 48, 384, 384)
		self.unpool_3 = nn.MaxUnpool2d(kernel_size=3, stride=2)

		self.bfire_4 = BackFire(256, 32, 256, 256)
		self.bfire_3 = BackFire(128, 32, 256, 256)
		self.unpool_2 = nn.MaxUnpool2d(kernel_size=3, stride=2)

		self.bfire_2 = BackFire(128, 16, 128, 128)
		self.bfire_1 = BackFire(64, 16, 128, 128)
		self.unpool_1 = nn.MaxUnpool2d(kernel_size=3, stride=2)
		self.relu_2 = nn.ReLU(inplace=True)
		self.conv_4 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def forward(self, x):
		x = self.conv_1(x)
		x_1 = self.relu_1(x)
		x, indices_1 = self.pool_1(x_1)

		x = self.fire_1(x)
		x_2 = self.fire_2(x)
		x, indices_2 = self.pool_2(x_2)
		
		x = self.fire_3(x)
		x_3 = self.fire_4(x)
		x, indices_3 = self.pool_3(x_3)

		x = self.fire_5(x)
		x = self.fire_6(x)
		x = self.fire_7(x)
		x = self.fire_8(x)
		x_4 = self.conv_2(x)

		x = self.conv_3(x_4)
		x = self.bfire_8(x)
		x = self.bfire_7(x)
		x = self.bfire_6(x)
		x_5 = self.bfire_5(x)
		x = self.unpool_3(x_5, indices_3)

		x = self.bfire_4(x)
		x_6 = self.bfire_3(x)
		x = self.unpool_2(x_6, indices_2)

		x = self.bfire_2(x)
		x_7 = self.bfire_1(x)
		x = self.unpool_1(x_7, indices_1)
		x = self.relu_2(x)
		x = self.conv_4(x)
		return x