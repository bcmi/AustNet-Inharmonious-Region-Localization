import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def conv3x3(in_planes, out_planes, stride=1):
	"3x3 convolution with padding"
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		if downsample is None and inplanes != planes:
			self.downsample = nn.Conv2d(inplanes, planes, 1,1,0)
		else:
			self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out
	
	
class conv_block(nn.Module):
	"""
	Convolution Block 
	"""
	def __init__(self, in_ch, out_ch):
		super(conv_block, self).__init__()
		
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True))

	def forward(self, x):

		x = self.conv(x)
		return x


class up_conv(nn.Module):
	"""
	Up Convolution Block
	"""
	def __init__(self, in_ch, out_ch):
		super(up_conv, self).__init__()
		self.up = nn.Sequential(
			nn.Upsample(scale_factor=2),
			nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.up(x)
		return x
	
class resnet_decoder(nn.Module):
	def __init__(self, stage = 5):
		super(resnet_decoder, self).__init__()
		self.stage = stage
		if stage == 5:
			self.Up5 = up_conv(512, 512)
			self.Up_conv5 = conv_block(512 + 512, 512)
		self.Up4 = up_conv(512, 256)
		#self.Up4 = up_conv(512, 256)
		self.Up_conv4 = conv_block(512, 256)

		self.Up3 = up_conv(256, 128)
		self.Up_conv3 = conv_block(256, 128)

		self.Up2 = up_conv(128, 64)
		self.Up_conv2 = conv_block(128, 64)

		self.Conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

		self.active = torch.nn.Sigmoid()

	def forward(self, skips, coarse = None, fine = None):
		if self.stage == 5:
			d5 = self.Up5(skips[4])
			d5 = torch.cat((skips[3], d5), dim=1)
			d5 = self.Up_conv5(d5)
			d4 = self.Up4(d5)
		else:
			d4 = self.Up4(skips[3])
		
		d4 = torch.cat((skips[2], d4), dim=1)
		d4 = self.Up_conv4(d4)

		d3 = self.Up3(d4)
		d3 = torch.cat((skips[1], d3), dim=1)
		
		d3 = self.Up_conv3(d3)

		d2 = self.Up2(d3)


		d2 = torch.cat((skips[0], d2), dim=1)
		d2 = self.Up_conv2(d2)

		out = self.Conv(d2)

		out = self.active(out)

		return out


class aust_decoder(nn.Module):
	def __init__(self):
		super(aust_decoder, self).__init__()

		self.mix1 = nn.Sequential(
			nn.Conv2d(512*2, 512, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
		)

		self.aux1 = nn.Sequential(
			nn.Conv2d(256, 128, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 64, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 1, kernel_size=1, padding=0),
			nn.Sigmoid()
		)


		self.mix2 = nn.Sequential(
			nn.Conv2d(256*2 + 1, 256, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
		)

		self.aux2 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 64, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 1, kernel_size=1, padding=0),
			nn.Sigmoid()
		)

		self.mix3 = nn.Sequential(
			nn.Conv2d(128*2 + 1, 128, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
		)

		self.aux3 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 32, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 1, kernel_size=1, padding=0),
			nn.Sigmoid()
		)

		self.mix4 = nn.Sequential(
			nn.Conv2d(64*2 + 1, 64, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
		)


		self.Up4 = up_conv(512, 256)
		self.Up_conv4 = conv_block(512, 256)

		self.Up3 = up_conv(256, 128)
		self.Up_conv3 = conv_block(256, 128)

		self.Up2 = up_conv(128, 64)
		self.Up_conv2 = conv_block(128, 64)

		self.Conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

		self.active = torch.nn.Sigmoid()

		self.pooling = nn.AdaptiveAvgPool2d((1,1))


	def cal_score(self, aux_score, yuv_feature_norm):
		target_H, target_W = aux_score.shape[2], aux_score.shape[3]
		B, _, init_H, init_W = yuv_feature_norm.shape
		
		aux_score_small = F.interpolate(aux_score, (init_H, init_W), mode='bilinear')

		avg_feature = self.pooling(aux_score_small * yuv_feature_norm)
		avg_feature = F.normalize(avg_feature, p=2, dim=1)
		
		score_map = avg_feature * yuv_feature_norm
		score_map = torch.sum(score_map, dim=1, keepdim=True)
		score_map = torch.clamp(score_map, min = -1, max = 1)

		return F.interpolate(score_map, (target_H, target_W), mode='bilinear')

	def forward(self, rgb_skips, yuv_skips, yuv_feature_norm):

		skip5 = self.mix1(torch.cat([rgb_skips[-1], yuv_skips[-1]], 1))

		d4 = self.Up4(skip5)
		aux1 = self.aux1(d4)
		aux_score = 1- aux1.detach()
		score_map = self.cal_score(aux_score, yuv_feature_norm)
		init_score = score_map
		skip4 = self.mix2(torch.cat([rgb_skips[-2], yuv_skips[-2], score_map], 1))
		d4 = torch.cat((skip4, d4), dim=1)
		d4 = self.Up_conv4(d4)

		d3 = self.Up3(d4)
		aux2 = self.aux2(d3)
		aux_score = 1- aux2.detach()
		score_map = self.cal_score(aux_score, yuv_feature_norm)
		skip3 = self.mix3(torch.cat([rgb_skips[-3], yuv_skips[-3], score_map], 1))
		d3 = torch.cat((skip3, d3), dim=1)
		d3 = self.Up_conv3(d3)

		d2 = self.Up2(d3)
		aux3 = self.aux3(d2)
		aux_score = 1- aux3.detach()
		score_map = self.cal_score(aux_score, yuv_feature_norm)
		skip2 = self.mix4(torch.cat([rgb_skips[-4], yuv_skips[-4], score_map], 1))

		d2 = torch.cat((skip2, d2), dim=1)
		d2 = self.Up_conv2(d2)

		out = self.Conv(d2)

		out = self.active(out)

		return out, [aux1, aux2, aux3], score_map, init_score


class austs_decoder(nn.Module):
	def __init__(self):
		super(austs_decoder, self).__init__()

		self.mix1 = nn.Sequential(
			nn.Conv2d(512*2, 512, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
		)

		self.aux1 = nn.Sequential(
			nn.Conv2d(256, 128, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 64, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 1, kernel_size=1, padding=0),
			nn.Sigmoid()
		)


		self.mix2 = nn.Sequential(
			nn.Conv2d(256*2 + 1, 256, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
		)

		self.aux2 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 64, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 1, kernel_size=1, padding=0),
			nn.Sigmoid()
		)

		self.mix3 = nn.Sequential(
			nn.Conv2d(128*2 + 1, 128, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
		)

		self.aux3 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 32, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 1, kernel_size=1, padding=0),
			nn.Sigmoid()
		)

		self.mix4 = nn.Sequential(
			nn.Conv2d(64*2 + 1, 64, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=1, padding=0),
			nn.ReLU(inplace=True),
		)


		self.Up4 = up_conv(512, 256)
		self.Up_conv4 = conv_block(512, 256)

		self.Up3 = up_conv(256, 128)
		self.Up_conv3 = conv_block(256, 128)

		self.Up2 = up_conv(128, 64)
		self.Up_conv2 = conv_block(128, 64)

		self.Conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

		self.active = torch.nn.Sigmoid()

	def resize_similarity_map(self, similariy, H, W):
		B, init_H, init_W, _, _ = similariy.shape
		similariy = similariy.view(B, -1, init_H, init_W)
		similariy = F.interpolate(similariy, (H, W), mode='bilinear')
		similariy = similariy.view(B, -1, H*W)
		similariy = similariy.permute(0, 2, 1).view(B, -1, init_H, init_W)
		similariy = F.interpolate(similariy, (H, W), mode='bilinear').view(B, H*W, H*W).permute(0, 2, 1).view(B, H, W, H, W)
		return similariy

	def cal_score(self, aux_score, semantic_similarity, yuv_similarity):
		target_H, target_W = aux_score.shape[2], aux_score.shape[3]
		B, init_H, init_W, _, _ = semantic_similarity.shape

		aux_score_small = F.interpolate(aux_score, (init_H, init_W), mode='bilinear')

		score_map = (semantic_similarity * aux_score_small.view(-1, 1, 1, init_H, init_W))
		score_map /= torch.sum(score_map, dim=(3,4), keepdim=True)

		score_map = (score_map*yuv_similarity)
		score_map = torch.sum(score_map, dim = (3,4))

		score_map =  score_map.view(-1, 1, init_H, init_W)

		return F.interpolate(score_map, (target_H, target_W), mode='bilinear')

	def forward(self, rgb_skips, yuv_skips, semantic_similarity, yuv_similarity):
		skip5 = self.mix1(torch.cat([rgb_skips[-1], yuv_skips[-1]], 1))

		d4 = self.Up4(skip5)
		aux1 = self.aux1(d4)
		aux_score = 1- aux1.detach()
		score_map = self.cal_score(aux_score, semantic_similarity, yuv_similarity)
		init_score = score_map
		skip4 = self.mix2(torch.cat([rgb_skips[-2], yuv_skips[-2], score_map], 1))
		d4 = torch.cat((skip4, d4), dim=1)
		d4 = self.Up_conv4(d4)

		d3 = self.Up3(d4)
		aux2 = self.aux2(d3)
		aux_score = 1- aux2.detach()
		score_map = self.cal_score(aux_score, semantic_similarity, yuv_similarity)
		skip3 = self.mix3(torch.cat([rgb_skips[-3], yuv_skips[-3], score_map], 1))
		d3 = torch.cat((skip3, d3), dim=1)
		d3 = self.Up_conv3(d3)

		d2 = self.Up2(d3)
		aux3 = self.aux3(d2)
		aux_score = 1- aux3.detach()
		score_map = self.cal_score(aux_score, semantic_similarity, yuv_similarity)
		skip2 = self.mix4(torch.cat([rgb_skips[-4], yuv_skips[-4], score_map], 1))

		d2 = torch.cat((skip2, d2), dim=1)
		d2 = self.Up_conv2(d2)

		out = self.Conv(d2)

		out = self.active(out)

		return out, [aux1, aux2, aux3], score_map, init_score
	
class resnet_encoder(nn.Module):
	def __init__(self, in_channels = 3, stage = 4):
		super(resnet_encoder, self).__init__()
		resnet = models.resnet34(pretrained=True)
		self.inconv = nn.Conv2d(in_channels, 64, 3,1,1)
		self.inbn = nn.BatchNorm2d(64)
		self.inrelu = nn.ReLU(inplace=True) #224,64
		#stage 1
		self.encoder1 = resnet.layer1 #112,64*4
		#stage 2
		self.encoder2 = resnet.layer2 #56,128*4
		#stage 3
		self.encoder3 = resnet.layer3 #28,256*4
		#stage 4
		self.encoder4 = resnet.layer4 #14,512*4
		self.maxpool = nn.MaxPool2d(3,2,1)
		if stage == 5:
			self.encoder5 = nn.Sequential(*[
			BasicBlock(resnet.inplanes, 512),
			BasicBlock(512, 512),
			BasicBlock(512, 512),
		])
		self.stage = stage        
	def forward(self, x):
		hx = x
		hx = self.inconv(hx)
		hx = self.inbn(hx)
		hx = self.inrelu(hx)
		
		h1 = self.encoder1(hx) # 224
		h2 = self.encoder2(h1) # 112
		h3 = self.encoder3(h2) # 56
		h4 = self.encoder4(h3) # 28
		if self.stage == 5:
			h5 = self.encoder5(self.maxpool(h4))
			return [h1,h2,h3,h4,h5]
		else:
			return h1,h2,h3,h4