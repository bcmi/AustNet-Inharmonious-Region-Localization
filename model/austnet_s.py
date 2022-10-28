import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder_decoder import resnet_encoder, austs_decoder

def feature_loss(feature, mask_small):
	batch, dim, h, w = feature.shape
	feature = F.normalize(feature, p=2, dim=1)
	feature = feature.view(batch, dim, h*w)
	mask = mask_small.view(batch, 1, h*w)
	sim_matrix = torch.matmul(feature.transpose(1,2), feature)
	sim_matrix = sim_matrix.view(batch, 1, h, w, h, w)
	mask_map = torch.where(mask == 0, torch.tensor(-1).to(mask.device).type_as(mask), mask).view(batch, 1, h*w)
	mask_map = torch.matmul(mask_map.transpose(1,2), mask_map)
	mask_map = mask_map.view(batch, 1, h, w, h, w)

	sim_matrix = torch.clamp(sim_matrix, -1, 1)
	inner_pair = (mask_map > 0)
	inter_pair = (mask_map < 0)
	if torch.sum(inner_pair) > 0:
		inner_loss = torch.sum(inner_pair*sim_matrix) / torch.sum(inner_pair)
	else:
		inner_loss = 0
	if torch.sum(inter_pair) > 0 :
		inter_loss = torch.sum(inter_pair*sim_matrix) / torch.sum(inter_pair)
	else:
		inter_loss = 0
	return inner_loss, inter_loss

class AustNet_S(nn.Module):

	def __init__(self):
		super(AustNet_S, self).__init__()
		self.color_transfer_encoder = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=7, padding=3),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=7, padding=3),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 6, kernel_size=3, padding=1),
		)

		self.yuv_encoder = resnet_encoder(stage=4)
		self.rgb_encoder = resnet_encoder(stage=4)

		self.decoder = austs_decoder()

		self.semantic_model = None

		self.H = 28
		self.W = 28

		self.relu = nn.ReLU(inplace=True)
		
		self.distance_map = self.calculate_distance_map()

	def calculate_distance_map(self):
		H = self.H
		W = self.W
		# mesh grid 
		xx = torch.arange(0, W).view(1,-1).repeat(H,1)
		yy = torch.arange(0, H).view(-1,1).repeat(1,W)
		xx = xx.view(1,H,W)
		yy = yy.view(1,H,W)
		grid = torch.cat((yy,xx),0).float()

		distance_map = torch.zeros([H,W,H,W])
		for i in range(H):
			for j in range(W):
				current_position = torch.tensor([i,j]).view(2,1,1)
				distance_map[i][j] = torch.sum(torch.abs(current_position - grid ), axis = 0)


		distance_map /= torch.max(distance_map)

		return distance_map.view(1,self.H, self.W, self.H, self.W)
		

	def forward(self, rgb, yuv=None, semantic_features=None):
		semantic_features = F.interpolate(semantic_features, (self.H, self.W), mode='bilinear')
		rgb_features = self.rgb_encoder(rgb)
		transfer_parameter = self.color_transfer_encoder(yuv)
		transfered_yuv = yuv * self.relu(transfer_parameter[:, 0:3]) + transfer_parameter[:, 3:]
		yuv_features = self.yuv_encoder(transfered_yuv)
		_, semantic_similarity, yuv_similarity = self.aggregate_yuv_features(semantic_features, yuv_features[-1])

		out, aux_list, final_score, init_score = self.decoder(rgb_features, yuv_features, semantic_similarity, yuv_similarity)

		return out, aux_list, init_score, final_score, yuv_features[-1], transfered_yuv


	def aggregate_yuv_features(self, semantic_feature, yuv_feature):
		B, _, _, _ = semantic_feature.shape
		H, W = self.H, self.W
		semantic_feature = semantic_feature.permute(0, 2, 3, 1)
		semantic_feature = semantic_feature.view(B, self.H*self.W, -1)
		semantic_feature = F.normalize(semantic_feature, p=2, dim=2)
		similarity_map = semantic_feature @ semantic_feature.permute(0,-1,-2)
		similarity_map = similarity_map.view(B, H, W, H, W)
		if self.distance_map.device != similarity_map.device:
			self.distance_map = self.distance_map.to(similarity_map.device) 
		score_map = self.distance_map * similarity_map
		score_map /= torch.sum(score_map, dim=(3,4), keepdim=True)

		yuv_feature = yuv_feature.permute(0, 2, 3, 1).view(B, self.H*self.W, -1)
		yuv_feature_norm = F.normalize(yuv_feature, p=2, dim=2)
		yuv_similarity_map = yuv_feature_norm @ yuv_feature_norm.permute(0,-1,-2)
		yuv_similarity_map = yuv_similarity_map.view(B, H, W, H, W)


		score_map = (score_map*yuv_similarity_map)
		score_map = torch.sum(score_map, dim = (3,4), keepdim=True).view(B, H, W, 1).permute(0, -1, 1, 2)
		return score_map.view(B, 1, H, W), similarity_map, yuv_similarity_map