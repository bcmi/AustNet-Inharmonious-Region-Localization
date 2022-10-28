import sys
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader

import numpy as np
import tqdm

from evaluation.metrics import FScore, normPRED, compute_mAP, compute_IoU
from options import ArgsParser


sys.path.append("/home/wph/voting_dirl/HRNet-Semantic-Segmentation-HRNet-OCR/lib")
# import models 
# from config import config
# from config import update_config
from data import iH4Dataset
from train_score_multiscale import Engine
# from train_simscore_multiscale import Engine

def denormalize(x, isMask=False):
	if isMask:
		mean = 0
		std=1
		x = x.numpy().transpose(0,2,3,1)
		# x = np.where(x>0.5, 1, 0)
		x = (x*std + mean)*255
		x = x.astype(np.uint8)
	else:
		mean = torch.zeros_like(x)
		mean[:,0,:,:] = .485
		mean[:,1,:,:] = .456
		mean[:,2,:,:] = .406
		std = torch.zeros_like(x)
		std[:,0,:,:] = 0.229
		std[:,1,:,:] = 0.224
		std[:,2,:,:] = 0.225
		x = (x*std + mean)*255
		x = x.numpy().transpose(0,2,3,1).astype(np.uint8)     
	return x


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count



if __name__ == '__main__':
	opt = ArgsParser()
	opt.seed = 42
	pl.seed_everything(opt.seed)
	opt.dataset_root = r"/mnt/disk01/townall"

	opt.logdir = "/home/wupenghao/transfuser/dirl/log_paper_base_cosine20/best_epoch=218-mAP=0.922.ckpt"

	# Data
	opt.no_flip = True
	opt.phase = "val"
	opt.preprocess = 'resize'
	val_set = iH4Dataset(opt)
	opt.phase = "train"
	
	dataloader_val = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=8)

	engine = Engine(opt)

	engine = Engine.load_from_checkpoint("/home/wupenghao/transfuser/dirl/log_paper_semantic_cosine20/best_419-0.930.ckpt", opt = opt)
	# engine = Engine.load_from_checkpoint("/home/wupenghao/transfuser/dirl/log_paper_base_cosine20/best_epoch=218-mAP=0.922.ckpt", opt = opt)

	engine.eval()
	engine.freeze()

	engine = engine.cuda()
	total_number = 0

	mAPMeter = AverageMeter()
	F1Meter = AverageMeter()
	FbMeter = AverageMeter()
	IoUMeter = AverageMeter()
	innerMeter = AverageMeter()
	interMeter = AverageMeter()

	total_number = 0
	total_time = 0

	for b_idx, batch in tqdm.tqdm(enumerate(dataloader_val), total = len(dataloader_val)):

		comp = batch["comp"].to("cuda")
		mask = batch["mask"].type(torch.FloatTensor).to("cuda")
		yuv = batch['yuv'].to("cuda")
		# mask_8 = batch["mask_8"].type(torch.FloatTensor).to(comp.device)
		# engine.semantic_model.eval()
		with torch.no_grad():

			# torch.cuda.synchronize()
			# start = time.time()
			semantic_feature, semantic_out = engine.semantic_model(comp)
			out, aux_list, init_score, final_score, feature, transfered_yuv = engine.model(comp, yuv, semantic_feature)
			# out, aux_list, init_score, final_score, feature,transfered_yuv = engine.model(comp, yuv)
			# torch.cuda.synchronize()
			# end = time.time()
			# total_time += end-start
			# total_number += 1

			
			# aux1, aux2, aux3 = aux_list[0], aux_list[1], aux_list[2]

			# aux1 = F.interpolate(aux1, (224, 224), mode='bilinear')
			# seman_vis = F.interpolate(seman_vis, (224, 224), mode='bilinear')
			# seman_vis = torch.pow(seman_vis, 5)

			mask = torch.clamp(mask, 0, 1)
			out = torch.clamp(out, 0, 1)
			# init_score = torch.clamp(1-init_score, 0, 1)
			# final_score = torch.clamp(1-final_score, 0, 1)

			# comp_vis = normPRED((1-aux1)*seman_vis)
			# aux1 = normPRED(torch.clamp(1-aux1, 0, 1))
			# seman_vis = normPRED(torch.clamp(seman_vis, -1, 1))
			

			# loss_inner, loss_inter = feature_loss(feature, mask_8)
			
			# yuv = torch.cat([normPRED(yuv[:, 0:1]), normPRED(yuv[:, 1:2]), normPRED(yuv[:, 2:3])], 3)
			# transfered_yuv = torch.cat([normPRED(transfered_yuv[:, 0:1]), normPRED(transfered_yuv[:, 1:2]), normPRED(transfered_yuv[:, 2:3])], 3)

			# yuv = torch.cat([yuv, transfered_yuv], 2)

			inharmonious_pred = normPRED(out)
			mask_gt = normPRED(mask)

			pred = inharmonious_pred
			
			label = mask_gt

			F1 = FScore(pred, label)
			FBeta = FScore(pred, label, threshold=-1, beta2=0.3)
			
			mAP = compute_mAP(pred, label)

			

			IoUMeter.update(compute_IoU(pred, label), label.size(0))
			# innerMeter.update(loss_inner.detach().cpu().numpy(), label.size(0))
			# interMeter.update(loss_inter.detach().cpu().numpy(), label.size(0))
			mAPMeter.update(mAP, inharmonious_pred.size(0))
			F1Meter.update(F1, inharmonious_pred.size(0))
			FbMeter.update(FBeta, inharmonious_pred.size(0))

			# if b_idx == 2064 or b_idx == 2197 or b_idx == 6453:
			# 	print(IoUMeter.count, mAPMeter.avg, F1Meter.avg, IoUMeter.avg)
			# 	IoUMeter.reset()
			# 	mAPMeter.reset()
			# 	F1Meter.reset()
			# 	IoUMeter.reset()
			# if b_idx % 5 == 0:
			# 	print(IoUMeter.count, mAPMeter.avg, F1Meter.avg, IoUMeter.avg)

			
			# comp = denormalize(comp.detach().cpu())
			# inharmonious_pred = denormalize(inharmonious_pred.detach().cpu().repeat(1,3,1,1), isMask= True)
			# comp = comp[:, :, :, ::-1]
			# aux1 = denormalize(aux1.detach().cpu().repeat(1,3,1,1), isMask= True)
			# seman_vis = denormalize(seman_vis.detach().cpu().repeat(1,3,1,1), isMask= True)
			# comp_vis = denormalize(comp_vis.detach().cpu().repeat(1,3,1,1), isMask= True)
			# mask_gt_img = denormalize(mask_gt.detach().cpu().repeat(1,3,1,1), isMask= True)

			# first_row = np.concatenate([comp[0], inharmonious_pred[0], mask_gt_img[0]], 1)

			# yuv = denormalize(yuv.detach().cpu().repeat(1,3,1,1), isMask= True)
			# transfered_yuv = denormalize(transfered_yuv.detach().cpu().repeat(1,3,1,1), isMask= True)

			# bg = np.concatenate([first_row, yuv[0]], 0)
			# init_score_img = denormalize(init_score.detach().cpu().repeat(1,3,1,1), isMask= True)
			# final_score_img = denormalize(final_score.detach().cpu().repeat(1,3,1,1), isMask= True)

			# cv2.imwrite(str(b_idx)+'aux1.png', aux1[0])
			# cv2.imwrite(str(b_idx)+'comp_vis.png', comp_vis[0])
			# cv2.imwrite(str(b_idx)+'pred.png', inharmonious_pred[0])
			# cv2.imwrite(str(b_idx)+'seman_vis.png', seman_vis[0])

			# cv2.imwrite(os.path.join("vis_multi_foreground", "pred_"+str(b_idx)+'.png'), first_row)
			# cv2.imwrite(os.path.join("vis_yuv", "pred_"+str(b_idx)+'.png'), inharmonious_pred[0])
			# cv2.imwrite(os.path.join("vis_yuv", "mask_gt_"+str(b_idx)+'.png'), mask_gt_img[0])
			# cv2.imwrite(os.path.join("vis_yuv", "yuv_"+str(b_idx)+'.png'), yuv[0])
			# cv2.imwrite(os.path.join("vis_yuv1", "transfered_yuv_"+str(b_idx)+'.png'), yuv[0])

			# cv2.imwrite(os.path.join("vis_yuv1", "transfered_yuv_"+str(b_idx)+'.png'), bg)

			# cv2.imwrite(os.path.join("init_score", "init_score_"+str(b_idx)+'.png'), init_score_img[0])
			# cv2.imwrite(os.path.join("final_score", "final_score_"+str(b_idx)+'.png'), final_score_img[0])

	print(mAPMeter.avg)
	print(F1Meter.avg)
	# print(FbMeter.avg)
	print(IoUMeter.avg)
	# print(innerMeter.avg)
	# print(interMeter.avg)
	# print(total_time/total_number)

