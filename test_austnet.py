import sys
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader

import numpy as np
import tqdm

from evaluation.metrics import FScore, normPRED, compute_mAP, compute_IoU
from options import ArgsParser


from data import iH4Dataset
from train_austnet import Engine


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
		mean[0,:,:] = .46962251
		mean[1,:,:] = .4464104
		mean[2,:,:] = .40718787
		std = torch.zeros_like(x)
		std[0,:,:] = 0.27469736
		std[1,:,:] = 0.27012361
		std[2,:,:] = 0.28515933
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
	pl.seed_everything(42)

	# Data
	opt.no_flip = True
	opt.phase = "val"
	opt.preprocess = 'resize'
	val_set = iH4Dataset(opt)
	
	dataloader_val = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=8)

	engine = Engine(opt)
	engine = Engine.load_from_checkpoint(opt.ckpt, opt = opt)

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
		with torch.no_grad():
			out, aux_list, init_score, final_score, feature = engine.model(comp, yuv)
			mask = torch.clamp(mask, 0, 1)
			out = torch.clamp(out, 0, 1)

			inharmonious_pred = normPRED(out)
			mask_gt = normPRED(mask)

			pred = inharmonious_pred
			
			label = mask_gt

			F1 = FScore(pred, label).item()
			FBeta = FScore(pred, label, threshold=-1, beta2=0.3)
			
			mAP = compute_mAP(pred, label)

			IoUMeter.update(compute_IoU(pred, label), label.size(0))
			mAPMeter.update(mAP, inharmonious_pred.size(0))
			F1Meter.update(F1, inharmonious_pred.size(0))
			FbMeter.update(FBeta, inharmonious_pred.size(0))

	print(mAPMeter.avg)
	print(F1Meter.avg)
	print(IoUMeter.avg)

