import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader
import torch.optim as optim

from data import iH4Dataset
import pytorch_iou
import pytorch_ssim
from evaluation.metrics import FScore, normPRED, compute_mAP, compute_IoU
from model.austnet import AustNet, feature_loss
from options import ArgsParser


class Engine(pl.LightningModule):
	def __init__(self, opt):
		super().__init__()
		self.opt = opt
		self.model = AustNet()

		self.bce_loss = nn.BCELoss(size_average=True)
		self.ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
		self.iou_loss = pytorch_iou.IOU(size_average=True)
	

	def forward(self, batch):
		comp = batch["comp"]
		yuv = batch['yuv']
		out, aux_list, init_score, final_score, feature = self.model(comp, yuv)
		return out

	

	def training_step(self, batch, batch_idx):
		comp = batch["comp"]
		real = batch["real"]
		mask = batch["mask"].type(torch.FloatTensor).to(comp.device)
		mask_2 = batch["mask_2"].type(torch.FloatTensor).to(comp.device)
		mask_4 = batch["mask_4"].type(torch.FloatTensor).to(comp.device)
		mask_8 = batch["mask_8"].type(torch.FloatTensor).to(comp.device)
		yuv = batch['yuv']
		
		out, aux_list, init_score, final_score, feature = self.model(comp, yuv)
		aux1, aux2, aux3 = aux_list[0], aux_list[1], aux_list[2]

		out = torch.clamp(out, 0, 1)
		aux1 = torch.clamp(aux1, 0, 1)
		aux2 = torch.clamp(aux2, 0, 1)
		aux3 = torch.clamp(aux3, 0, 1)
		mask = torch.clamp(mask, 0, 1)
		mask_2 = torch.clamp(mask_2, 0, 1)
		mask_4 = torch.clamp(mask_4, 0, 1)
		mask_8 = torch.clamp(mask_8, 0, 1)
		init_score = torch.clamp(1- (1 + init_score)/2, 0, 1)
		final_score = torch.clamp(1- (1 + final_score)/2, 0, 1)

		bce_loss = self.bce_loss(out, mask)
		ssim_loss = 1 - self.ssim_loss(out, mask)
		iou_loss = self.iou_loss(out,mask)

		bce_loss_aux1 = self.bce_loss(aux1, mask_4)
		ssim_loss_aux1 = 1 - self.ssim_loss(aux1, mask_4)
		iou_loss_aux1 = self.iou_loss(aux1,mask_4)

		bce_loss_aux2 = self.bce_loss(aux2, mask_2)
		ssim_loss_aux2 = 1 - self.ssim_loss(aux2, mask_2)
		iou_loss_aux2 = self.iou_loss(aux2,mask_2)

		bce_loss_aux3 = self.bce_loss(aux3, mask)
		ssim_loss_aux3 = 1 - self.ssim_loss(aux3, mask)
		iou_loss_aux3 = self.iou_loss(aux3,mask)

		loss_inner, loss_inter = feature_loss(feature, mask_8)

		loss = bce_loss + ssim_loss + iou_loss

		aux_loss = bce_loss_aux1 + ssim_loss_aux1 + iou_loss_aux1 + bce_loss_aux2 + ssim_loss_aux2 + iou_loss_aux2 + bce_loss_aux3 + ssim_loss_aux3 + iou_loss_aux3
		loss += aux_loss/3
		loss +=  max(0, loss_inter - loss_inner + 0.5)

		self.log('train_loss', loss.item())
		self.log('bce_loss', bce_loss.item())
		self.log('ssim_loss', ssim_loss.item())
		self.log('iou_loss', iou_loss.item())
		self.log('aux_loss', aux_loss.item())
		self.log('loss_inner', loss_inner.item())
		self.log('loss_inter', loss_inter.item())

		return loss
	def configure_optimizers(self):
		optimizer = optim.Adam(self.model.parameters(), lr=self.opt.lr, betas=(0.9,0.999),  weight_decay=1e-4)
		lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
		return [optimizer], [lr_scheduler]

	def validation_step(self, batch, batch_idx):
		comp = batch["comp"]
		real = batch["real"]
		mask = batch["mask"].type(torch.FloatTensor).to(comp.device)
		yuv = batch['yuv']
		out, _, _,_,_ = self.model(comp, yuv)

		out = normPRED(out)
		mask = normPRED(mask)

		out = torch.clamp(out, 0, 1)
		mask = torch.clamp(mask, 0, 1)

		F1 = FScore(out, mask)
		mAP = compute_mAP(out, mask)

		IoU = compute_IoU(out, mask)

		self.log('F1', F1, sync_dist=True)
		self.log('mAP', mAP, sync_dist=True)
		self.log("IoU", IoU, sync_dist=True)

if __name__ == '__main__':
	opt = ArgsParser()
	pl.seed_everything(42)

	# Data
	opt.phase = "train"
	train_set = iH4Dataset(opt)
	opt.no_flip = True
	opt.phase = "val"
	opt.preprocess = 'resize'
	val_set = iH4Dataset(opt)
	opt.phase = "train"
	
	dataloader_train = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=8)
	dataloader_val = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=8)

	engine = Engine(opt)

	checkpoint_callback = ModelCheckpoint(save_weights_only=False, mode="max", monitor="mAP", save_top_k=2, save_last=True,
											dirpath=opt.logdir, filename="best_{epoch:02d}-{mAP:.3f}")
	checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"
	trainer = pl.Trainer.from_argparse_args(opt,
											default_root_dir=opt.logdir,
											sync_batchnorm=True,
											gpus = opt.gpus ,
											accelerator='ddp',
											profiler='simple',
											benchmark=True,
											log_every_n_steps=1,
											plugins=DDPPlugin(find_unused_parameters=False),
											flush_logs_every_n_steps=5,
											callbacks=[checkpoint_callback,
														],
											check_val_every_n_epoch = 3,
											max_epochs = opt.nepochs
											)

	trainer.fit(engine, dataloader_train, dataloader_val)