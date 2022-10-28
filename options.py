import argparse

def ArgsParser():
        parser = argparse.ArgumentParser()
        # Datasets
        parser.add_argument('--dataset_root', type=str, default="iHarmony4/", help='dataset path')
        parser.add_argument('--logdir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')

        parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=224, help='then crop to this size')

        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        # training parameters
        parser.add_argument('--nepochs', type=int, default=250, help='# of training epochs')
        parser.add_argument('--gpus', type=int, default=4, help='# of GPUs used for training')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        
        parser = parser.parse_args()
        return parser
