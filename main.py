# Main file to launch training or evaluation
import os
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import argparse

import torch
from torch.distributed import init_process_group, destroy_process_group


def main(args):
    
    if args.model_mode == 'hybrid':
        from Trainer.vit_hybrid import MaskLayout
        masklayout = MaskLayout(args)

    elif args.model_mode == 'img':
        from Trainer.vit_img import MaskLayout
        masklayout = MaskLayout(args)

    elif args.model_mode == 'vec':
        from Trainer.vit_vec import MaskLayout
        masklayout = MaskLayout(args)

    os.makedirs(args.saved_model,exist_ok=True)

    if args.test_only:  # Evaluate the networks
        masklayout.eval()

    elif args.debug:  # custom code for testing inference
        import torchvision.utils as vutils
        from torchvision.utils import save_image
        with torch.no_grad():
            labels, name = [1, 7, 282, 604, 724, 179, 681, 367, 635, random.randint(0, 999)] * 1, "r_row"
            labels = torch.LongTensor(labels).to(args.device)
            sm_temp = 1.3          # Softmax Temperature
            r_temp = 7             # Gumbel Temperature
            w = 9                  # Classifier Free Guidance
            randomize = "linear"   # Noise scheduler
            step = 32              # Number of step
            sched_mode = "arccos"  # Mode of the scheduler
            # Generate sample
            gen_sample, _, _ = masklayout.sample(nb_sample=labels.size(0), labels=labels, sm_temp=sm_temp, r_temp=r_temp, w=w,
                                              randomize=randomize, sched_mode=sched_mode, step=step)
            gen_sample = vutils.make_grid(gen_sample, nrow=5, padding=2, normalize=True)
            # Save image
            save_image(gen_sample, f"saved_img/sched_{sched_mode}_step={step}_temp={sm_temp}"
                                   f"_w={w}_randomize={randomize}_{name}.jpg")
    else:  # Begin training
        masklayout.fit()


def ddp_setup():
    """ Initialization of the multi_gpus training"""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def launch_multi_main(args):
    """ Launch multi training"""
    ddp_setup()
    args.device = int(os.environ["LOCAL_RANK"])
    args.is_master = args.device == 0
    main(args)
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         type=str,   default="MaskLayout", help="dataset on which dataset to train")
    parser.add_argument("--composed-img",  type=str,   default="Data/img/composed",help="folder containing the composed img")
    parser.add_argument("--split",        type=int,   default=32*2700,    help="split for train and val")
    parser.add_argument("--vqgan-folder", type=str,   default="",         help="folder of the pretrained VQGAN")
    parser.add_argument("--model_size",   type=str,   default="huge",    help="small,big,huge")
    parser.add_argument("--vit-folder",   type=str,   default="vit",         help="folder where to save the Transformer")
    parser.add_argument("--writer-log",   type=str,   default="",         help="folder where to store the logs")
    parser.add_argument("--sched_mode",   type=str,   default="arccos",   help="scheduler mode whent sampling")
    parser.add_argument("--model_mode",   type=str,   default="hybrid",   help="img,vec,hybrid(img+vec)")
    parser.add_argument("--mask_mode",   type=int,    default=0,          help="see details in MaskLayout.py")
    parser.add_argument("--sqe_len",      type=int,   default=14,         help="sqe len")
    parser.add_argument("--codebook_shape",type=int,  default=16,         help="codebook shape")
    parser.add_argument("--codebook_size",type=int,   default=32,         help="codebook_size")
    parser.add_argument("--attri_count",  type=int,   default=6,          help="attri_count")
    parser.add_argument("--grad-cum",     type=int,   default=1,          help="accumulate gradient")
    parser.add_argument('--channel',      type=int,   default=4,          help="rgb or black/white image")
    parser.add_argument("--num_workers",  type=int,   default=8,          help="number of workers")
    parser.add_argument("--step",         type=int,   default=8,          help="number of step for sampling")
    parser.add_argument('--seed',         type=int,   default=42,         help="fix seed")
    parser.add_argument("--epoch",        type=int,   default=100,        help="number of epoch")
    parser.add_argument("--save_iter",    type=int,   default=10000,       help="save after how many iterations")
    parser.add_argument('--img-size',     type=int,   default=256,        help="image size")
    parser.add_argument("--bsize",        type=int,   default=64,        help="batch size")
    parser.add_argument("--tsize",        type=int,   default=10,        help="test batch size")
    parser.add_argument("--mask_value",   type=int,   default=31,       help="number of epoch")
    parser.add_argument("--lr_decay",     type=int,   default=30,       help="number of epoch to reduce lr")
    parser.add_argument("--lr",           type=float, default=1e-4,       help="learning rate to train the transformer")
    parser.add_argument("--cfg_w",        type=float, default=3,          help="classifier free guidance wight")
    parser.add_argument("--r_temp",       type=float, default=4.5,        help="Gumbel noise temperature when sampling")
    parser.add_argument("--sm_temp",      type=float, default=1.,         help="temperature before softmax when sampling")
    parser.add_argument("--img_weight",   type=float, default=3.0,        help="img loss weight")
    parser.add_argument("--thres",        type=float, default=0.8,        help="thresh to predict in vec")
    parser.add_argument("--img_loss",     action='store_true',            help="training with img recons loss")
    parser.add_argument("--test-only",    action='store_true',            help="only evaluate the model")
    parser.add_argument("--resume",       action='store_true',            help="resume training of the model")
    parser.add_argument("--debug",        action='store_true',            help="debug")
    # VQ PARAM
    parser.add_argument("--n_hiddens", type=int, default=256)
    parser.add_argument("--n_residual_hiddens", type=int, default=64)
    parser.add_argument("--n_residual_layers", type=int, default=3)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--n_embeddings", type=int, default=32)
    
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.iter = 0
    args.global_epoch = 0
    if args.img_loss: args.saved_model = args.vit_folder+'_'+args.model_mode+'_'+args.model_size+'_mode_'+str(args.mask_mode)+'_img_loss/'
    else: args.saved_model = args.vit_folder+'_'+args.model_mode+'_'+args.model_size+'_mode_'+str(args.mask_mode)+'/'

    if args.seed > 0: # Set the seed for reproducibility
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.enable = True
        torch.backends.cudnn.deterministic = True

    world_size = torch.cuda.device_count()

    if world_size > 1:  # launch multi training
        print(f"{world_size} GPU(s) found, launch multi-gpus training")
        args.is_multi_gpus = True
        launch_multi_main(args)
    else:  # launch single Gpu training
        print(f"{world_size} GPU found")
        args.is_master = True
        args.is_multi_gpus = False
        main(args)
