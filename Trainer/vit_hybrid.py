import os
import random
import time
import math

import numpy as np
from tqdm import tqdm
from collections import deque
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.nn.parallel import DistributedDataParallel as DDP

# import tensorflow as tf
# from tensorflow.keras.models import load_model
# cpu=tf.config.list_physical_devices("CPU")
# tf.config.set_visible_devices(cpu)
# gpu_list = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu_list[0], True)

from Network.VQVAE.vqvae import *
from Network.VQVAE.utils import *

# from Network.VQVAE import VectorQuantizer
# from Network.vqgan import VQModel

from Trainer.trainer import Trainer
from Network.MaskPLAN_PP import *
from Trainer.decode_function import *
from Trainer.Custom_Loss import *

import numpy as np
import torch

def adjust_learning_rate(lr,optimizer, epoch, decay):
    learning_rate = lr * (0.5 ** (epoch // decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

class MaskLayout(Trainer):

    def __init__(self, args):
        """ Initialization of the model (VQGAN and Masked Transformer), optimizer, criterion, etc."""
        super().__init__(args)
        self.args = args                                                        # Main argument see main.py
        self.scaler = torch.cuda.amp.GradScaler()                               # Init Scaler for multi GPUs
        self.sqe_len = self.args.sqe_len
        self.attri_count = self.args.attri_count
        self.batch_size = self.args.bsize
        self.img_weight = self.args.img_weight
        self.thres = self.args.thres
        self.device = self.args.device

        self.codebook_shape = self.args.codebook_shape
        self.codebook_size = self.args.codebook_size
        self.v_T_dim = 8
        self.v_L_dim = 7
        self.v_S_dim = 6
        self.R_num = 10
        # if self.args.img_loss:
        self.vq_model = load_model('vqvae_30.pth',self.args.device)[0]
        print("Acquired codebook size:", self.codebook_size)
        
        self.criterion = self.get_loss("cross_entropy", label_smoothing=0.1)  # Get cross entropy loss
        self.mse = self.get_loss("mse")
        self.cd = ChamferDistanceLoss_Filter()
        self.CodesReconsLoss = ComposeImgLoss(self.img_weight,self.attri_count,self.sqe_len,self.args.device)
        self.VecsRLoss = VecsRLoss(self.img_weight,self.args.device)
        
        self.vit = self.get_network("vit")
        self.scaler = torch.cuda.amp.GradScaler()                               # Init Scaler for multi GPUs
        self.optim = self.get_optim(self.vit, self.args.lr, betas=(0.9, 0.96))  # Get Adam Optimizer with weight decay
        
        # Load data if aim to train or test the model
        if not self.args.debug:
            self.train_data, self.test_data, self.img_data, valid = self.get_data()

        self.valid = valid.to(self.device)
        
        self.test_len = len(self.test_data[0])

        # Initialize evaluation object if testing
        if self.args.test_only:
            from Metrics.sample_and_eval import SampleAndEval
            self.sae = SampleAndEval(device=self.args.device, num_images=50_000)

    def get_network(self, archi):
        """ return the network, load checkpoint if self.args.resume == True
            :param
                archi -> str: vit|autoencoder, the architecture to load
            :return
                model -> nn.Module: the network
        """
        if archi == "vit":
            
            if self.args.model_size == 'small':
                h = 800
                d = 24
            elif self.args.model_size == 'big':
                h = 1040
                d = 36
            elif self.args.model_size == 'huge':
                h = 1280
                d = 48

            model = MaskTransformer_HYB_Graph(self.v_T_dim, self.v_L_dim, self.v_S_dim, self.R_num, self.args.mask_mode,
                sqe_len = self.sqe_len, attri_count=self.attri_count, hidden_dim=h, codebook_shape=self.codebook_shape
                ,codebook_size=self.codebook_size, depth=d, heads=16, mlp_dim=3072, dropout=0.1)

            if self.args.resume:
                ckpt = self.args.saved_model+"current.pth"
                # ckpt += "current.pth" #if os.path.isdir(self.args.vit_folder) else ""
                if self.args.is_master:
                    print("load ckpt from:", ckpt)
                # Read checkpoint file
                checkpoint = torch.load(ckpt, map_location='cpu')
                # Update the current epoch and iteration
                self.args.iter += checkpoint['iter']
                self.args.global_epoch += checkpoint['global_epoch']
                # Load network
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            model = model.to(self.args.device)
            if self.args.is_multi_gpus:  # put model on multi GPUs if available
                model = DDP(model, device_ids=[self.args.device])
            
            if self.args.is_multi_gpus: # put model on multi GPUs if available
                model = DDP(model, device_ids=[self.args.device])
                model = model.module
            
            if self.args.is_master:
                print(f"Size of model {archi}: "
                    f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")

        return model
    
    def vq_decode(self,code,device):
        # find closest encodings
        num_batch = code.size(0)
        min_encoding_indices = code.view(-1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.args.n_embeddings).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.vq_model.vector_quantization.embedding.weight).view(num_batch,4,4,self.args.embedding_dim)

        # preserve gradients
        # z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return self.vq_model.decoder(z_q)

    def get_mask_code_graph(self, code, T, S, L, A, R, W, valid, mode="square", value=1, value_img=31):
        """ Replace the code token by *value* according the the *mode* scheduler
           :param
            code  -> torch.LongTensor(): bsize x (6 x 14) x **, the unmasked code
            mode  -> str:                the rate of value to mask
            value -> int:                mask the code by the value
           :return
            masked_code -> torch.LongTensor(): bsize x (6 x 14) x **, the masked version of the code
            mask        -> torch.LongTensor(): bsize x 6 x 14 x 25, the binary mask of the mask
        """
        r = torch.rand(T.shape[0],self.sqe_len, device=self.device)

        mask_map = torch.ones(T.shape[0], self.attri_count, self.sqe_len).long().to(self.args.device)
         
        if mode == "linear":                # linear scheduler
            val_to_mask = r
        elif mode == "square":              # square scheduler
            val_to_mask = (r ** 2)
        elif mode == "cosine":              # cosine scheduler
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":              # arc cosine scheduler
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        else:
            val_to_mask = None

        mask_code = code.detach().clone()

        mask_T = T.detach().clone()
        mask_S = S.detach().clone()
        mask_L = L.detach().clone()
        mask_A = A.detach().clone()
        mask_R = R.detach().clone()
        mask_W = W.detach().clone()

        # random level mask
        # for i in range(self.batch_size): 
            
        #     random_mask = ((torch.rand(size=torch.Size([self.attri_count, int(valid[i])])) > val_to_mask[i,:int(valid[i])].view(1, int(valid[i]))).float()).long()
            
        #     condition = (random_mask[4] == 0) & ((random_mask[:4] != 0).any(dim=0) | (random_mask[-1] == 0))
        #     random_mask[4, condition] = 1  # Update the 4th row where the condition is met

        #     mask_map[i,:,:int(valid[i])] = random_mask

        # === Step 2. Create a random mask for all attributes and all sequence positions ===
        # Generate a tensor of uniform random numbers of shape (batch_size, attri_count, sqe_len)
        rand_vals = torch.rand(T.shape[0], self.attri_count, self.sqe_len, device=self.device)
        # Compare against the broadcasted threshold for each batch & sequence position.
        # The threshold is broadcast along the attribute dimension.
        random_mask = (rand_vals > val_to_mask.unsqueeze(1)).long() # shape: (batch_size, attri_count, sqe_len)

        # === Step 3. Update the 4th row based on a condition, only for valid positions ===
        # For each batch element, only consider columns < valid[i].
        col_idx = torch.arange(self.sqe_len, device=self.device).unsqueeze(0).expand(T.shape[0], self.sqe_len)
        valid_mask = col_idx < valid.unsqueeze(1)   # shape: (batch_size, sqe_len)

        # The condition is applied on the 4th attribute row.
        # For each batch element, for each valid column j, check:
        #   (a) random_mask[b, 4, j] == 0, and
        #   (b) either at least one of rows 0-3 at column j is nonzero OR the last row is zero.
        cond = (random_mask[:, 4, :] == 0) & (
                    (random_mask[:, :4, :] != 0).any(dim=1) | (random_mask[:, -1, :] == 0)
                )
        # Restrict the condition to valid columns only.
        cond = cond & valid_mask

        # Where the condition holds, set the 4th row to 1.
        random_mask[:, 4, :][cond] = 1

        # === Step 4. Write back only for the valid columns ===
        # We want to update only the first valid[i] columns of mask_map for each batch.
        # Create an expanded valid mask so that it covers all attributes.
        valid_mask_exp = valid_mask.unsqueeze(1).expand(T.shape[0], self.attri_count, self.sqe_len)
        # For positions that are valid, take the values from random_mask;
        # for positions that are not valid, keep the existing values in mask_map.
        mask_map = torch.where(valid_mask_exp, random_mask, mask_map).long()

        mask_code[mask_map] = torch.full_like(mask_code[mask_map], value_img)
        mask_T[mask_map[:,0,:]] = torch.full_like(mask_T[mask_map[:,0,:]], value)
        mask_S[mask_map[:,1,:]] = torch.full_like(mask_S[mask_map[:,1,:]], value)
        mask_L[mask_map[:,2,:]] = torch.full_like(mask_L[mask_map[:,2,:]], value)
        mask_A[mask_map[:,3,:]] = torch.full_like(mask_A[mask_map[:,3,:]], value)
        mask_R[mask_map[:,4,:]] = torch.full_like(mask_R[mask_map[:,4,:]], value)
        mask_W[mask_map[:,5,:]] = torch.full_like(mask_W[mask_map[:,5,:]], value)

        if self.args.mask_mode in [0,2,3,4,5,8]:
            all_mask = create_All_Mask(mask_map,self.args.device)
        else: all_mask = None

        if self.args.mask_mode in [0,1,3,4,7,8]:
            graph_mask = create_Graph_Mask(mask_A,mask_map,self.args.device)
        else: graph_mask = None

        if self.args.mask_mode in [0,1,2,4,7,8]:
            attr_mask = create_Attr_Mask(mask_map,self.args.device)
        else: attr_mask = None
        
        if self.args.mask_mode in [0,1,2,3,7,8]:
            room_mask = create_Room_Mask(mask_map,self.args.device)
        else: room_mask = None

        return mask_code, mask_T, mask_S, mask_L, mask_A, mask_R, mask_W, mask_map, all_mask, graph_mask, attr_mask, room_mask

    def adap_sche(self, step, mode="arccos", leave=False):
        """ Create a sampling scheduler
           :param
            step  -> int:  number of prediction during inference
            mode  -> str:  the rate of value to unmask
            leave -> bool: tqdm arg on either to keep the bar or not
           :return
            scheduler -> torch.LongTensor(): the list of token to predict at each step
        """
        r = torch.linspace(1, 0, step)
        if mode == "root":              # root scheduler
            val_to_mask = 1 - (r ** .5)
        elif mode == "linear":          # linear scheduler
            val_to_mask = 1 - r
        elif mode == "square":          # square scheduler
            val_to_mask = 1 - (r ** 2)
        elif mode == "cosine":          # cosine scheduler
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":          # arc cosine scheduler
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        else:
            return

        # fill the scheduler by the ratio of tokens to predict at each step
        sche = (val_to_mask / val_to_mask.sum()) * (self.sqe_len * self.attri_count)
        sche = sche.round()
        sche[sche == 0] = 1                                                  # add 1 to predict a least 1 token / step
        sche[-1] += (self.sqe_len * self.attri_count) - sche.sum()         # need to sum up nb of code
        return tqdm(sche.int(), leave=leave)

    def train_one_epoch(self, log_iter=10000):
        """ Train the model for 1 epoch """
        self.vit.train()

        cum_loss = 0.
        cumV_recons_loss = 0.
        cumI_recons_loss = 0.
        Code_loss = 0.
        T_loss = 0.
        S_loss = 0.
        L_loss = 0.
        A_loss = 0.
        R_loss = 0.
        W_loss = 0.
        CD_loss = 0.

        window_loss = deque(maxlen=self.args.grad_cum)
        bar = tqdm(self.train_data, leave=False) if self.args.is_master else self.train_data
        n = len(self.train_data)
        img_load_num = self.attri_count*self.sqe_len
        # Start training for 1 epoch
        idx = 0

        img_iter = iter(self.img_data)

        for x, site, T, S, L, A, R, W in bar:

            T = T.to(self.args.device)
            S = S.to(self.args.device)
            L = L.to(self.args.device)
            A = A.to(self.args.device)
            R = R.to(self.args.device)
            W = W.to(self.args.device)

            x = x.to(self.args.device)
            site = site.to(self.args.device)

            valid = self.valid[idx*self.batch_size:(idx+1)*self.batch_size]

            masked_code, mask_T, mask_S, mask_L, mask_A, mask_R, mask_W, mask_map, all_mask, graph_mask, attr_mask, room_mask = self.get_mask_code_graph(x, T, S, L, A, R, W, valid, mode="square")

            # mT = mT.to(self.args.device)
            # mS = mS.to(self.args.device)
            # mL = mL.to(self.args.device)
            # mA = mA.to(self.args.device)
            # mR = mR.to(self.args.device)
            # mW = mW.to(self.args.device)

            # choose for partial input
            # partial_rate = torch.empty(T.size(0)).uniform_(0, 1) < self.args.mask_rate

            # # assgin pre-defined partial input
            # masked_code[partial_rate] = m[partial_rate]

            # mask_T[partial_rate] = mT[partial_rate]
            # mask_S[partial_rate] = mS[partial_rate]
            # mask_L[partial_rate] = mL[partial_rate]
            # mask_A[partial_rate] = mA[partial_rate]
            # mask_R[partial_rate] = mR[partial_rate]
            # mask_W[partial_rate] = mW[partial_rate]

            with torch.cuda.amp.autocast():   # half precision
                out_img, out_T, out_S, out_L, out_A, out_R, out_W = self.vit(masked_code, mask_T, mask_S, mask_L, mask_A, mask_R, mask_W, site, all_mask, graph_mask, attr_mask, room_mask)  # The unmasked tokens prediction
                
                # Cross-entropy loss
                # self.v_T_dim, self.v_L_dim, self.v_S_dim
                loss_code = self.criterion(out_img.reshape(-1, self.codebook_size), x.view(-1)) / self.args.grad_cum
                loss_T = self.mse(out_T, T) / self.args.grad_cum
                loss_S = self.mse(out_S, S) / self.args.grad_cum
                loss_L = self.mse(out_L, L) / self.args.grad_cum
                loss_A = self.mse(out_A, A) / self.args.grad_cum
                loss_R = self.mse(out_R, R) / self.args.grad_cum
                loss_W = self.mse(out_W, W) / self.args.grad_cum
                loss_CD = self.cd(bit2int(out_R,self.v_L_dim).view(-1,self.R_num,2),bit2int(R,self.v_L_dim).view(-1,self.R_num,2)) / self.args.grad_cum

            # img recons loss
            if self.args.img_loss:
                prob = torch.softmax(out_img, -1)
                distri = torch.distributions.Categorical(probs=prob)
                pred_code = distri.sample()
                pred_imgs = self.vq_decode(pred_code.view(-1, self.codebook_shape),self.args.device)
                recons_loss_img = self.CodesReconsLoss(next(img_iter)[0],pred_imgs) / self.args.grad_cum

                # recons_loss_img = self.ImgRLoss(self.img_data[idx*self.batch_size:(idx+1)*self.batch_size,0], out_T, out_R, self.args.device)
                # cumV_recons_loss += recons_loss_vec.cpu().item()
                cumI_recons_loss += recons_loss_img.cpu().item()

                loss = loss_code + loss_T + loss_S + loss_L + loss_A + loss_R + loss_W + loss_CD + recons_loss_img # + recons_loss_vec
            
            else:

                loss = loss_code + loss_T + loss_S + loss_L + loss_A + loss_R + loss_W + loss_CD

            self.optim.zero_grad()

            self.scaler.scale(loss).backward()  # rescale to get more precise loss

            self.scaler.unscale_(self.optim)                      # rescale loss
            nn.utils.clip_grad_norm_(self.vit.parameters(), 1.0)  # Clip gradient
            self.scaler.step(self.optim)
            self.scaler.update()

            cum_loss += loss.cpu().item()
            Code_loss += loss_code.cpu().item()
            T_loss += loss_T.cpu().item()
            S_loss += loss_S.cpu().item()
            L_loss += loss_L.cpu().item()
            A_loss += loss_A.cpu().item()
            R_loss += loss_R.cpu().item()
            W_loss += loss_W.cpu().item()
            CD_loss += loss_CD.cpu().item()
            window_loss.append(loss.data.cpu().numpy().mean())

            # logs
            if self.args.is_master:
                self.log_add_scalar('Train/Loss', np.array(window_loss).sum(), self.args.iter)

            if self.args.iter % log_iter == 0 and self.args.is_master:

                # Save Network
                self.save_network(model=self.vit, path=self.args.saved_model+"current.pth",
                                  iter=self.args.iter, optimizer=self.optim, global_epoch=self.args.global_epoch)

            self.args.iter += 1
            idx += 1

        
        # Generate sample for visualization
        vec, img_1, img_2 = self.sample_hybrid(nb_sample=4)

        # Save visualization
        cv2.imwrite(self.args.saved_model + 'epo%d_img1.jpg'% (self.args.global_epoch),img_1[0])
        cv2.imwrite(self.args.saved_model + 'epo%d_vec.png'% (self.args.global_epoch),vec[0])
        cv2.imwrite(self.args.saved_model + 'epo%d_img2.png'% (self.args.global_epoch),img_2)

        if self.args.img_loss: return cum_loss / n, Code_loss / n,CD_loss / n, T_loss / n, S_loss / n, L_loss / n, A_loss / n, R_loss / n, W_loss / n, cumV_recons_loss / n, cumI_recons_loss / n
        else: return cum_loss / n, Code_loss / n, CD_loss / n, T_loss / n, S_loss / n, L_loss / n, A_loss / n, R_loss / n, W_loss / n, None, None

    def fit(self):
        """ Train the model """
        if self.args.is_master:
            print("Start training:")

        start = time.time()
        # Start training
        for e in range(self.args.global_epoch, self.args.epoch):
            adjust_learning_rate(self.args.lr,self.optim, self.args.global_epoch, self.args.lr_decay)
            print(f"lr: {self.optim.param_groups[0]['lr']}")
            # synch every GPUs
            if self.args.is_multi_gpus:
                self.train_data.sampler.set_epoch(e)

            # Train for one epoch
            sum_loss, Code_loss, CD_loss, T_loss, S_loss, L_loss, A_loss, R_loss, W_loss, Vrecons_loss, Irecons_loss = self.train_one_epoch(self.args.save_iter)

            # Synch loss
            if self.args.is_multi_gpus:
                train_loss = self.all_gather(train_loss, torch.cuda.device_count())

            # Save model
            # if e % 50 == 0 and self.args.is_master:
            #     self.save_network(model=self.vit, path=self.args.vit_folder + f"epoch_{self.args.global_epoch:03d}.pth",
            #                       iter=self.args.iter, optimizer=self.optim, global_epoch=self.args.global_epoch)

            # Clock time
            clock_time = (time.time() - start)
            if self.args.is_master:
                self.log_add_scalar('Train/GlobalLoss', sum_loss, self.args.global_epoch)
                if self.args.img_loss:
                    print(f"\rEpoch {self.args.global_epoch},"
                        f" Iter {self.args.iter :},"
                        f" SUM {sum_loss:.4f},"
                        f" Code {Code_loss:.4f},"
                        f" CD {CD_loss:.4f},"
                        f" Recon V {Vrecons_loss:.4f},"
                        f" Recon I {Irecons_loss:.4f},"
                        f" T {T_loss:.4f},"
                        f" S {S_loss:.4f},"
                        f" L {L_loss:.4f},"
                        f" A {A_loss:.4f},"
                        f" R {R_loss:.4f},"
                        f" W {W_loss:.4f},"
                        f" Time: {clock_time // 3600:.0f}h {(clock_time % 3600) // 60:.0f}min {clock_time % 60:.2f}s")
                else:
                    print(f"\rEpoch {self.args.global_epoch},"
                        f" Iter {self.args.iter :},"
                        f" SUM {sum_loss:.4f},"
                        f" Code {Code_loss:.4f},"
                        f" CD {CD_loss:.4f},"
                        f" T {T_loss:.4f},"
                        f" S {S_loss:.4f},"
                        f" L {L_loss:.4f},"
                        f" A {A_loss:.4f},"
                        f" R {R_loss:.4f},"
                        f" W {W_loss:.4f},"
                        f" Time: {clock_time // 3600:.0f}h {(clock_time % 3600) // 60:.0f}min {clock_time % 60:.2f}s")
            self.args.global_epoch += 1


    def eval(self):
        """ Evaluation of the model"""
        self.vit.eval()
        if self.args.is_master:
            print(f"Evaluation with hyper-parameter ->\n"
                  f"scheduler: {self.args.sched_mode}, number of step: {self.args.step}, "
                  f"softmax temperature: {self.args.sm_temp}, cfg weight: {self.args.cfg_w}, "
                  f"gumbel temperature: {self.args.r_temp}")
        # Evaluate the model
        m = self.sae.compute_and_log_metrics(self)
        self.vit.train()
        return m

    def reco(self, x=None, code=None, masked_code=None, unmasked_code=None, mask=None):
        """ For visualization, show the model ability to reconstruct masked img
           :param
            x             -> torch.FloatTensor: bsize x 3 x 256 x 256, the real image
            code          -> torch.LongTensor: bsize x 16 x 16, the encoded image tokens
            masked_code   -> torch.LongTensor: bsize x 16 x 16, the masked image tokens
            unmasked_code -> torch.LongTensor: bsize x 16 x 16, the prediction of the transformer
            mask          -> torch.LongTensor: bsize x 16 x 16, the binary mask of the encoded image
           :return
            l_visual      -> torch.LongTensor: bsize x 3 x (256 x ?) x 256, the visualization of the images
        """
        l_visual = [x]
        with torch.no_grad():
            if code is not None:
                code = code.view(code.size(0), self.patch_size, self.patch_size)
                # Decoding reel code
                _x = self.ae.decode_code(torch.clamp(code, 0, self.codebook_size-1))
                if mask is not None:
                    # Decoding reel code with mask to hide
                    mask = mask.view(code.size(0), 1, self.patch_size, self.patch_size).float()
                    __x2 = _x * (1 - F.interpolate(mask, (self.args.img_size, self.args.img_size)).to(self.args.device))
                    l_visual.append(__x2)
            if masked_code is not None:
                # Decoding masked code
                masked_code = masked_code.view(code.size(0), self.patch_size, self.patch_size)
                __x = self.ae.decode_code(torch.clamp(masked_code, 0,  self.codebook_size-1))
                l_visual.append(__x)

            if unmasked_code is not None:
                # Decoding predicted code
                unmasked_code = unmasked_code.view(code.size(0), self.patch_size, self.patch_size)
                ___x = self.ae.decode_code(torch.clamp(unmasked_code, 0, self.codebook_size-1))
                l_visual.append(___x)

        return torch.cat(l_visual, dim=0)
    
    def sample_hybrid(self, P=False, nb_sample=6, sm_temp=1, w=0, randomize="linear", r_temp=4.5, sched_mode="arccos", step=10):

        picked = np.random.choice(self.test_len, nb_sample, replace=False)
        self.vit.eval()

        l_codes = []  # Save the intermediate codes predicted
        l_mask = []   # Save the intermediate masks

        valid = torch.tensor([self.sqe_len]*nb_sample, dtype=torch.long).to(self.device)

        with torch.no_grad():

            x, site, T, S, L, A, R, W = self.test_data[0][picked], self.test_data[1][picked], self.test_data[2][picked], self.test_data[3][picked],self.test_data[4][picked], self.test_data[5][picked],self.test_data[6][picked], self.test_data[7][picked]

            T = T.to(self.args.device)
            S = S.to(self.args.device)
            L = L.to(self.args.device)
            A = A.to(self.args.device)
            R = R.to(self.args.device)
            W = W.to(self.args.device)
            x = torch.tensor(x, dtype=torch.long).to(self.device)
            site = torch.tensor(site, dtype=torch.long).to(self.device)

            masked_code, mask_T, mask_S, mask_L, mask_A, mask_R, mask_W, mask_map, all_mask, graph_mask, attr_mask, room_mask = self.get_mask_code_graph(x, T, S, L, A, R, W, valid, mode="square")

            # print(mask_map[0])
            mask = mask_map.float().view(nb_sample, self.attri_count*self.sqe_len)

            # Instantiate scheduler
            if isinstance(sched_mode, str):  # Standard ones
                scheduler = self.adap_sche(step, mode=sched_mode)
            else:  # Custom one
                scheduler = sched_mode

            # Beginning of sampling, t = number of token to predict a step "indice"
            for indice, t in enumerate(scheduler):
                if mask.sum() < t:  # Cannot predict more token than 6*14
                    t = int(mask.sum().item())

                if mask.sum() == 0:  # Break if code is fully predicted
                    break

                with torch.cuda.amp.autocast():  # half precision

                    out_img, out_T, out_S, out_L, out_A, out_R, out_W = self.vit(masked_code.clone(), mask_T.clone(), mask_S.clone(), mask_L.clone(), mask_A.clone(), mask_R.clone(), mask_W.clone(), site, all_mask, graph_mask, attr_mask, room_mask)
                        
                out_T = (out_T>self.thres).float()
                out_S = (out_S>self.thres).float()
                out_L = (out_L>self.thres).float()
                out_A = (out_A>self.thres).float()
                out_R = (out_R>self.thres).float()
                out_W = (out_W>self.thres).float()

                prob = torch.softmax(out_img * sm_temp, -1)
                # Sample the code from the softmax prediction
                distri = torch.distributions.Categorical(probs=prob)
                pred_code = distri.sample()

                # conf for each visual token element
                conf_element = torch.gather(prob, 3, pred_code.view(nb_sample, self.attri_count*self.sqe_len, self.codebook_shape, 1))
                # conf of each attribute
                conf = torch.sum(conf_element.view(nb_sample, self.attri_count*self.sqe_len,self.codebook_shape),-1)

                if randomize == "linear":  # add gumbel noise decreasing over the sampling process
                    ratio = (indice / (len(scheduler)-1))
                    rand = r_temp * np.random.gumbel(size=(nb_sample, self.attri_count*self.sqe_len)) * (1 - ratio)
                    conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(self.args.device)
                elif randomize == "warm_up":  # chose random sample for the 2 first steps
                    conf = torch.rand_like(conf) if indice < 2 else conf
                elif randomize == "random":   # chose random prediction at each step
                    conf = torch.rand_like(conf)

                # do not predict on already predicted tokens
                conf[~mask.bool()] = -math.inf

                # chose the predicted token with the highest confidence
                tresh_conf, indice_mask = torch.topk(conf.view(nb_sample, -1), k=t, dim=-1)
                tresh_conf = tresh_conf[:, -1]

                # replace the chosen tokens
                conf = (conf >= tresh_conf.unsqueeze(-1))
                f_mask = (mask.view(nb_sample, self.attri_count, self.sqe_len).float() * conf.view(nb_sample, self.attri_count, self.sqe_len).float()).bool()
                
                mask_T[f_mask[:,0,:]] = out_T[f_mask[:,0,:]]
                mask_S[f_mask[:,1,:]] = out_S[f_mask[:,1,:]]
                mask_L[f_mask[:,2,:]] = out_L[f_mask[:,2,:]]
                mask_A[f_mask[:,3,:]] = out_A[f_mask[:,3,:]]
                mask_R[f_mask[:,4,:]] = out_R[f_mask[:,4,:]]
                mask_W[f_mask[:,5,:]] = out_W[f_mask[:,5,:]]
                masked_code[f_mask] = pred_code.view(nb_sample, self.attri_count, self.sqe_len, self.codebook_shape)[f_mask]

                # update the mask
                for i_mask, ind_mask in enumerate(indice_mask):
                    mask[i_mask, ind_mask] = 0
                # l_codes.append(pred_code.view(nb_sample, self.sqe_len, self.attri_count, self.codebook_shape).clone())
                # l_mask.append(mask.view(nb_sample, self.sqe_len, self.attri_count).clone())

            # decode the final prediction
            _code = torch.clamp(masked_code, 0,  self.codebook_size-1)
            pred_imgs = self.vq_decode(_code.view(-1, self.codebook_shape),self.device)
            Pred_normalized = normalize_per_batch(pred_imgs).view(nb_sample, self.attri_count, self.sqe_len, 4, 128, 128)

            img_code = vq_imgs_decode(Pred_normalized,self.device)
            img_vec = vecs_R_test(nb_sample,mask_T,mask_R,mask_W,self.sqe_len)
            img_all = Pred_normalized[0].view(self.attri_count*self.sqe_len, 4, 128, 128).permute(0, 2, 3, 1).reshape(self.attri_count*self.sqe_len*128,  128, 4).cpu().numpy()*255

        self.vit.train()
        return img_vec, img_code, img_all

