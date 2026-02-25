# Trainer for MaskGIT
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

# from Network.VQVAE import VectorQuantizer

from Trainer.trainer import Trainer
from Network.MaskLAYOUT import *
from Trainer.decode_function import *
from Trainer.Custom_Loss import *

import numpy as np
import torch

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

        self.codebook_shape = self.args.codebook_shape
        self.codebook_size = self.args.codebook_size
        self.v_T_dim = 8
        self.v_L_dim = 7
        self.v_S_dim = 6
        self.R_num = 20

        self.criterion = self.get_loss("cross_entropy", label_smoothing=0.1)  # Get cross entropy loss
        self.mse = self.get_loss("mse")
        self.cd = ChamferDistanceLoss()
        self.vit = self.get_network("vit")
        self.scaler = torch.cuda.amp.GradScaler()                               # Init Scaler for multi GPUs
        # self.init_vq_decoder()
        self.optim = self.get_optim(self.vit, self.args.lr, betas=(0.9, 0.96))  # Get Adam Optimizer with weight decay
        
        # Load data if aim to train or test the model
        if not self.args.debug:
            self.train_data, self.test_data, self.img_data = self.get_data()
        
        self.test_len = len(self.test_data)

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
                h = 900
                d = 24
            elif self.args.model_size == 'big':
                h = 1050
                d = 36
            elif self.args.model_size == 'huge':
                h = 1200
                d = 48

            model = MaskTransformer_VEC(self.v_T_dim, self.v_L_dim, self.v_S_dim,
                sqe_len = self.sqe_len, attri_count=self.attri_count, hidden_dim=h, codebook_shape=self.codebook_shape
                ,codebook_size=self.codebook_size, depth=d, heads=15, mlp_dim=3072, dropout=0.1)

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
    
    # def init_vq_decoder(self):
    #     self.vq_decoder = load_model('VQ_Pretrained/55_32/vqvae_de/decoder.keras')
    #     vq_value = np.load('VQ_Pretrained/55_32/vqvae_q/quantizer.npy')
    #     self.quantizer = VectorQuantizer(32, 32)
    #     self.quantizer.embeddings = tf.Variable(
    #         initial_value=vq_value,
    #         trainable=False,
    #         name="embeddings_vqvae",)

    def get_mask_code_vec(self, T, S, L, A, R, W, mode="arccos", value=None):
        """ Replace the code token by *value* according the the *mode* scheduler
           :param
            code  -> torch.LongTensor(): bsize x (6 x 14) x **, the unmasked code
            mode  -> str:                the rate of value to mask
            value -> int:                mask the code by the value
           :return
            masked_code -> torch.LongTensor(): bsize x (6 x 14) x **, the masked version of the code
            mask        -> torch.LongTensor(): bsize x 6 x 14 x 25, the binary mask of the mask
        """
        r = torch.rand(T.size(0))
         
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

        mask_T = T.detach().clone()
        mask_S = S.detach().clone()
        mask_L = L.detach().clone()
        mask_A = A.detach().clone()
        mask_R = R.detach().clone()
        mask_W = W.detach().clone()

        # Sample the amount of tokens + localization to mask
        mask = torch.rand(size=torch.Size([T.shape[0], self.attri_count, self.sqe_len])) < val_to_mask.view(T.size(0), 1, 1)

        mask_T[mask[:,0,:]] = torch.full_like(mask_T[mask[:,0,:]], value)
        mask_S[mask[:,1,:]] = torch.full_like(mask_S[mask[:,1,:]], value)
        mask_L[mask[:,2,:]] = torch.full_like(mask_L[mask[:,2,:]], value)
        mask_A[mask[:,3,:]] = torch.full_like(mask_A[mask[:,3,:]], value)
        mask_R[mask[:,4,:]] = torch.full_like(mask_R[mask[:,4,:]], value)
        mask_W[mask[:,5,:]] = torch.full_like(mask_W[mask[:,5,:]], value)

        return mask_T, mask_S, mask_L, mask_A, mask_R, mask_W, mask

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
        cum_recons_loss = 0.
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
        # self.batch_img = np.zeros((self.batch_size,self.attri_count*self.sqe_len,128,128,4))
        # Start training for 1 epoch
        idx = 0
        for T, S, L, A, R, W, site, mT, mS, mL, mA, mR, mW, in bar:

            T = T.to(self.args.device)
            S = S.to(self.args.device)
            L = L.to(self.args.device)
            A = A.to(self.args.device)
            R = R.to(self.args.device)
            W = W.to(self.args.device)

            site = site.to(self.args.device)

            mT = mT.to(self.args.device)
            mS = mS.to(self.args.device)
            mL = mL.to(self.args.device)
            mA = mA.to(self.args.device)
            mR = mR.to(self.args.device)
            mW = mW.to(self.args.device)

            # choose for partial input
            partial_rate = torch.empty(T.size(0)).uniform_(0, 1) < self.args.mask_rate

            # Mask the encoded tokens
            mask_T, mask_S, mask_L, mask_A, mask_R, mask_W, mask = self.get_mask_code_vec(T, S, L, A, R, W, mode="arccos", value=1)

            # assgin pre-defined partial input
            mask_T[partial_rate] = mT[partial_rate]
            mask_S[partial_rate] = mS[partial_rate]
            mask_L[partial_rate] = mL[partial_rate]
            mask_A[partial_rate] = mA[partial_rate]
            mask_R[partial_rate] = mR[partial_rate]
            mask_W[partial_rate] = mW[partial_rate]

            with torch.cuda.amp.autocast():   # half precision
                out_T, out_S, out_L, out_A, out_R, out_W = self.vit(mask_T, mask_S, mask_L, mask_A, mask_R, mask_W, site)  # The unmasked tokens prediction
                
                # Cross-entropy loss
                # self.v_T_dim, self.v_L_dim, self.v_S_dim
                loss_T = self.mse(out_T, T) / self.args.grad_cum
                loss_S = self.mse(out_S, S) / self.args.grad_cum
                loss_L = self.mse(out_L, L) / self.args.grad_cum
                loss_A = self.mse(out_A, A) / self.args.grad_cum
                loss_R = self.mse(out_R, R) / self.args.grad_cum
                loss_W = self.mse(out_W, W) / self.args.grad_cum
                loss_CD = self.cd(bit2int(out_R,self.v_L_dim).view(-1,self.R_num,2),bit2int(R,self.v_L_dim).view(-1,self.R_num,2))
                
                # img recons loss
            if self.args.img_loss:
                #     img = load_img_batch_all(self.img_data[idx*self.batch_size:(idx+1)*self.batch_size,0],img_load_num,self.args.device)
                #     prob = torch.softmax(pred, -1)
                #     # Sample the code from the softmax prediction
                #     distri = torch.distributions.Categorical(probs=prob)
                #     pred_code = distri.sample().cpu().numpy()#.view(m.size(0),img_load_num, self.codebook_shape)
                #     priors_ohe = tf.one_hot(pred_code.astype("int32"), 32).numpy()
                #     quantized = tf.matmul(priors_ohe.astype("float32"), self.quantizer.embeddings, transpose_b=True)
                #     quantized = tf.reshape(quantized, (-1, *((5,5,32))))
                #     recons_img = self.vq_decoder.predict(quantized)
                #     recons_tensor = torch.tensor(recons_img).to(self.args.device).view(m.size(0),img_load_num*128,128,4)
                #     recons_loss = self.mse(recons_tensor,img)*self.img_weight#.cpu().item()
                recons_loss = self.VecsRLoss(self.img_data[idx*self.batch_size:(idx+1)*self.batch_size,0], out_T, out_R, self.args.device)
                cum_recons_loss += recons_loss.cpu().item()

                loss = loss_T + loss_S + loss_L + loss_A + loss_R + loss_W + loss_CD + recons_loss
            
            else:

                loss = loss_T + loss_S + loss_L + loss_A + loss_R + loss_W + loss_CD

            self.optim.zero_grad()

            self.scaler.scale(loss).backward()  # rescale to get more precise loss

            self.scaler.unscale_(self.optim)                      # rescale loss
            nn.utils.clip_grad_norm_(self.vit.parameters(), 1.0)  # Clip gradient
            self.scaler.step(self.optim)
            self.scaler.update()

            cum_loss += loss.cpu().item()
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
        # gen_sample = self.sample(nb_sample=10)[0]
        # gen_sample = vutils.make_grid(gen_sample, nrow=10, padding=2, normalize=True)
        # self.log_add_img("Images/Sampling", gen_sample, self.args.iter)
        # # Show reconstruction
        # unmasked_code = torch.softmax(pred, -1).max(-1)[1]
        # reco_sample = self.reco(x=x[:10], code=code[:10], unmasked_code=unmasked_code[:10], mask=mask[:10])
        # reco_sample = vutils.make_grid(reco_sample.data, nrow=10, padding=2, normalize=True)
        # self.log_add_img("Images/Reconstruction", reco_sample, self.args.iter)

        if self.args.img_loss: return cum_loss / n, CD_loss / n, T_loss / n, S_loss / n, L_loss / n, A_loss / n, R_loss / n, W_loss / n, cum_recons_loss / n
        else: return cum_loss / n, CD_loss / n, T_loss / n, S_loss / n, L_loss / n, A_loss / n, R_loss / n, W_loss / n, None

    def fit(self):
        """ Train the model """
        if self.args.is_master:
            print("Start training:")

        start = time.time()
        # Start training
        for e in range(self.args.global_epoch, self.args.epoch):
            # synch every GPUs
            if self.args.is_multi_gpus:
                self.train_data.sampler.set_epoch(e)

            # Train for one epoch
            sum_loss, CD_loss, T_loss, S_loss, L_loss, A_loss, R_loss, W_loss, recons_loss = self.train_one_epoch(self.args.save_iter)

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
                        f" CD {CD_loss:.4f},"
                        f" T {T_loss:.4f},"
                        f" S {S_loss:.4f},"
                        f" L {L_loss:.4f},"
                        f" A {A_loss:.4f},"
                        f" R {R_loss:.4f},"
                        f" W {W_loss:.4f},"
                        f" Recons Loss {recons_loss:.4f},"
                        f" Time: {clock_time // 3600:.0f}h {(clock_time % 3600) // 60:.0f}min {clock_time % 60:.2f}s")
                else:
                    print(f"\rEpoch {self.args.global_epoch},"
                        f" Iter {self.args.iter :},"
                        f" SUM {sum_loss:.4f},"
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

    def sample(self, nb_sample=10, sm_temp=1, w=0,
               randomize="linear", r_temp=4.5, sched_mode="arccos", step=10):
        """ Generate sample with the MaskGIT model
           :param
            init_code   -> torch.LongTensor: nb_sample x 16 x 16, the starting initialization code
            nb_sample   -> int:              the number of image to generated
            sm_temp     -> float:            the temperature before softmax
            w           -> float:            scale for the classifier free guidance
            randomize   -> str:              linear|warm_up|random|no, either or not to add randomness
            r_temp      -> float:            temperature for the randomness
            sched_mode  -> str:              root|linear|square|cosine|arccos, the shape of the scheduler
            step:       -> int:              number of step for the decoding
           :return
            x          -> torch.FloatTensor: nb_sample x 3 x 256 x 256, the generated images
            code       -> torch.LongTensor:  nb_sample x step x 16 x 16, the code corresponding to the generated images
        """
        picked = np.random.choice(1, self.test_len, replace=False)
        self.vit.eval()

        l_codes = []  # Save the intermediate codes predicted
        l_mask = []   # Save the intermediate masks

        with torch.no_grad():

            _, y_t, m_t, _ = self.test_data[picked]

            y_t = y_t.to(self.args.device)
            m_t = m_t.to(self.args.device)

            mask = (m_t == int(self.codebook_size-1)).float().view(nb_sample, self.attri_count*self.sqe_len, self.codebook_shape)[:,:,0]

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

                    logit = self.vit(m_t.clone(), y_t)
                        
                prob = torch.softmax(logit * sm_temp, -1)
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
                conf = (conf >= tresh_conf.unsqueeze(-1)).view(nb_sample, self.sqe_len, self.attri_count)
                f_mask = (mask.view(nb_sample, self.sqe_len, self.attri_count).float() * conf.view(nb_sample, self.sqe_len, self.attri_count).float()).bool()
                m_t[f_mask] = pred_code.view(nb_sample, self.sqe_len, self.attri_count, self.codebook_shape)[f_mask]

                # update the mask
                for i_mask, ind_mask in enumerate(indice_mask):
                    mask[i_mask, ind_mask] = 0
                l_codes.append(pred_code.view(nb_sample, self.sqe_len, self.attri_count, self.codebook_shape).clone())
                l_mask.append(mask.view(nb_sample, self.sqe_len, self.attri_count).clone())

            # decode the final prediction
            # _code = torch.clamp(m_t, 0,  self.codebook_size-1)
            # x = self.ae.decode_code(_code)

        self.vit.train()
        return l_codes, l_mask#, x
