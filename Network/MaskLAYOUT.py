import torch
from torch import nn
import math
import torch.nn.functional as F

class MaskTransformer(nn.Module):
    def __init__(self, sqe_len=14, attri_count=6, hidden_dim=768, codebook_shape=16, codebook_size=32, depth=24, heads=8, mlp_dim=3072, dropout=0.1):
        """ Initialize the Transformer model.
            :param:
                hidden_dim     -> int:     Hidden dimension for the transformer (default: 768)
                codebook_size  -> int:     Size of the codebook (default: 32)
                depth          -> int:     Depth of the transformer (default: 24)
                heads          -> int:     Number of attention heads (default: 8)
                mlp_dim        -> int:     MLP dimension (default: 3072)
                dropout        -> float:   Dropout rate (default: 0.1)
        """

        super().__init__()
        self.sqe_len = sqe_len
        self.attri_count = attri_count
        # self.v_T_dim = v_T_dim # 8 one-hot
        # self.v_L_dim = v_L_dim # 7 bit
        # self.v_S_dim = v_S_dim # 6 bit

        self.codebook_shape = codebook_shape
        self.codebook_size = codebook_size
        self.img_emb = nn.Embedding(codebook_size, hidden_dim)
        self.pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, (self.sqe_len*self.attri_count)+1,self.codebook_shape, hidden_dim)), 0., 0.02)
        
        # First layer before the Transformer block
        self.first_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim/self.codebook_shape)),
            nn.Flatten(-2,-1),
        )

        self.transformer = TransformerEncoder(dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)

        # Last layer after the Transformer block
        self.last_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim*self.codebook_shape)),
            nn.Unflatten(2, (self.codebook_shape, hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
        )

        # Bias for the last linear output
        self.bias = nn.Parameter(torch.zeros((self.sqe_len*self.attri_count)+1, self.codebook_shape, self.codebook_size))

    def forward(self, attri_token, site=None, return_attn=False):
        """ Forward.
            :param:
                attri_token      -> torch.LongTensor: bsize x 6 x 14 x 25, the encoded image tokens
                site           -> torch.LongTensor: bsize x 25, site condition 
                return_attn    -> Bool: return the attn for visualization
            :return:
                logit:         -> torch.FloatTensor: bsize x 6 x 14 x 25 * 32, the predicted logit
                attn:          -> list(torch.FloatTensor): list of attention for visualization
        """
        b, w, h, d = attri_token.size()

        cls_token = site

        input = torch.cat([attri_token.view(b, w*h, d), cls_token.view(b, 1, d)], -2)  # concat visual tokens and class tokens
        tok_embeddings = self.img_emb(input)

        # Position embedding
        pos_embeddings = self.pos_emb
        x = tok_embeddings + pos_embeddings

        # transformer forward pass
        x = self.first_layer(x)
        x, attn = self.transformer(x)
        x = self.last_layer(x)

        logit = torch.matmul(x, self.img_emb.weight.T) + self.bias   # Shared layer with the embedding

        if return_attn:  # return list of attention
            return logit[:, :self.sqe_len*self.attri_count, :self.codebook_shape, :self.codebook_size], attn

        return logit[:, :self.sqe_len*self.attri_count, :self.codebook_shape, :self.codebook_size]

class MaskTransformer_IMG(nn.Module):
    def __init__(self, sqe_len=14, attri_count=6, hidden_dim=768, codebook_shape=25, codebook_size=32, depth=24, heads=8, mlp_dim=3072, dropout=0.1):
        """ Initialize the Transformer model.
            :param:
                hidden_dim     -> int:     Hidden dimension for the transformer (default: 768)
                codebook_size  -> int:     Size of the codebook (default: 32)
                depth          -> int:     Depth of the transformer (default: 24)
                heads          -> int:     Number of attention heads (default: 8)
                mlp_dim        -> int:     MLP dimension (default: 3072)
                dropout        -> float:   Dropout rate (default: 0.1)
        """

        super().__init__()
        self.sqe_len = sqe_len
        self.attri_count = attri_count
        # self.v_T_dim = v_T_dim # 8 one-hot
        # self.v_L_dim = v_L_dim # 7 bit
        # self.v_S_dim = v_S_dim # 6 bit

        self.codebook_shape = codebook_shape
        self.codebook_size = codebook_size
        self.img_emb = nn.Embedding(codebook_size, hidden_dim)
        self.pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, (self.sqe_len*self.attri_count)+1,self.codebook_shape, hidden_dim)), 0., 0.02)
        
        # First layer before the Transformer block
        self.first_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim/self.codebook_shape)),
            nn.Flatten(-2,-1),
        )

        self.transformer = TransformerEncoder(dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)

        # Last layer after the Transformer block
        self.last_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim*self.codebook_shape)),
            nn.Unflatten(2, (self.codebook_shape, hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
        )

        # Bias for the last linear output
        self.bias = nn.Parameter(torch.zeros((self.sqe_len*self.attri_count)+1, self.codebook_shape, self.codebook_size))

    def forward(self, attri_token, site=None, return_attn=False):
        """ Forward.
            :param:
                attri_token      -> torch.LongTensor: bsize x 6 x 14 x 25, the encoded image tokens
                site           -> torch.LongTensor: bsize x 25, site condition 
                return_attn    -> Bool: return the attn for visualization
            :return:
                logit:         -> torch.FloatTensor: bsize x 6 x 14 x 25 * 32, the predicted logit
                attn:          -> list(torch.FloatTensor): list of attention for visualization
        """
        b, w, h, d = attri_token.size()

        cls_token = site

        input = torch.cat([attri_token.view(b, w*h, d), cls_token.view(b, 1, d)], -2)  # concat visual tokens and class tokens
        tok_embeddings = self.img_emb(input)

        # Position embedding
        pos_embeddings = self.pos_emb
        x = tok_embeddings + pos_embeddings

        # transformer forward pass
        x = self.first_layer(x)
        x, attn = self.transformer(x)
        x = self.last_layer(x)

        logit = torch.matmul(x, self.img_emb.weight.T) + self.bias   # Shared layer with the embedding

        if return_attn:  # return list of attention
            return logit[:, :self.sqe_len*self.attri_count, :self.codebook_shape, :self.codebook_size], attn

        return logit[:, :self.sqe_len*self.attri_count, :self.codebook_shape, :self.codebook_size]

class MaskTransformer_VEC(nn.Module):
    def __init__(self, v_T_dim, v_L_dim, v_S_dim, sqe_len=14, attri_count=6, hidden_dim=768, codebook_shape=25, codebook_size=32, depth=24, heads=8, mlp_dim=3072, dropout=0.1):
        """ Initialize the Transformer model.
            :param:
                hidden_dim     -> int:     Hidden dimension for the transformer (default: 768)
                codebook_size  -> int:     Size of the codebook (default: 32)
                depth          -> int:     Depth of the transformer (default: 24)
                heads          -> int:     Number of attention heads (default: 8)
                mlp_dim        -> int:     MLP dimension (default: 3072)
                dropout        -> float:   Dropout rate (default: 0.1)
        """

        super().__init__()
        self.sqe_len = sqe_len
        self.attri_count = attri_count
 
        self.v_T_dim = v_T_dim # 8 one-hot
        self.v_L_dim = v_L_dim # 7 bit
        self.v_S_dim = v_S_dim # 6 bit
        self.hidden_dim = hidden_dim

        self.codebook_shape = codebook_shape
        self.codebook_size = codebook_size

        self.pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, (self.sqe_len*self.attri_count)+1, hidden_dim)), 0., 0.02)
        
        # encode
        self.encode_T = nn.Sequential(
            nn.Linear(in_features=v_T_dim, out_features=hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout), nn.GELU())

        self.encode_S = nn.Sequential(
            nn.Linear(in_features=v_S_dim, out_features=hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout), nn.GELU())
        
        self.encode_L = nn.Sequential(
            nn.Linear(in_features=v_L_dim, out_features=int(hidden_dim/2)),
            nn.LayerNorm(int(hidden_dim/2), eps=1e-12),
            nn.Dropout(p=dropout), nn.GELU(),
            nn.Flatten(-2,-1))

        self.encode_A = nn.Sequential(
            nn.Linear(in_features=sqe_len, out_features=hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout), nn.GELU())
        
        self.encode_R = nn.Sequential(
            nn.Linear(in_features=v_L_dim, out_features=int(hidden_dim/2)),
            nn.LayerNorm(int(hidden_dim/2), eps=1e-12),
            nn.Dropout(p=dropout), nn.GELU(),nn.Flatten(-2,-1),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim/20)),
            nn.LayerNorm(int(hidden_dim/20), eps=1e-12),
            nn.Dropout(p=dropout), nn.GELU(),nn.Flatten(-2,-1))
        
        self.encode_W = nn.Sequential(
            nn.Linear(in_features=v_L_dim, out_features=int(hidden_dim/2)),
            nn.LayerNorm(int(hidden_dim/2), eps=1e-12),
            nn.Dropout(p=dropout), nn.GELU(),
            nn.Flatten(-2,-1))
        
        self.encode_cond = nn.Sequential(
            nn.Embedding(codebook_size, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.GELU(), nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim/self.codebook_shape)), nn.Flatten(-2,-1))
        
        # transformer
        self.transformer = TransformerEncoder_Basic(dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)

        # decode
        self.decode_T = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=self.v_T_dim),
            nn.LayerNorm(self.v_T_dim, eps=1e-12), nn.Sigmoid())
        
        self.decode_S = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=self.v_S_dim),
            nn.LayerNorm(self.v_S_dim, eps=1e-12), nn.Sigmoid())

        self.decode_L = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim*2)),
            nn.Unflatten(2, (2, hidden_dim)), nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=self.v_L_dim),
            nn.LayerNorm(self.v_L_dim, eps=1e-12), nn.Sigmoid())
        
        self.decode_A = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=self.sqe_len),
            nn.LayerNorm(self.sqe_len, eps=1e-12), nn.Sigmoid())
        
        self.decode_R = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim*20)),
            nn.Unflatten(2, (20, hidden_dim)), nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim*2)),
            nn.Unflatten(3, (2, hidden_dim)), nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=self.v_L_dim),
            nn.LayerNorm(self.v_L_dim, eps=1e-12), nn.Sigmoid())
        
        self.decode_W = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim*2)),
            nn.Unflatten(2, (2, hidden_dim)), nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=self.v_L_dim),
            nn.LayerNorm(self.v_L_dim, eps=1e-12), nn.Sigmoid())

        # Bias 
        # self.bias_T = nn.Parameter(torch.zeros(self.sqe_len, self.v_T_dim))
        # self.bias_S = nn.Parameter(torch.zeros(self.sqe_len, self.v_S_dim))
        # self.bias_L = nn.Parameter(torch.zeros(self.sqe_len, 2, self.v_L_dim))
        # self.bias_A = nn.Parameter(torch.zeros(self.sqe_len, self.sqe_len))
        # self.bias_R = nn.Parameter(torch.zeros(self.sqe_len, 20, 2, self.v_L_dim))
        # self.bias_W = nn.Parameter(torch.zeros(self.sqe_len, 2, self.v_L_dim))

    def forward(self, T, S, L, A, R, W, site=None, return_attn=False):
        
        b, l, d = T.size()


        emb_T = self.encode_T(T)
        emb_S = self.encode_S(S)
        emb_L = self.encode_L(L)
        emb_A = self.encode_A(A)
        emb_R = self.encode_R(R)
        emb_W = self.encode_W(W)
        emb_cls = self.encode_cond(site)

        input = torch.cat([emb_T, emb_S, emb_L, emb_A, emb_R, emb_W, emb_cls.view(b,1,self.hidden_dim)], -2) 

        # Position embedding
        pos_embeddings = self.pos_emb
        x = input + pos_embeddings

        # transformer forward pass
        x, attn = self.transformer(x)

        out_T = self.decode_T(x[:,:l])
        out_S = self.decode_S(x[:,l:2*l])
        out_L = self.decode_L(x[:,2*l:3*l])
        out_A = self.decode_A(x[:,3*l:4*l])
        out_R = self.decode_R(x[:,4*l:5*l])
        out_W = self.decode_W(x[:,5*l:6*l])

        if return_attn:  # return list of attention
            return out_T, out_S, out_L, out_A, out_R, out_W, attn

        return out_T, out_S, out_L, out_A, out_R, out_W

class Fusion_Layer(nn.Module):
    def __init__(self,hidden_dim):
        super(Fusion_Layer, self).__init__()
        self.Bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
    
    def forward(self, input):
        x1, x2 = input

        return self.Bilinear(x1,x2)

class MaskTransformer_HYB2(nn.Module):
    def __init__(self, v_T_dim, v_L_dim, v_S_dim, sqe_len=14, attri_count=6, hidden_dim=768, codebook_shape=25, codebook_size=32, depth=24, heads=8, mlp_dim=3072, dropout=0.1):
        """ Initialize the Transformer model.
            :param:
                hidden_dim     -> int:     Hidden dimension for the transformer (default: 768)
                codebook_size  -> int:     Size of the codebook (default: 32)
                depth          -> int:     Depth of the transformer (default: 24)
                heads          -> int:     Number of attention heads (default: 8)
                mlp_dim        -> int:     MLP dimension (default: 3072)
                dropout        -> float:   Dropout rate (default: 0.1)
        """

        super().__init__()
        self.sqe_len = sqe_len
        self.attri_count = attri_count
 
        self.v_T_dim = v_T_dim # 8 one-hot
        self.v_L_dim = v_L_dim # 7 bit
        self.v_S_dim = v_S_dim # 6 bit
        self.hidden_dim = hidden_dim

        self.codebook_shape = codebook_shape
        self.codebook_size = codebook_size

        self.vec_pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, (self.sqe_len*self.attri_count)+1, hidden_dim)), 0., 0.02)
        # self.img_pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, (self.sqe_len*self.attri_count)+1, hidden_dim)), 0., 0.02)#self.codebook_shape

        # encode
        self.encode_T = nn.Sequential(
            nn.Linear(in_features=v_T_dim, out_features=hidden_dim),
            nn.GELU(), nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout))

        self.encode_S = nn.Sequential(
            nn.Linear(in_features=v_S_dim, out_features=hidden_dim),
            nn.GELU(), nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout))
        
        self.encode_L = nn.Sequential(
            nn.Linear(in_features=v_L_dim, out_features=int(hidden_dim/2)),
            nn.GELU(), nn.LayerNorm(int(hidden_dim/2), eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Flatten(-2,-1))

        self.encode_A = nn.Sequential(
            nn.Linear(in_features=sqe_len, out_features=hidden_dim),
            nn.GELU(), nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout))
        
        self.encode_R = nn.Sequential(
            nn.Linear(in_features=v_L_dim, out_features=int(hidden_dim/2)),
            nn.LayerNorm(int(hidden_dim/2), eps=1e-12),
            nn.Dropout(p=dropout), nn.GELU(),nn.Flatten(-2,-1),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim/20)),
            nn.GELU(), nn.LayerNorm(int(hidden_dim/20), eps=1e-12),
            nn.Dropout(p=dropout),nn.Flatten(-2,-1))
        
        self.encode_W = nn.Sequential(
            nn.Linear(in_features=v_L_dim, out_features=int(hidden_dim/2)),
            nn.GELU(), nn.LayerNorm(int(hidden_dim/2), eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Flatten(-2,-1))
        
        self.encode_cond = nn.Sequential(
            nn.Embedding(codebook_size, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim/self.codebook_shape)),
            nn.GELU(), nn.LayerNorm(int(hidden_dim/self.codebook_shape), eps=1e-12), nn.Dropout(p=dropout), nn.Flatten(-2,-1))
        
        self.encode_img = nn.Sequential(
            nn.Embedding(codebook_size, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim/self.codebook_shape)),
            nn.GELU(), nn.LayerNorm(int(hidden_dim/self.codebook_shape), eps=1e-12), nn.Dropout(p=dropout), nn.Flatten(-2,-1))
        

        # transformer
        self.transformer = TransformerEncoder_Basic(dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)

        # decode
        self.decode_T = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=self.v_T_dim),
            nn.LayerNorm(self.v_T_dim, eps=1e-12), nn.Sigmoid())
        
        self.decode_S = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=self.v_S_dim),
            nn.LayerNorm(self.v_S_dim, eps=1e-12), nn.Sigmoid())

        self.decode_L = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim*2)),
            nn.Unflatten(2, (2, hidden_dim)), nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=self.v_L_dim),
            nn.LayerNorm(self.v_L_dim, eps=1e-12), nn.Sigmoid())
        
        self.decode_A = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=self.sqe_len),
            nn.LayerNorm(self.sqe_len, eps=1e-12), nn.Sigmoid())
        
        self.decode_R = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim*20)),
            nn.Unflatten(2, (20, hidden_dim)), nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim*2)),
            nn.Unflatten(3, (2, hidden_dim)), nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=self.v_L_dim),
            nn.LayerNorm(self.v_L_dim, eps=1e-12), nn.Sigmoid())
        
        self.decode_W = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim*2)),
            nn.Unflatten(2, (2, hidden_dim)), nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=self.v_L_dim),
            nn.LayerNorm(self.v_L_dim, eps=1e-12), nn.Sigmoid())
        
        self.decode_img = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(self.codebook_size*self.codebook_shape)),
            nn.Unflatten(2, (self.codebook_shape, self.codebook_size)),
            nn.GELU(), nn.LayerNorm(self.codebook_size, eps=1e-12))
        
        self.modality_fusion = nn.Sequential(
            nn.Linear(in_features=self.codebook_size, out_features=int(hidden_dim/self.codebook_shape)),
            nn.GELU(), nn.LayerNorm(int(hidden_dim/self.codebook_shape), eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Flatten(-2,-1)
            )

        # Bias 
        # self.bias_T = nn.Parameter(torch.zeros(self.sqe_len, self.v_T_dim))
        # self.bias_S = nn.Parameter(torch.zeros(self.sqe_len, self.v_S_dim))
        # self.bias_L = nn.Parameter(torch.zeros(self.sqe_len, 2, self.v_L_dim))
        # self.bias_A = nn.Parameter(torch.zeros(self.sqe_len, self.sqe_len))
        # self.bias_R = nn.Parameter(torch.zeros(self.sqe_len, 20, 2, self.v_L_dim))
        # self.bias_W = nn.Parameter(torch.zeros(self.sqe_len, 2, self.v_L_dim))

    def forward(self, attri_token, T, S, L, A, R, W, site=None, return_attn=False):
        
        b, w, h, d = attri_token.size()
        _, l, _ = T.size()

        emb_img = self.encode_img(torch.cat([attri_token.view(b, w*h, d), site.view(b, 1, d)], -2))

        emb_T = self.encode_T(T)
        emb_S = self.encode_S(S)
        emb_L = self.encode_L(L)
        emb_A = self.encode_A(A)
        emb_R = self.encode_R(R)
        emb_W = self.encode_W(W)
        emb_cls = self.encode_cond(site)

        vec_input = torch.cat([emb_T, emb_S, emb_L, emb_A, emb_R, emb_W, emb_cls.view(b,1,self.hidden_dim)], -2) 

        # Position embedding
        # x_vec = self.vec_pos_emb + vec_input
        # x_img = self.vec_pos_emb + emb_img

        # moality Fusion
        # fused = self.modality_fusion((x_vec,x_img))
        fused = self.vec_pos_emb + vec_input + emb_img

        # transformer forward pass
        out, attn = self.transformer(fused)

        # decode img
        out_img = self.decode_img(out)

        # modality_fusion
        f = self.modality_fusion(out_img)
        x = out + f

        # decode vec
        out_T = self.decode_T(x[:,:l])
        out_S = self.decode_S(x[:,l:2*l])
        out_L = self.decode_L(x[:,2*l:3*l])
        out_A = self.decode_A(x[:,3*l:4*l])
        out_R = self.decode_R(x[:,4*l:5*l])
        out_W = self.decode_W(x[:,5*l:6*l])

        if return_attn:  # return list of attention
            return out_img[:, :self.sqe_len*self.attri_count], out_T, out_S, out_L, out_A, out_R, out_W, attn

        return out_img[:, :self.sqe_len*self.attri_count], out_T, out_S, out_L, out_A, out_R, out_W

class MaskTransformer_HYB(nn.Module):
    def __init__(self, v_T_dim, v_L_dim, v_S_dim, sqe_len=14, attri_count=6, hidden_dim=768, codebook_shape=25, codebook_size=32, depth=24, heads=8, mlp_dim=3072, dropout=0.1):
        """ Initialize the Transformer model.
            :param:
                hidden_dim     -> int:     Hidden dimension for the transformer (default: 768)
                codebook_size  -> int:     Size of the codebook (default: 32)
                depth          -> int:     Depth of the transformer (default: 24)
                heads          -> int:     Number of attention heads (default: 8)
                mlp_dim        -> int:     MLP dimension (default: 3072)
                dropout        -> float:   Dropout rate (default: 0.1)
        """

        super().__init__()
        self.sqe_len = sqe_len
        self.attri_count = attri_count
 
        self.v_T_dim = v_T_dim # 8 one-hot
        self.v_L_dim = v_L_dim # 7 bit
        self.v_S_dim = v_S_dim # 6 bit
        self.hidden_dim = hidden_dim

        self.codebook_shape = codebook_shape
        self.codebook_size = codebook_size

        self.vec_pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, (self.sqe_len*self.attri_count)+1, hidden_dim)), 0., 0.02)
        # self.img_pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, (self.sqe_len*self.attri_count)+1, hidden_dim)), 0., 0.02)#self.codebook_shape

        # encode
        self.encode_T = nn.Sequential(
            nn.Linear(in_features=v_T_dim, out_features=hidden_dim),
            nn.GELU(), nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout))

        self.encode_S = nn.Sequential(
            nn.Linear(in_features=v_S_dim, out_features=hidden_dim),
            nn.GELU(), nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout))
        
        self.encode_L = nn.Sequential(
            nn.Linear(in_features=v_L_dim, out_features=int(hidden_dim/2)),
            nn.GELU(), nn.LayerNorm(int(hidden_dim/2), eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Flatten(-2,-1))

        self.encode_A = nn.Sequential(
            nn.Linear(in_features=sqe_len, out_features=hidden_dim),
            nn.GELU(), nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout))
        
        self.encode_R = nn.Sequential(
            nn.Linear(in_features=v_L_dim, out_features=int(hidden_dim/2)),
            nn.LayerNorm(int(hidden_dim/2), eps=1e-12),
            nn.Dropout(p=dropout), nn.GELU(),nn.Flatten(-2,-1),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim/20)),
            nn.GELU(), nn.LayerNorm(int(hidden_dim/20), eps=1e-12),
            nn.Dropout(p=dropout),nn.Flatten(-2,-1))
        
        self.encode_W = nn.Sequential(
            nn.Linear(in_features=v_L_dim, out_features=int(hidden_dim/2)),
            nn.GELU(), nn.LayerNorm(int(hidden_dim/2), eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Flatten(-2,-1))
        
        self.encode_cond = nn.Sequential(
            nn.Embedding(codebook_size, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim/self.codebook_shape)),
            nn.GELU(), nn.LayerNorm(int(hidden_dim/self.codebook_shape), eps=1e-12), nn.Dropout(p=dropout), nn.Flatten(-2,-1))
        
        self.encode_img = nn.Sequential(
            nn.Embedding(codebook_size, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim/self.codebook_shape)),
            nn.GELU(), nn.LayerNorm(int(hidden_dim/self.codebook_shape), eps=1e-12), nn.Dropout(p=dropout), nn.Flatten(-2,-1))
        
        self.modality_fusion = nn.Sequential(
            Fusion_Layer(hidden_dim),
            nn.GELU(), nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            )

        # transformer
        self.transformer = TransformerEncoder(dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)

        # decode
        self.decode_T = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=self.v_T_dim),
            nn.LayerNorm(self.v_T_dim, eps=1e-12), nn.Sigmoid())
        
        self.decode_S = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=self.v_S_dim),
            nn.LayerNorm(self.v_S_dim, eps=1e-12), nn.Sigmoid())

        self.decode_L = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim*2)),
            nn.Unflatten(2, (2, hidden_dim)), nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=self.v_L_dim),
            nn.LayerNorm(self.v_L_dim, eps=1e-12), nn.Sigmoid())
        
        self.decode_A = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=self.sqe_len),
            nn.LayerNorm(self.sqe_len, eps=1e-12), nn.Sigmoid())
        
        self.decode_R = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim*20)),
            nn.Unflatten(2, (20, hidden_dim)), nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim*2)),
            nn.Unflatten(3, (2, hidden_dim)), nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=self.v_L_dim),
            nn.LayerNorm(self.v_L_dim, eps=1e-12), nn.Sigmoid())
        
        self.decode_W = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim*2)),
            nn.Unflatten(2, (2, hidden_dim)), nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=self.v_L_dim),
            nn.LayerNorm(self.v_L_dim, eps=1e-12), nn.Sigmoid())
        
        self.decode_img = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim*self.codebook_shape)),
            nn.Unflatten(2, (self.codebook_shape, hidden_dim)),
            nn.GELU(), nn.LayerNorm(hidden_dim, eps=1e-12))

        # Bias 
        # self.bias_T = nn.Parameter(torch.zeros(self.sqe_len, self.v_T_dim))
        # self.bias_S = nn.Parameter(torch.zeros(self.sqe_len, self.v_S_dim))
        # self.bias_L = nn.Parameter(torch.zeros(self.sqe_len, 2, self.v_L_dim))
        # self.bias_A = nn.Parameter(torch.zeros(self.sqe_len, self.sqe_len))
        # self.bias_R = nn.Parameter(torch.zeros(self.sqe_len, 20, 2, self.v_L_dim))
        # self.bias_W = nn.Parameter(torch.zeros(self.sqe_len, 2, self.v_L_dim))

    def forward(self, attri_token, T, S, L, A, R, W, site=None, return_attn=False):
        
        b, w, h, d = attri_token.size()
        _, l, _ = T.size()

        emb_img = self.encode_img(torch.cat([attri_token.view(b, w*h, d), site.view(b, 1, d)], -2))

        emb_T = self.encode_T(T)
        emb_S = self.encode_S(S)
        emb_L = self.encode_L(L)
        emb_A = self.encode_A(A)
        emb_R = self.encode_R(R)
        emb_W = self.encode_W(W)
        emb_cls = self.encode_cond(site)

        vec_input = torch.cat([emb_T, emb_S, emb_L, emb_A, emb_R, emb_W, emb_cls.view(b,1,self.hidden_dim)], -2) 

        # Position embedding
        # x_vec = self.vec_pos_emb + vec_input
        # x_img = self.vec_pos_emb + emb_img

        # moality Fusion
        # fused = self.modality_fusion((x_vec,x_img))
        fused = self.vec_pos_emb + vec_input + emb_img

        # transformer forward pass
        x, attn = self.transformer(fused)

        out_img = self.decode_img(x)

        out_T = self.decode_T(x[:,:l])
        out_S = self.decode_S(x[:,l:2*l])
        out_L = self.decode_L(x[:,2*l:3*l])
        out_A = self.decode_A(x[:,3*l:4*l])
        out_R = self.decode_R(x[:,4*l:5*l])
        out_W = self.decode_W(x[:,5*l:6*l])

        if return_attn:  # return list of attention
            return out_img[:, :self.sqe_len*self.attri_count, :self.codebook_shape, :self.codebook_size], out_T, out_S, out_L, out_A, out_R, out_W, attn

        return out_img[:, :self.sqe_len*self.attri_count, :self.codebook_shape, :self.codebook_size], out_T, out_S, out_L, out_A, out_R, out_W

class MaskTransformer_HYB(nn.Module):
    def __init__(self, v_T_dim, v_L_dim, v_S_dim, R_num, sqe_len=14, attri_count=6, hidden_dim=768, codebook_shape=16, codebook_size=32, depth=24, heads=8, mlp_dim=3072, dropout=0.1):
        """ Initialize the Transformer model.
            :param:
                hidden_dim     -> int:     Hidden dimension for the transformer (default: 768)
                codebook_size  -> int:     Size of the codebook (default: 32)
                depth          -> int:     Depth of the transformer (default: 24)
                heads          -> int:     Number of attention heads (default: 8)
                mlp_dim        -> int:     MLP dimension (default: 3072)
                dropout        -> float:   Dropout rate (default: 0.1)
        """

        super().__init__()
        self.sqe_len = sqe_len
        self.attri_count = attri_count
 
        self.v_T_dim = v_T_dim # 8 one-hot
        self.v_L_dim = v_L_dim # 7 bit
        self.v_S_dim = v_S_dim # 6 bit
        self.R_num = R_num # 10 room corners
        self.hidden_dim = hidden_dim

        self.codebook_shape = codebook_shape
        self.codebook_size = codebook_size

        self.vec_pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, (self.sqe_len*self.attri_count)+1, hidden_dim)), 0., 0.02)

        # encode
        self.encode_T = nn.Sequential(
            nn.Linear(in_features=v_T_dim, out_features=hidden_dim),
            nn.GELU(), nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout))

        self.encode_S = nn.Sequential(
            nn.Linear(in_features=v_S_dim, out_features=hidden_dim),
            nn.GELU(), nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout))
        
        self.encode_L = nn.Sequential(
            nn.Linear(in_features=v_L_dim, out_features=int(hidden_dim/2)),
            nn.GELU(), nn.LayerNorm(int(hidden_dim/2), eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Flatten(-2,-1))

        self.encode_A = nn.Sequential(
            nn.Linear(in_features=sqe_len, out_features=hidden_dim),
            nn.GELU(), nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout))
        
        self.encode_R = nn.Sequential(
            nn.Linear(in_features=v_L_dim, out_features=int(hidden_dim/2)),
            nn.LayerNorm(int(hidden_dim/2), eps=1e-12),
            nn.Dropout(p=dropout), nn.GELU(),nn.Flatten(-2,-1),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim/self.R_num)),
            nn.GELU(), nn.LayerNorm(int(hidden_dim/self.R_num), eps=1e-12),
            nn.Dropout(p=dropout),nn.Flatten(-2,-1))
        
        self.encode_W = nn.Sequential(
            nn.Linear(in_features=v_L_dim, out_features=int(hidden_dim/2)),
            nn.GELU(), nn.LayerNorm(int(hidden_dim/2), eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Flatten(-2,-1))
        
        self.encode_cond = nn.Sequential(
            nn.Embedding(codebook_size, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim/self.codebook_shape)),
            nn.GELU(), nn.LayerNorm(int(hidden_dim/self.codebook_shape), eps=1e-12), nn.Dropout(p=dropout), nn.Flatten(-2,-1))
        
        self.encode_img = nn.Sequential(
            nn.Embedding(codebook_size, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim/self.codebook_shape)),
            nn.GELU(), nn.LayerNorm(int(hidden_dim/self.codebook_shape), eps=1e-12), nn.Dropout(p=dropout), nn.Flatten(-2,-1))
        

        # transformer
        self.transformer = TransformerEncoder(dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)

        # decode
        self.decode_T = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=self.v_T_dim),
            nn.LayerNorm(self.v_T_dim, eps=1e-12), nn.Sigmoid())
        
        self.decode_S = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=self.v_S_dim),
            nn.LayerNorm(self.v_S_dim, eps=1e-12), nn.Sigmoid())

        self.decode_L = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim*2)),
            nn.Unflatten(2, (2, hidden_dim)), nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=self.v_L_dim),
            nn.LayerNorm(self.v_L_dim, eps=1e-12), nn.Sigmoid())
        
        self.decode_A = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=self.sqe_len),
            nn.LayerNorm(self.sqe_len, eps=1e-12), nn.Sigmoid())
        
        self.decode_R = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim*self.R_num)),
            nn.Unflatten(2, (self.R_num, hidden_dim)), nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim*2)),
            nn.Unflatten(3, (2, hidden_dim)), nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=self.v_L_dim),
            nn.LayerNorm(self.v_L_dim, eps=1e-12), nn.Sigmoid())
        
        self.decode_W = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim*2)),
            nn.Unflatten(2, (2, hidden_dim)), nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=self.v_L_dim),
            nn.LayerNorm(self.v_L_dim, eps=1e-12), nn.Sigmoid())
        
        self.decode_img = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(self.codebook_size*self.codebook_shape)),
            nn.Unflatten(2, (self.codebook_shape, self.codebook_size)),
            nn.GELU(), nn.LayerNorm(self.codebook_size, eps=1e-12))
        
        self.modality_fusion = nn.Sequential(
            nn.Linear(in_features=self.codebook_size, out_features=int(hidden_dim/self.codebook_shape)),
            nn.GELU(), nn.LayerNorm(int(hidden_dim/self.codebook_shape), eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Flatten(-2,-1)
            )
        
        self.cond_decode_R = nn.Sequential(
            nn.Linear(in_features=14*5,out_features=14),
            nn.GELU(), nn.LayerNorm(14, eps=1e-12),
            nn.Dropout(p=dropout))

    def forward(self, attri_token, T, S, L, A, R, W, site=None, return_attn=False):
        
        b, w, h, d = attri_token.size()
        _, l, _ = T.size()

        emb_img = self.encode_img(torch.cat([attri_token.view(b, w*h, d), site.view(b, 1, d)], -2))

        emb_T = self.encode_T(T)
        emb_S = self.encode_S(S)
        emb_L = self.encode_L(L)
        emb_A = self.encode_A(A)
        emb_R = self.encode_R(R)
        emb_W = self.encode_W(W)
        emb_cls = self.encode_cond(site)

        vec_input = torch.cat([emb_T, emb_S, emb_L, emb_A, emb_R, emb_W, emb_cls.view(b,1,self.hidden_dim)], -2) 

        # moality Fusion Encoder
        fused = self.vec_pos_emb + vec_input + emb_img

        # transformer forward pass
        out, l_att0, l_att1, l_att2, l_att3 = self.transformer(fused)

        # decode img
        out_img = self.decode_img(out)

        # moality Fusion Decoder
        f = self.modality_fusion(out_img)
        x = out + f

        # condition for decode R
        cond_R = self.cond_decode_R(torch.cat([x[:,:4*l], x[:,5*l:6*l]], 1).permute(0, 2, 1).contiguous()).permute(0, 1, 2).contiguous() + x[:,4*l:5*l]

        # decode vec
        out_T = self.decode_T(x[:,:l])
        out_S = self.decode_S(x[:,l:2*l])
        out_L = self.decode_L(x[:,2*l:3*l])
        out_A = self.decode_A(x[:,3*l:4*l])
        out_R = self.decode_R(cond_R)
        out_W = self.decode_W(x[:,5*l:6*l])

        if return_attn:  # return list of attention
            return out_img[:, :self.sqe_len*self.attri_count], out_T, out_S, out_L, out_A, out_R, out_W, l_att0, l_att1, l_att2, l_att3

        return out_img[:, :self.sqe_len*self.attri_count], out_T, out_S, out_L, out_A, out_R, out_W

class MaskTransformer_HYB_Graph(nn.Module):
    def __init__(self, v_T_dim, v_L_dim, v_S_dim, R_num, mode, sqe_len=14, attri_count=6, hidden_dim=768, codebook_shape=25, codebook_size=32, depth=24, heads=8, mlp_dim=3072, dropout=0.1):
        """ Initialize the Transformer model.
            :param:
                hidden_dim     -> int:     Hidden dimension for the transformer (default: 768)
                codebook_size  -> int:     Size of the codebook (default: 32)
                depth          -> int:     Depth of the transformer (default: 24)
                heads          -> int:     Number of attention heads (default: 8)
                mlp_dim        -> int:     MLP dimension (default: 3072)
                dropout        -> float:   Dropout rate (default: 0.1)
        """

        super().__init__()
        self.sqe_len = sqe_len
        self.attri_count = attri_count
 
        self.v_T_dim = v_T_dim # 8 one-hot
        self.v_L_dim = v_L_dim # 7 bit
        self.v_S_dim = v_S_dim # 6 bit
        self.R_num = R_num # 10 room corners
        self.hidden_dim = hidden_dim

        self.codebook_shape = codebook_shape
        self.codebook_size = codebook_size

        self.vec_pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, (self.sqe_len*self.attri_count)+1, hidden_dim)), 0., 0.02)
        # self.img_pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, (self.sqe_len*self.attri_count)+1, hidden_dim)), 0., 0.02)#self.codebook_shape

        # encode
        self.encode_T = nn.Sequential(
            nn.Linear(in_features=v_T_dim, out_features=hidden_dim),
            nn.GELU(), nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout))

        self.encode_S = nn.Sequential(
            nn.Linear(in_features=v_S_dim, out_features=hidden_dim),
            nn.GELU(), nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout))
        
        self.encode_L = nn.Sequential(
            nn.Linear(in_features=v_L_dim, out_features=int(hidden_dim/2)),
            nn.GELU(), nn.LayerNorm(int(hidden_dim/2), eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Flatten(-2,-1))

        self.encode_A = nn.Sequential(
            nn.Linear(in_features=sqe_len, out_features=hidden_dim),
            nn.GELU(), nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout))
        
        self.encode_R = nn.Sequential(
            nn.Linear(in_features=v_L_dim, out_features=int(hidden_dim/2)),
            nn.LayerNorm(int(hidden_dim/2), eps=1e-12),
            nn.Dropout(p=dropout), nn.GELU(),nn.Flatten(-2,-1),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim/self.R_num)),
            nn.GELU(), nn.LayerNorm(int(hidden_dim/self.R_num), eps=1e-12),
            nn.Dropout(p=dropout),nn.Flatten(-2,-1))
        
        self.encode_W = nn.Sequential(
            nn.Linear(in_features=v_L_dim, out_features=int(hidden_dim/2)),
            nn.GELU(), nn.LayerNorm(int(hidden_dim/2), eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Flatten(-2,-1))
        
        self.encode_cond = nn.Sequential(
            nn.Embedding(codebook_size, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim/self.codebook_shape)),
            nn.GELU(), nn.LayerNorm(int(hidden_dim/self.codebook_shape), eps=1e-12), nn.Dropout(p=dropout), nn.Flatten(-2,-1))
        
        self.encode_img = nn.Sequential(
            nn.Embedding(codebook_size, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim/self.codebook_shape)),
            nn.GELU(), nn.LayerNorm(int(hidden_dim/self.codebook_shape), eps=1e-12), nn.Dropout(p=dropout), nn.Flatten(-2,-1))
        

        # transformer
        self.transformer = TransformerEncoder(dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, mode=mode, dropout=dropout)

        # decode
        self.decode_T = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=self.v_T_dim),
            nn.LayerNorm(self.v_T_dim, eps=1e-12), nn.Sigmoid())
        
        self.decode_S = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=self.v_S_dim),
            nn.LayerNorm(self.v_S_dim, eps=1e-12), nn.Sigmoid())

        self.decode_L = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim*2)),
            nn.Unflatten(2, (2, hidden_dim)), nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=self.v_L_dim),
            nn.LayerNorm(self.v_L_dim, eps=1e-12), nn.Sigmoid())
        
        self.decode_A = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=self.sqe_len),
            nn.LayerNorm(self.sqe_len, eps=1e-12), nn.Sigmoid())
        
        self.decode_R = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim*self.R_num)),
            nn.Unflatten(2, (self.R_num, hidden_dim)), nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim*2)),
            nn.Unflatten(3, (2, hidden_dim)), nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=self.v_L_dim),
            nn.LayerNorm(self.v_L_dim, eps=1e-12), nn.Sigmoid())
        
        self.decode_W = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12), nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim*2)),
            nn.Unflatten(2, (2, hidden_dim)), nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=self.v_L_dim),
            nn.LayerNorm(self.v_L_dim, eps=1e-12), nn.Sigmoid())
        
        self.decode_img = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=int(self.codebook_size*self.codebook_shape)),
            nn.Unflatten(2, (self.codebook_shape, self.codebook_size)),
            nn.GELU(), nn.LayerNorm(self.codebook_size, eps=1e-12))
        
        self.modality_fusion = nn.Sequential(
            nn.Linear(in_features=self.codebook_size, out_features=int(hidden_dim/self.codebook_shape)),
            nn.GELU(), nn.LayerNorm(int(hidden_dim/self.codebook_shape), eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Flatten(-2,-1)
            )
        
        self.cond_decode_R = nn.Sequential(
            nn.Linear(in_features=14*5,out_features=14),
            nn.GELU(), nn.LayerNorm(14, eps=1e-12),
            nn.Dropout(p=dropout))

        # Bias 
        # self.bias_T = nn.Parameter(torch.zeros(self.sqe_len, self.v_T_dim))
        # self.bias_S = nn.Parameter(torch.zeros(self.sqe_len, self.v_S_dim))
        # self.bias_L = nn.Parameter(torch.zeros(self.sqe_len, 2, self.v_L_dim))
        # self.bias_A = nn.Parameter(torch.zeros(self.sqe_len, self.sqe_len))
        # self.bias_R = nn.Parameter(torch.zeros(self.sqe_len, 20, 2, self.v_L_dim))
        # self.bias_W = nn.Parameter(torch.zeros(self.sqe_len, 2, self.v_L_dim))

    def forward(self, attri_token, T, S, L, A, R, W, site, all_mask, graph_mask, attr_mask, room_mask, return_attn=False):
        
        b, w, h, d = attri_token.size()
        _, l, _ = T.size()

        emb_img = self.encode_img(torch.cat([attri_token.view(b, w*h, d), site.view(b, 1, d)], -2))

        emb_T = self.encode_T(T)
        emb_S = self.encode_S(S)
        emb_L = self.encode_L(L)
        emb_A = self.encode_A(A)
        emb_R = self.encode_R(R)
        emb_W = self.encode_W(W)
        emb_cls = self.encode_cond(site)

        vec_input = torch.cat([emb_T, emb_S, emb_L, emb_A, emb_R, emb_W, emb_cls.view(b,1,self.hidden_dim)], -2) 

        # Position embedding
        # x_vec = self.vec_pos_emb + vec_input
        # x_img = self.vec_pos_emb + emb_img

        # moality Fusion
        # fused = self.modality_fusion((x_vec,x_img))
        fused = self.vec_pos_emb + vec_input + emb_img

        # transformer forward pass
        out, l_att0, l_att1, l_att2, l_att3 = self.transformer(fused, all_mask, graph_mask, attr_mask, room_mask)

        # decode img
        out_img = self.decode_img(out)

        # moality Fusion Decoder
        f = self.modality_fusion(out_img)
        x = out + f

        # condition for decode R
        cond_R = self.cond_decode_R(torch.cat([x[:,:4*l], x[:,5*l:6*l]], 1).permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous() + x[:,4*l:5*l]

        # decode vec
        out_T = self.decode_T(x[:,:l])
        out_S = self.decode_S(x[:,l:2*l])
        out_L = self.decode_L(x[:,2*l:3*l])
        out_A = self.decode_A(x[:,3*l:4*l])
        out_R = self.decode_R(cond_R)
        out_W = self.decode_W(x[:,5*l:6*l])

        if return_attn:  # return list of attention
            return out_img[:, :self.sqe_len*self.attri_count], out_T, out_S, out_L, out_A, out_R, out_W, l_att0, l_att1, l_att2, l_att3

        return out_img[:, :self.sqe_len*self.attri_count], out_T, out_S, out_L, out_A, out_R, out_W

class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        """ PreNorm module to apply layer normalization before a given function
            :param:
                dim  -> int: Dimension of the input
                fn   -> nn.Module: The function to apply after layer normalization
            """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """ Forward pass through the PreNorm module
            :param:
                x        -> torch.Tensor: Input tensor
                **kwargs -> _ : Additional keyword arguments for the function
            :return
                torch.Tensor: Output of the function applied after layer normalization
        """
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        """ Initialize the Multi-Layer Perceptron (MLP).
            :param:
                dim        -> int : Dimension of the input
                dim        -> int : Dimension of the hidden layer
                dim        -> float : Dropout rate
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """ Forward pass through the MLP module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                torch.Tensor: Output of the function applied after layer
        """
        return self.net(x)


class Attention_Basic(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        """ Initialize the Attention module.
            :param:
                embed_dim     -> int : Dimension of the embedding
                num_heads     -> int : Number of heads
                dropout       -> float : Dropout rate
        """
        super(Attention_Basic, self).__init__()
        self.dim = embed_dim
        self.mha = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True, bias=True)

    def forward(self, x):
        """ Forward pass through the Attention module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                attention_value  -> torch.Tensor: Output the value of the attention
                attention_weight -> torch.Tensor: Output the weight of the attention
        """
        attention_value, attention_weight = self.mha(x, x, x)
        return attention_value, attention_weight

# class FeedForward(nn.Module):
#     def __init__(self, d_model, d_ff, dropout, activation):
#         super().__init__() 
#         # We set d_ff as a default to 2048
#         self.linear_1 = nn.Linear(d_model, d_ff)
#         self.dropout = nn.Dropout(dropout)
#         self.linear_2 = nn.Linear(d_ff, d_model)
#         self.activation = activation
#     def forward(self, x):
#         x = self.dropout(self.activation(self.linear_1(x)))
#         x = self.linear_2(x)
#         return x

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 1, -1e4)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)# calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output,scores
    
class EncoderLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, mode):
        super().__init__()
        self.norm_1 = nn.InstanceNorm1d(dim)
        self.norm_2 = nn.InstanceNorm1d(dim)
        self.all_attn = MultiHeadAttention(heads, dim)
        self.graph_attn = MultiHeadAttention(heads, dim)
        self.attri_attn = MultiHeadAttention(heads, dim)
        self.room_attn = MultiHeadAttention(heads, dim)
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.mode = mode
        
    def forward(self, x, all_mask, graph_mask, attr_mask, room_mask):
        # assert (gen_mask.max()==1 and gen_mask.min()==0), f"{gen_mask.max()}, {gen_mask.min()}"
        x2 = self.norm_1(x)

        if self.mode == 0: # complete
            att0,score0 = self.all_attn(x2,x2,x2,all_mask)
            att1,score1 = self.graph_attn(x2,x2,x2,graph_mask)
            att2,score2 = self.attri_attn(x2,x2,x2,attr_mask)
            att3,score3 = self.room_attn(x2,x2,x2,room_mask)

            x = x + self.dropout(att0) + self.dropout(att1) + self.dropout(att2) + self.dropout(att3)

        elif self.mode == 1: # all no mask
            att0,score0 = self.all_attn(x2,x2,x2,None)
            att1,score1 = self.graph_attn(x2,x2,x2,graph_mask)
            att2,score2 = self.attri_attn(x2,x2,x2,attr_mask)
            att3,score3 = self.room_attn(x2,x2,x2,room_mask)

            x = x + self.dropout(att0) + self.dropout(att1) + self.dropout(att2) + self.dropout(att3)

        elif self.mode == 2: 
            att0,score0 = self.all_attn(x2,x2,x2,all_mask)
            score1 = None
            att2,score2 = self.attri_attn(x2,x2,x2,attr_mask)
            att3,score3 = self.room_attn(x2,x2,x2,room_mask)

            x = x + self.dropout(att0) + self.dropout(att2) + self.dropout(att3)

        elif self.mode == 3: 
            att0,score0 = self.all_attn(x2,x2,x2,all_mask)
            att1,score1 = self.graph_attn(x2,x2,x2,graph_mask)
            score2 = None
            att3,score3 = self.room_attn(x2,x2,x2,room_mask)

            x = x + self.dropout(att0) + self.dropout(att1) + self.dropout(att3)

        elif self.mode == 4: 
            att0,score0 = self.all_attn(x2,x2,x2,all_mask)
            att1,score1 = self.graph_attn(x2,x2,x2,graph_mask)
            att2,score2 = self.attri_attn(x2,x2,x2,attr_mask)
            score3 = None

            x = x + self.dropout(att0) + self.dropout(att1) + self.dropout(att2)

        elif self.mode == 5: 
            att0,score0 = self.all_attn(x2,x2,x2,all_mask)
            att1,score1 = self.graph_attn(x2,x2,x2,all_mask)
            att2,score2 = self.attri_attn(x2,x2,x2,all_mask)
            att3,score3 = self.room_attn(x2,x2,x2,all_mask)

            x = x + self.dropout(att0) + self.dropout(att1) + self.dropout(att2) + self.dropout(att3)

        elif self.mode == 6: 
            att0,score0 = self.all_attn(x2,x2,x2,None)
            score1 = None
            score2 = None
            score3 = None

            x = x + self.dropout(att0)

        elif self.mode == 7: # all no mask
            score0 = None
            att1,score1 = self.graph_attn(x2,x2,x2,graph_mask)
            att2,score2 = self.attri_attn(x2,x2,x2,attr_mask)
            att3,score3 = self.room_attn(x2,x2,x2,room_mask)

            x = x + self.dropout(att1) + self.dropout(att2) + self.dropout(att3)

        elif self.mode == 8: # all no mask
            att0,score0 = self.all_attn(x2,x2,x2,all_mask)
            graph_mask[room_mask==0] = 0
            att1,score1 = self.graph_attn(x2,x2,x2,graph_mask)
            att2,score2 = self.attri_attn(x2,x2,x2,attr_mask)
            score3 = None

            x = x + self.dropout(att0) + self.dropout(att1) + self.dropout(att2)

        x2 = self.norm_2(x)
        x = x + self.dropout(self.ff(x2))
        return x, score0, score1, score2, score3

class TransformerEncoder_Basic(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        """ Initialize the Attention module.
            :param:
                dim       -> int : number of hidden dimension of attention
                depth     -> int : number of layer for the transformer
                heads     -> int : Number of heads
                mlp_dim   -> int : number of hidden dimension for mlp
                dropout   -> float : Dropout rate
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention_Basic(dim, heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        """ Forward pass through the Attention module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                x -> torch.Tensor: Output of the Transformer
                l_attn -> list(torch.Tensor): list of the attention
        """
        l_attn = []
        for attn, ff in self.layers:
            attention_value, attention_weight = attn(x)
            x = attention_value + x
            x = ff(x) + x
            l_attn.append(attention_weight)
        return x, l_attn

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, mode, dropout=0.1):
        """ Initialize the Attention module.
            :param:
                dim       -> int : number of hidden dimension of attention
                depth     -> int : number of layer for the transformer
                heads     -> int : Number of heads
                mlp_dim   -> int : number of hidden dimension for mlp
                dropout   -> float : Dropout rate
        """
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(dim, heads, mlp_dim, dropout, mode) for _ in range(depth)])

    def forward(self, x, all_mask, graph_mask, attr_mask, room_mask):
        """ Forward pass through the Attention module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                x -> torch.Tensor: Output of the Transformer
                l_attn -> list(torch.Tensor): list of the attention
        """
        l_att0 = []
        l_att1 = []
        l_att2 = []
        l_att3 = []

        for layer in self.layers:
            x, score0, score1, score2, score3 = layer(x, all_mask, graph_mask, attr_mask, room_mask)
            l_att0.append(score0)
            l_att1.append(score1)
            l_att2.append(score2)
            l_att3.append(score3)
            
        return x, l_att0, l_att1, l_att2, l_att3
