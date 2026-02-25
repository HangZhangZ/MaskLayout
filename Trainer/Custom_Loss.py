import tensorflow as tf
import numpy as np
import cv2
from shapely.geometry import LineString,LinearRing,Polygon,Point,MultiPoint,MultiPolygon
from shapely.ops import unary_union
from shapely import affinity
import torch
from scipy.spatial import KDTree,distance 
import torch.nn as nn  
# from Trainer.decode_function import check_color

max_sentence_length = 14

color_list_rgb = [[0, 0, 0], # 0: None
 [0.5, 0.5, 1], # 1: Living
 [0, 1, 1], # 2: Bath
 [0.5, 1, 0], # 3: CLoset
 [1, 0, 1], # 4: Bed
 [1, 0.5, 0.5], # 5: Kitchen
 [1, 0, 0.5], # 6: Dining
 [0.5, 1, 0.5], # 7: Balcony
 [1, 1, 0], # 8: Corridor
 [0.5, 0.5, 0.5], # 9: end
 [1, 1, 1], # 10: start
] 

room_order = [2,3,5,6,4,1,7,8]
room_order_reverse = [1,8,7,4,6,5,3,2]
attribute_list = ['T','S','L','A','R']
Types_List = ['None','Living', 'Bath', 'CLoset', 'Bed', 'Kitchen', 'Dining', 'Balcony', 'Corridor']

def check_color(img):
    """
    Check the color distribution of an image tensor (shape: [C, H, W]) and return a list of values [a, b, c].
    """
    # Convert conditions to PyTorch operations (channel-first format)
    cond_0_0 = ((img[0] > 0) & (img[0] < 0.2) & (img[-1] > 0.8)).to(torch.int).sum()
    cond_0_1 = ((img[0] > 0.4) & (img[0] < 0.6)).to(torch.int).sum()
    cond_0_2 = (img[0] > 0.8).to(torch.int).sum()

    cond_1_0 = ((img[1] > 0) & (img[1] < 0.2) & (img[-1] > 0.8)).to(torch.int).sum()
    cond_1_1 = ((img[1] > 0.4) & (img[1] < 0.6)).to(torch.int).sum()
    cond_1_2 = (img[1] > 0.8).to(torch.int).sum()

    cond_2_0 = ((img[2] > 0) & (img[2] < 0.2) & (img[-1] > 0.8)).to(torch.int).sum()
    cond_2_1 = ((img[2] > 0.4) & (img[2] < 0.6)).to(torch.int).sum()
    cond_2_2 = (img[2] > 0.8).to(torch.int).sum()

    # Determine the values of a, b, c based on conditions
    a = 1 if cond_0_2 > cond_0_1 and cond_0_2 > cond_0_0 else (0.5 if cond_0_1 > cond_0_0 else 0)
    b = 1 if cond_1_2 > cond_1_1 and cond_1_2 > cond_1_0 else (0.5 if cond_1_1 > cond_1_0 else 0)
    c = 1 if cond_2_2 > cond_2_1 and cond_2_2 > cond_2_0 else (0.5 if cond_2_1 > cond_2_0 else 0)

    return [a, b, c]

class ChamferDistanceLoss(nn.Module):
    """
    Chamfer Distance Loss function for PyTorch.
    Computes the Chamfer Distance between two sets of points.
    """
    def __init__(self):
        super(ChamferDistanceLoss, self).__init__()

    def forward(self, A, B):

        # Compute pairwise distances between all points in A and B

        dist_matrix = torch.cdist(A, B)  # Shape: (batch_size, num_points_A, num_points_B)

        # Find the minimum distance for each point in A to any point in B
        min_dist_A_to_B, _ = torch.min(dist_matrix, dim=2)  # Shape: (batch_size, num_points_A)

        # Find the minimum distance for each point in B to any point in A
        min_dist_B_to_A, _ = torch.min(dist_matrix, dim=1)  # Shape: (batch_size, num_points_B)

        # Compute the Chamfer Distance for each batch
        chamfer_dist = torch.mean(min_dist_A_to_B, dim=1) + torch.mean(min_dist_B_to_A, dim=1)

        # Return the mean Chamfer Distance across the batch
        return torch.mean(chamfer_dist)/12.8


class ChamferDistanceLoss_Filter(nn.Module):
    """
    Chamfer Distance Loss function for PyTorch.
    Computes the Chamfer Distance between two sets of points, filtering out zero points.
    """
    def __init__(self):
        super(ChamferDistanceLoss_Filter, self).__init__()

    def forward(self, A, B):
        """
        Compute the Chamfer Distance between two sets of points A and B, filtering out zero points.
        
        Args:
            A (torch.Tensor): Set of points with shape (batch_size, num_points_A, 3).
            B (torch.Tensor): Set of points with shape (batch_size, num_points_B, 3).
        
        Returns:
            torch.Tensor: Chamfer Distance as a scalar tensor.
        """
        # Ensure the inputs are in float format for distance computation
        A = A.to(torch.float32)
        B = B.to(torch.float32)

        # Filter out zero points
        valid_A = A.abs().sum(dim=-1) > 0  # Shape: (batch_size, num_points_A)
        valid_B = B.abs().sum(dim=-1) > 0  # Shape: (batch_size, num_points_B)

        # Initialize the loss
        total_loss = torch.tensor(0.0, device=A.device, requires_grad=True)

        # Iterate over the batch
        for i in range(A.shape[0]):
            # Get valid points for the current batch
            A_valid = A[i, valid_A[i]]  # Shape: (num_valid_points_A, 3)
            B_valid = B[i, valid_B[i]]  # Shape: (num_valid_points_B, 3)

            if A_valid.shape[0] == 0 or B_valid.shape[0] == 0:
                # If one of the sets has no valid points, skip this batch
                continue

            # Compute pairwise distances between valid points in A and B
            dist_matrix = torch.cdist(A_valid, B_valid)  # Shape: (num_valid_points_A, num_valid_points_B)

            # Find the minimum distance for each point in A to any point in B
            min_dist_A_to_B, _ = torch.min(dist_matrix, dim=1)  # Shape: (num_valid_points_A,)

            # Find the minimum distance for each point in B to any point in A
            min_dist_B_to_A, _ = torch.min(dist_matrix, dim=0)  # Shape: (num_valid_points_B,)

            # Compute the Chamfer Distance for the current batch
            chamfer_dist = (torch.mean(min_dist_A_to_B) + torch.mean(min_dist_B_to_A))/12.8
            total_loss = total_loss + chamfer_dist

        # Normalize by the number of valid batches
        num_valid_batches = sum(1 for i in range(A.shape[0]) if valid_A[i].any() and valid_B[i].any())
        if num_valid_batches > 0: total_loss = total_loss/num_valid_batches

        return total_loss

class VecsRLoss(nn.Module):
    def __init__(self,weight,device):
        """
        Args:
            bound (list): Boundary polygon coordinates.
            color_list_rgb (list): List of RGB colors for each room type.
            room_order_reverse (list): Order of room types for decoding.
        """
        super(VecsRLoss, self).__init__()

        self.color_list_rgb = color_list_rgb
        self.room_order_reverse = room_order_reverse
        self.weight = weight
        self.device = device

    def orient_room(self, pts):
        """
        Rotate and align the room.
        """

        room_back = MultiPoint(pts).minimum_rotated_rectangle
        return room_back

    def forward(self, batch, T, R):
        """
        Compute the loss.

        Args:
            T (torch.Tensor): Room type tensor of shape (nb_sample, sqe_len).
            R (torch.Tensor): Room region tensor of shape (nb_sample, sqe_len, 20, 2).
            vec (torch.Tensor): Vector tensor of shape (2,).

        Returns:
            torch.Tensor: Loss value.
        """
        nb_sample, sqe_len, _ = T.shape
        imgs = torch.zeros((nb_sample, 128, 128, 3), dtype=torch.float32, requires_grad=True).to(self.device)
        GT = torch.zeros((nb_sample, 128, 128, 3), dtype=torch.float32, requires_grad=True).to(self.device)
        for i,j in enumerate(batch): GT[i] = torch.from_numpy(cv2.imread('Data/img/composed/%d.png' % (j%42644), cv2.IMREAD_UNCHANGED)[:,:,:-1])

        GT = GT/255
        # Decode room types
        Type_list = torch.zeros((nb_sample, sqe_len), dtype=torch.int32)
        idxs = torch.where(T == 1)
        for m in range(len(idxs[0])):
            Type_list[idxs[0][m], idxs[1][m]] = int(idxs[-1][m]) + 1

        # Decode room regions
        R_np = R.detach().cpu().numpy()
        gen_R = R_np.dot(1 << np.arange(R_np.shape[-1] - 1, -1, -1))

        # Compute rotation degree
        # rotate_degree = torch.acos(vec.dot(torch.tensor([0.0, 1.0]))) * 180 / torch.pi

        # Generate images
        for r in range(nb_sample):
            for o in self.room_order_reverse:
                for i in range(sqe_len):
                    if Type_list[r, i] == o and not torch.all(torch.eq(gen_R[r, i], torch.zeros((20, 2)))):
                        corners = gen_R[r, i][~np.all(gen_R[r, i] == 0, axis=1)]
                        if len(corners) > 3:
                            # room_back = self.orient_room(corners)
                            coords = np.array(Polygon(corners).exterior.coords[:-1])[:, np.newaxis, :].astype(np.int32)
                            color = [c / 255 for c in self.color_list_rgb[o]]
                            imgs[r] = torch.from_numpy(cv2.fillPoly(imgs[r].numpy(), [coords], color))

        # Compute loss (example: mean squared error between generated images and a target)

        # loss = nn.MSELoss(reduction='mean')(torch.clamp(imgs, 0, 1),GT)
        loss = nn.MSELoss(reduction='mean')(imgs, GT)
        return loss*self.weight

def normalize_per_batch(x, beta=10.0, eps=1e-6):
    """
    Differentiably normalizes a batch of tensors to [0,1] per sample.
    
    Args:
        x (Tensor): Input tensor of shape (batch_size, ...).
        beta (float): Temperature parameter for the softmax approximation.
        eps (float): Small constant to avoid division by zero.
        
    Returns:
        Tensor: Normalized tensor with the same shape as x.
    """
    # batch_size = x.shape[0]
    # x_reshaped = x.view(batch_size, -1)  # Flatten all dimensions except batch
    
    # # Differentiable approximation of the maximum via log-sum-exp.
    # x_max = torch.logsumexp(beta * x_reshaped, dim=1, keepdim=True) / beta
    # # Differentiable approximation of the minimum via log-sum-exp.
    # x_min = -torch.logsumexp(-beta * x_reshaped, dim=1, keepdim=True) / beta

    # x_range = x_max - x_min + eps  # Add eps to avoid division by zero

    # # Normalize to [0,1]
    # x_normalized = (x_reshaped - x_min) / x_range

    # # Reshape back to original dimensions
    # x_normalized = x_normalized.view(x.shape)

    maxx = x.max()
    minx = x.min()

    x = (x-minx)/(maxx-minx)
    # x_normalized = x
    
    return x


# class ComposeImgLoss(nn.Module):
#     def __init__(self,img_weight,attri_count,sqe_len,device):
#         """
#         Args:
#             color_list_rgb (list): List of RGB colors for each room type.
#         """
#         super(ComposeImgLoss, self).__init__()

#         self.color_list_rgb = color_list_rgb
#         self.img_weight = img_weight
#         self.attri_count = attri_count
#         self.sqe_len = sqe_len
#         self.device = device

#     def forward(self, GT, Pred):

#         nb_sample = GT.shape[0]

#         GT_normalized = (GT + 1)/2
#         Pred_normalized = normalize_per_batch(Pred)

#         # all
#         gen = Pred_normalized.view((nb_sample,self.attri_count,self.sqe_len,4,128,128))

#         # decode Attributes

#         gen_T = gen[:,0]
#         gen_L = torch.sum(gen[:,2,:,:-1], 1)
#         gen_A = torch.sum(gen[:,3,:,:-1], 1)
#         gen_R = gen[:,4,:,:-1]
#         gen_W = torch.sum(gen[:,5,:,:-1], 1)

#         gen_base = gen_R[:,-1]
#         gen_base[:] = 0

#         # # Decode Type
#         # Type_list = torch.zeros((nb_sample, self.sqe_len), dtype=torch.int, device=self.device)
#         # for n in range(nb_sample):
#         #     for i in range(self.sqe_len):
#         #         check = check_color(gen_T[n, i])
#         #         if check in color_list_rgb:
#         #             Type_list[n, i] = color_list_rgb.index(check)
#         #         else:
#         #             Type_list[n, i] = 0

#         # # Decode Region
#         # Region_list = ((gen_R[:, :, 0] > 0.9) & (gen_R[:, :, 1] > 0.9) & (gen_R[:, :, 2] > 0.9)).to(torch.int)
#         # for n in range(nb_sample):
#         #     for i in range(self.sqe_len):
#         #         if 8 > Type_list[n, i] > 0:
#         #             for j in range(3): gen_base[n, j, :, :] = gen_base[n, j, :, :] + (Region_list[n, i] * (color_list_rgb[int(Type_list[n, i])][j])).to(self.device)


#         # Decode Type
#         Type_list = torch.zeros((nb_sample, self.sqe_len), dtype=torch.int, device=self.device)

#         for n in range(nb_sample):
#             for i in range(self.sqe_len):
#                 check = check_color(gen_T[n, i])
#                 if check in color_list_rgb: Type_list[n, i] = color_list_rgb.index(check)
#                 else: Type_list[n, i] = 0

#         # Decode Region
#         Region_list = ((gen_R[:, :, 0] > 0.9) & (gen_R[:, :, 1] > 0.9) & (gen_R[:, :, 2] > 0.9)).to(torch.int)

#         # Create a mask for valid Type_list values
#         valid_mask = (Type_list > 0) & (Type_list < 8)  # Shape: (nb_sample, sqe_len)

#         # Expand valid_mask to match the shape of gen_base
#         valid_mask_expanded = valid_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # Shape: (nb_sample, 1, sqe_len, 1, 1)

#         # Convert color_list_rgb to a tensor for efficient indexing
#         color_list_tensor = torch.tensor(color_list_rgb, dtype=torch.float32, device=self.device)  # Shape: (num_colors, 3)

#         # Index into color_list_tensor using Type_list
#         selected_colors = color_list_tensor[Type_list]  # Shape: (nb_sample, sqe_len, 3)

#         # Expand selected_colors to match the shape of Region_list
#         selected_colors_expanded = selected_colors.unsqueeze(-1).unsqueeze(-1)  # Shape: (nb_sample, sqe_len, 3, 1, 1)

#         # Multiply Region_list with selected_colors and apply the valid_mask
#         region_contributions = Region_list.unsqueeze(2) * selected_colors_expanded  # Shape: (nb_sample, sqe_len, 3, H, W)
#         region_contributions = region_contributions * valid_mask_expanded  # Apply mask

#         gen_base = torch.clamp(gen_base + region_contributions.sum(dim=1) + gen_L + gen_A + gen_W, 0, 1)

#         # Compute loss
#         loss = torch.nn.MSELoss(reduction='mean')(gen_base, GT_normalized)
#         return loss * self.img_weight

class ComposeImgLoss(nn.Module):
    def __init__(self, img_weight, attri_count, sqe_len, device):
        """
        Args:
            img_weight (float): Scalar loss weight.
            attri_count (int): Number of attribute channels.
            sqe_len (int): Number of sequence elements.
            device (torch.device): Computation device.
            color_list_rgb (list): List of RGB colors (e.g. lists or tuples) for each room type.
        """
        super(ComposeImgLoss, self).__init__()
        self.img_weight = img_weight
        self.attri_count = attri_count
        self.sqe_len = sqe_len
        self.device = device

        # Save the color list and precompute a mapping from color to index.
        # (This avoids doing repeated "in" and "index" lookups.)
        self.color_list_rgb = color_list_rgb
        self.color_to_index = {tuple(color): i for i, color in enumerate(color_list_rgb)}

        # Pre-create a tensor of colors (shape: [num_colors, 3]) so that later we can index
        # into it in a vectorized way. (Ensure the tensor is on the proper device.)
        self.colors_tensor = torch.tensor(color_list_rgb, dtype=torch.float32, device=device)
        self.mse = torch.nn.MSELoss(reduction='mean')

    def forward(self, GT, Pred):
        nb_sample = GT.shape[0]

        # Normalize ground-truth and predictions.
        GT_normalized = (GT[:, :-1].to(self.device) + 1) / 2
        Pred_normalized = normalize_per_batch(Pred)  # Ensure this function is differentiable

        # Reshape predictions into the expected tensor shape:
        # (nb_sample, attri_count, sqe_len, 4, 128, 128)
        gen = Pred_normalized.to(self.device).view(nb_sample, self.attri_count, self.sqe_len, 4, 128, 128)

        # --- Decode Attributes ---
        # gen_T: Type channel (we assume the first 3 channels encode a color)

        gen_T = gen[:, 0]  # Shape: (nb_sample, sqe_len, 4, 128, 128)
        # gen_L, gen_A, gen_W: Other channels decoded via summation over the sequence elements.
        gen_L = torch.sum(gen[:, 2, :, :-1], dim=1)  # Shape: (nb_sample, 3, 128, 128)
        gen_A = torch.sum(gen[:, 3, :, :-1], dim=1)  # Shape: (nb_sample, 3, 128, 128)
        # gen_R: Region channel (will be used for masking)
        gen_R = gen[:, 4, :, :-1]                     # Shape: (nb_sample, sqe_len, 3, 128, 128)
        gen_W = torch.sum(gen[:, 5, :, :-1], dim=1)   # Shape: (nb_sample, 3, 128, 128)

        # Create the base image (a constant zero image, kept differentiable).
        gen_base = torch.zeros_like(gen_R[:, -1], requires_grad=True).to(self.device)  # (nb_sample, 3, 128, 128)

        # --- Differentiable Type Decoding ---
        # Instead of using a non-differentiable loop with check_color, we average over the spatial
        # dimensions of the first 3 channels of gen_T to get a predicted color per sequence element.
        # pred_color = gen_T[:, :, :3, :, :].mean(dim=(-2, -1))  # (nb_sample, sqe_len, 3)

        # Compute distances to each prototype color in self.colors_tensor.
        # Expand dimensions so that broadcasting works:
        # pred_color: (nb_sample, sqe_len, 3)
        # self.colors_tensor: (num_colors, 3)

        # diff = pred_color.unsqueeze(2) - colors_tensor.unsqueeze(0).unsqueeze(0)  # (nb_sample, sqe_len, num_colors, 3)
        # squared_distance = (diff ** 2).sum(dim=-1)  # (nb_sample, sqe_len, num_colors)

        # # Turn distances into similarity scores (negative distance).
        # sim = -squared_distance
        # # Use a softmax to get a weight (probability) distribution over the color prototypes.
        # type_weights = torch.softmax(sim, dim=-1)  # (nb_sample, sqe_len, num_colors)

        # # Compute a differentiable predicted type color as a weighted sum of prototypes.
        # pred_type_color = (type_weights.unsqueeze(-1) * colors_tensor.unsqueeze(0).unsqueeze(0)).sum(dim=2)  # (nb_sample, sqe_len, 3)
        # # print(pred_type_color[0])
        # print(pred_type_color)
        Type_list = torch.zeros((gen_T.shape[0], 14, 3), dtype=torch.float32, requires_grad=True).to(self.device)
        for n in range(gen_T.shape[0]):
            for i in range(14):
                check = check_color(gen_T[n, i])
                if check in color_list_rgb:
                    Type_list[n, i,:] = torch.tensor(check)#color_list_rgb.index(check)
                else:
                    Type_list[n, i] = 0
        

        # --- Differentiable Region Decoding ---
        # Replace hard thresholding with soft thresholds using sigmoid functions.
        temperature = 10.0  # higher temperature makes the sigmoid steeper (closer to a hard threshold)
        region_soft = torch.sigmoid((gen_R[:, :, 0] - 0.9) * temperature) * \
                        torch.sigmoid((gen_R[:, :, 1] - 0.9) * temperature) * \
                        torch.sigmoid((gen_R[:, :, 2] - 0.9) * temperature)  # (nb_sample, sqe_len, 128, 128)

        # --- Compose the Image ---
        # Expand pred_type_color so that it can be multiplied spatially with region_soft.
        # pred_type_color: (nb_sample, sqe_len, 3) → (nb_sample, sqe_len, 3, 1, 1)
        pred_type_color_expanded = Type_list.unsqueeze(-1).unsqueeze(-1)
        # Expand region_soft: (nb_sample, sqe_len, 128, 128) → (nb_sample, sqe_len, 1, 128, 128)
        region_soft_expanded = region_soft.unsqueeze(2)
        # Multiply and sum over the sequence dimension.
        weighted_region = region_soft_expanded * pred_type_color_expanded  # (nb_sample, sqe_len, 3, 128, 128)
        region_contribution = normalize_per_batch(weighted_region.sum(dim=1))   # (nb_sample, 3, 128, 128)
        

        # Combine all contributions, clamping the composite image to [0, 1].
        gen_base = gen_base + region_contribution + gen_L + gen_A + gen_W
        composite = torch.clamp(gen_base, 0, 1)#.detach().cpu().numpy()
        # print(composite[0,0,60:70,50:70])

        # --- Compute Loss ---
        loss = self.mse(composite, GT_normalized)
        return loss * self.img_weight


# class CodesReconsLoss(nn.Module):
#     def __init__(self,img_weight,attri_count,sqe_len,quantizer,decoder,device):
#         """
#         Args:
#             color_list_rgb (list): List of RGB colors for each room type.
#         """
#         super(CodesReconsLoss, self).__init__()

#         self.color_list_rgb = color_list_rgb
#         self.img_weight = img_weight
#         self.attri_count = attri_count
#         self.sqe_len = sqe_len
#         self.quantizer = quantizer
#         self.vq_decoder = decoder
#         self.device = device

#     def forward(self, batch,codes):

#         nb_sample, _, _ = codes.shape

#         imgs = torch.zeros((nb_sample, 128, 128, 3), dtype=torch.float32, requires_grad=True).to(self.device)
#         GT = torch.zeros((nb_sample, 128, 128, 3), dtype=torch.float32, requires_grad=True).to(self.device)
#         for i,j in enumerate(batch): GT[i] = torch.from_numpy(cv2.imread('Data/img/composed/%d.png' % (j%42644), cv2.IMREAD_UNCHANGED)[:,:,:-1])

#         GT = GT/255
#         # Decode room types
#         priors = tf.one_hot(codes.cpu().detach().numpy().astype("int32"), 32).numpy()
#         quant = tf.matmul(priors.astype("float32"), self.quantizer.embeddings, transpose_b=True)
#         quant = tf.reshape(quant, (-1, *((5,5,32))))
#         gen = self.vq_decoder.predict(quant).reshape((nb_sample,self.attri_count,self.sqe_len,128,128,4))

#         # decode Attributes

#         gen_T = gen[:,0]
#         gen_L = gen[:,2]
#         gen_A = gen[:,3]
#         gen_R = gen[:,4]
#         gen_W = gen[:,5]

#         # decode Type
#         Type_list = np.zeros((nb_sample,self.sqe_len))
#         for n in range(nb_sample):
#             for i in range(self.sqe_len): 
#                 check = check_color(gen_T[n,i])
#                 if check in color_list_rgb: Type_list[n,i] = int(color_list_rgb.index(check))
#                 else: Type_list[n,i] = 0

#         # decode Region
#         Region_list = ((gen_R[:,:,:,:,0]>0.9)&(gen_R[:,:,:,:,1]>0.9)&(gen_R[:,:,:,:,2]>0.9)&(gen_R[:,:,:,:,3]>0.9)).astype(int)
#         for n in range(nb_sample):
#             for i in range(self.sqe_len):
#                 if 8 > Type_list[n,i] > 0:
#                     for j in range(3): imgs[n,:,:,j] += torch.from_numpy(Region_list[n,i]*(color_list_rgb[int(Type_list[n,i])][j]/255)).to(self.device)
        
#         # decode Location
#         Loc_list = np.sum(((gen_L[:,:,:,:,0]>0.9)&(gen_L[:,:,:,:,1]>0.9)&(gen_L[:,:,:,:,2]>0.9)&(gen_L[:,:,:,:,3]>0.9)).astype(int),axis=1)
#         imgs += torch.from_numpy(np.stack([Loc_list for _ in range(3)],axis=-1)).to(self.device)
        
#         # decode Adjacency
#         Ada_list = np.sum(((gen_A[:,:,:,:,0]>0.9)&(gen_A[:,:,:,:,1]>0.9)&(gen_A[:,:,:,:,2]>0.9)&(gen_A[:,:,:,:,3]>0.9)).astype(int),axis=1)
#         imgs += torch.from_numpy(np.stack([Ada_list for _ in range(3)],axis=-1)).to(self.device)
        
#         # decode Window
#         Window_list = np.sum(((gen_W[:,:,:,:,0]>0.9)&(gen_W[:,:,:,:,1]>0.9)&(gen_W[:,:,:,:,2]>0.9)&(gen_W[:,:,:,:,3]>0.9)).astype(int),axis=1)
#         imgs += torch.from_numpy(np.stack([Window_list for _ in range(3)],axis=-1)).to(self.device)

#         # loss = nn.MSELoss(reduction='mean')(torch.clamp(imgs, 0, 1),GT)
#         loss = nn.MSELoss(reduction='mean')(imgs, GT)
#         return loss*self.img_weight