import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
from .block import BlockDataset, LatentBlockDataset
import numpy as np


def load_cifar():
    train = datasets.CIFAR10(root="data", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

    val = datasets.CIFAR10(root="data", train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    return train, val


def load_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/randact_traj_length_100_n_trials_1000_n_contexts_1.npy'

    train = BlockDataset(data_file_path, train=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ]))

    val = BlockDataset(data_file_path, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(
                               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
    return train, val

def load_latent_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/latent_e_indices.npy'

    train = LatentBlockDataset(data_file_path, train=True,
                         transform=None)

    val = LatentBlockDataset(data_file_path, train=False,
                       transform=None)
    return train, val


def data_loaders(train_data, val_data, batch_size):

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    return train_loader, val_loader


def load_data_and_data_loaders(dataset, batch_size):
    if dataset == 'CIFAR10':
        training_data, validation_data = load_cifar()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.train_data / 255.0)

    elif dataset == 'BLOCK':
        training_data, validation_data = load_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data / 255.0)
    elif dataset == 'LATENT_BLOCK':
        training_data, validation_data = load_latent_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data)

    else:
        raise ValueError(
            'Invalid dataset: only CIFAR10 and BLOCK datasets are supported.')

    return training_data, validation_data, training_loader, validation_loader, x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def save_model_and_results(model, results, hyperparameters, timestamp):
    os.makedirs('Pretrain_VQVAE',exist_ok=True)
    SAVE_MODEL_PATH = 'Pretrain_VQVAE'

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save,
               SAVE_MODEL_PATH + '/vqvae_data_' + timestamp + '.pth')

def reconstruct(data,model,device):
    
    x = torch.tensor(data).float().to(device)
    # x = x.permute(0,3,1,2).contiguous()
    vq_encoder_output = model.pre_quantization_conv(model.encoder(x))
    _, z_q, _, _,e_indices = model.vector_quantization(vq_encoder_output)
    
    x_recon = model.decoder(z_q)
    return x,x_recon, z_q,e_indices

# def create_All_Mask(mask_map,device):
#     batch_size, attri_count, sqe_len = mask_map.shape
#     mask_map_flat = mask_map.view(batch_size,-1)
#     M = torch.ones(batch_size, attri_count * sqe_len + 1, attri_count * sqe_len + 1).to(device)
    
#     for batch_index in range(batch_size):
#         for i,j in enumerate(mask_map_flat[batch_index]):
#             if j == 0: 
#                 M[batch_index, i, :-1] = mask_map_flat[batch_index].detach().clone()
#                 M[batch_index, i, -1] = 0
#         M[batch_index,-1, :-1] = mask_map_flat[batch_index].detach().clone()
#         M[batch_index,-1, -1] = 0

#     return M

def create_All_Mask(mask_map, device):
    batch_size, attri_count, sqe_len = mask_map.shape
    T = attri_count * sqe_len  # Total number of flattened elements per batch
    mask_map_flat = mask_map.view(batch_size, T)
    M = torch.ones(batch_size, T + 1, T + 1, device=device).long()

    # For every batch element, find the flattened indices where the mask is zero.
    # b_idx and r_idx are 1D tensors containing the batch indices and row indices respectively.
    b_idx, r_idx = torch.where(mask_map_flat == 0)
    
    # For each (batch, row) where the mask is zero, update the row in M:
    # - Set all elements except the last one equal to the entire flattened mask for that batch.
    # - Set the last element to 0.
    M[b_idx, r_idx, :-1] = mask_map_flat[b_idx]
    M[b_idx, r_idx, -1] = 0

    # Update the last row of M for each batch:
    # - Set all elements except the last one equal to the entire flattened mask.
    # - Set the last element to 0.
    M[:, -1, :-1] = mask_map_flat
    M[:, -1, -1] = 0

    return M

# def create_Room_Mask(mask_map,device):
#     batch_size, attri_count, sqe_len = mask_map.shape
#     M = torch.ones(batch_size, attri_count * sqe_len + 1, attri_count * sqe_len + 1).to(device)
    
#     for batch_index in range(batch_size):
#         for i in range(sqe_len):
#             temp = []
#             for j in range(attri_count):
#                 if mask_map[batch_index,j,i] == 0: temp.append(j)
#             for v1 in temp: 
#                 for v2 in temp:
#                     M[batch_index, v1*sqe_len + i, v2*sqe_len + i] = 0
#                     M[batch_index, v2*sqe_len + i, v1*sqe_len + i] = 0
#                     M[batch_index, v1*sqe_len + i, -1] = 0
#                     M[batch_index, v2*sqe_len + i, -1] = 0
#                     M[batch_index, -1, v1*sqe_len + i] = 0
#                     M[batch_index, -1, v2*sqe_len + i] = 0

#     return M

def create_Room_Mask(mask_map, device):

    batch_size, attri_count, sqe_len = mask_map.shape
    T = attri_count * sqe_len  # total number of flattened positions per batch
    # Initialize M to ones.
    M = torch.ones(batch_size, T + 1, T + 1, device=device).long()
    
    # Precompute a matrix of flattened indices (for each attribute j and sequence pos i)
    # such that row_indices[j, i] == j*sqe_len + i.
    j_idx = torch.arange(attri_count, device=device).unsqueeze(1)  # shape (attri_count, 1)
    i_idx = torch.arange(sqe_len, device=device).unsqueeze(0)      # shape (1, sqe_len)
    row_indices = j_idx * sqe_len + i_idx  # shape (attri_count, sqe_len)
    
    # Loop over sequence positions (i). For each i we vectorize over batch and attribute.
    for i in range(sqe_len):
        # For each batch, determine which attributes j have mask_map[b, j, i] == 0.
        # cond has shape (batch_size, attri_count)
        cond = (mask_map[:, :, i] == 0)
        # rows is a 1D tensor of length attri_count holding the flattened index for each attribute at position i.
        rows = row_indices[:, i]  # rows[j] = j*sqe_len + i

        # -------------------------------
        # Update the "pair" entries.
        # For each batch b, for every pair (j, k) such that cond[b, j] and cond[b, k] are True,
        # set M[b, rows[j], rows[k]] and its symmetric counterpart to 0.
        # We can compute the outer product of cond along the attribute dimension:
        cond_outer = cond.unsqueeze(2) & cond.unsqueeze(1)  # shape (batch_size, attri_count, attri_count)

        # Create index grids for batch and for the two attribute dimensions:
        b_idx = torch.arange(batch_size, device=device).view(batch_size, 1, 1).expand(batch_size, attri_count, attri_count)
        # r1 and r2 are the flattened indices corresponding to attributes j and k at position i.
        r1 = rows.view(1, attri_count, 1).expand(batch_size, attri_count, attri_count)
        r2 = rows.view(1, 1, attri_count).expand(batch_size, attri_count, attri_count)
        # Update all positions where both corresponding mask entries are zero.
        M[b_idx[cond_outer], r1[cond_outer], r2[cond_outer]] = 0

        # -------------------------------
        # Update the last column and last row.
        # For every batch b and every attribute j with cond[b, j] True,
        # set M[b, rows[j], -1] = 0 and M[b, -1, rows[j]] = 0.
        b_idx_single = torch.arange(batch_size, device=device).view(batch_size, 1).expand(batch_size, attri_count)
        rows_expanded = rows.unsqueeze(0).expand(batch_size, attri_count)
        M[b_idx_single[cond], rows_expanded[cond], T] = 0  # last column index is T
        M[b_idx_single[cond], T, rows_expanded[cond]] = 0  # last row index is T

    return M

# def create_Attr_Mask(mask_map,device):
#     batch_size, attri_count, sqe_len = mask_map.shape
#     M = torch.ones(batch_size, attri_count * sqe_len + 1, attri_count * sqe_len + 1).to(device)
    
#     for batch_index in range(batch_size):
#         for i in range(attri_count):
#             for j in range(sqe_len):
#                 if mask_map[batch_index,i,j] == 0:
#                     M[batch_index, i*sqe_len+j, i*sqe_len:(i+1)*sqe_len] = mask_map[batch_index,i]
#                     M[batch_index, i*sqe_len+j, -1] = 0
#             M[batch_index, -1, i*sqe_len:(i+1)*sqe_len] = mask_map[batch_index,i]

#     return M

def create_Attr_Mask(mask_map, device):
    batch_size, attri_count, sqe_len = mask_map.shape
    # Create M of shape (batch_size, attri_count*sqe_len + 1, attri_count*sqe_len + 1)
    M = torch.ones(batch_size, attri_count * sqe_len + 1,
                   attri_count * sqe_len + 1, device=device).long()
    
    # Find all indices where mask_map == 0.
    # b_idx, i_idx, j_idx are 1D tensors giving the batch, attribute, and sequence indices.
    b_idx, i_idx, j_idx = torch.where(mask_map == 0)
    if b_idx.numel() > 0:
        # For each such (batch, attribute, sequence) location, the corresponding row in M is:
        # row = i * sqe_len + j.
        rows = i_idx * sqe_len + j_idx  # shape (N,)
        
        # For each such row, we want to set the slice corresponding to the same attribute,
        # i.e. columns from i*sqe_len to (i+1)*sqe_len, equal to mask_map[b, i, :].
        # To do this we first build the corresponding column indices.
        # For each attribute index i (from i_idx) the columns are: i*sqe_len + k for k=0...sqe_len-1.
        k = torch.arange(sqe_len, device=device)  # shape (sqe_len,)
        col_idx = i_idx.unsqueeze(1) * sqe_len + k.unsqueeze(0)  # shape (N, sqe_len)
        
        # Now, using advanced indexing, assign:
        #   M[b, row, col_idx] = mask_map[b, i, :]  for each corresponding b, i.
        # Note that mask_map[b_idx, i_idx, :] has shape (N, sqe_len)
        M[b_idx.unsqueeze(1), rows.unsqueeze(1), col_idx] = mask_map[b_idx, i_idx, :]
        
        # Also, set the last column of these rows to zero.
        M[b_idx, rows, -1] = 0

    # Finally, update the last row of M. For each attribute i, we want:
    #   M[b, -1, i*sqe_len:(i+1)*sqe_len] = mask_map[b, i, :]
    # We can do this for all attributes at once by reshaping mask_map.
    M[:, -1, :attri_count * sqe_len] = mask_map.reshape(batch_size, attri_count * sqe_len)
    
    return M

# def create_Graph_Mask(A,mask_map,device):
#     batch_size, attri_count, sqe_len = mask_map.shape
#     M = torch.ones(batch_size, attri_count * sqe_len + 1, attri_count * sqe_len + 1).to(device)

#     for batch_index in range(batch_size):
#         for i in range(sqe_len):
#             if not A[batch_index, i].equal(torch.ones(sqe_len).to(device)):
#                 for j in range(i,sqe_len):
#                     if A[batch_index, i, j] == 1:
#                         temp = []
#                         for idx,m in enumerate(mask_map[batch_index,:,j]):
#                             if m == 0: temp.append([idx,j])
#                         for idx,n in enumerate(mask_map[batch_index,:,i]):
#                             if n == 0: temp.append([idx,i])
#                         for v1,r1 in temp: 
#                             for v2,r2 in temp:
#                                 M[batch_index, v1*sqe_len + r1, v2*sqe_len + r2] = 0
#                                 M[batch_index, v2*sqe_len + r2, v1*sqe_len + r1] = 0
#                                 M[batch_index, v1*sqe_len + r1, -1] = 0
#                                 M[batch_index, v2*sqe_len + r2, -1] = 0
#                                 M[batch_index, -1, v1*sqe_len + r1] = 0
#                                 M[batch_index, -1, v2*sqe_len + r2] = 0

#     return M

def create_Graph_Mask(A, mask_map, device):

    batch_size, attri_count, sqe_len = mask_map.shape
    T = attri_count * sqe_len
    # Initialize M to ones.
    M = torch.ones(batch_size, T + 1, T + 1, device=device).long()
    # Pre-create a vector of ones for comparison.
    ones_vec = torch.ones(sqe_len, device=device)

    # Process each batch independently.
    for b in range(batch_size):
        # We'll accumulate an update mask for the top-left T x T block,
        # and a 1D mask for which flattened indices should force the last row/column to 0.
        update_block = torch.zeros(T, T, dtype=torch.bool, device=device)
        update_row = torch.zeros(T, dtype=torch.bool, device=device)
        A_b = A[b]            # shape (sqe_len, sqe_len)
        mask_b = mask_map[b]  # shape (attri_count, sqe_len)
        
        # Loop over sequence positions i.
        for i in range(sqe_len):
            # Only process if the i-th row of A_b is not all ones.
            if torch.all(A_b[i] == ones_vec):
                continue

            # For each j from i to the end...
            # (j runs over the second sequence dimension)
            for j in range(i, sqe_len):
                # Only process if A[b, i, j] == 1.
                if A_b[i, j] != 1:
                    continue

                # For column j, determine which attributes are “masked” (==0).
                zeros_j = (mask_b[:, j] == 0)
                # For column i, determine which attributes are masked.
                zeros_i = (mask_b[:, i] == 0)
                # If neither column has any zeros, there’s nothing to update.
                if not (zeros_i.any() or zeros_j.any()):
                    continue

                # Get the attribute indices where the mask is zero.
                idx_j = torch.nonzero(zeros_j, as_tuple=False).squeeze(1)
                idx_i = torch.nonzero(zeros_i, as_tuple=False).squeeze(1)
                # Map these into flattened indices. For a given attribute a and sequence position r,
                # the flattened index is: a * sqe_len + r.
                flat_j = idx_j * sqe_len + j
                flat_i = idx_i * sqe_len + i
                # The union of indices (unique values) for this (i,j) pair:
                union = torch.unique(torch.cat([flat_i, flat_j]))
                if union.numel() == 0:
                    continue
                # For every pair (p, q) in the union, we want to update M[b, p, q] = 0.
                # Instead of a nested Python loop, we update the entire submatrix via advanced indexing.
                update_block[union.unsqueeze(1), union.unsqueeze(0)] = True
                # Also record that these flattened indices must have their corresponding row/column
                # in the last row/column set to 0.
                update_row[union] = True

        # Now that we have accumulated all updates for batch b, apply them in one go.
        # Update the T x T block.
        M[b, :T, :T][update_block] = 0
        # Update the last column and last row.
        M[b, :T, T][update_row] = 0
        M[b, T, :T][update_row] = 0

    return M
