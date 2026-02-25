import tensorflow as tf
import numpy as np
import cv2
from shapely.geometry import LineString,LinearRing,Polygon,Point,MultiPoint,MultiPolygon
from shapely.ops import unary_union
from shapely import affinity
import torch
from scipy.spatial import KDTree,distance 
import torch.nn as nn  
from itertools import combinations
from Trainer.Custom_Loss import normalize_per_batch

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

room_order = [2,3,4,5,6,7,8,1]
room_order_reverse = [1,8,7,6,5,4,3,2]
attribute_list = ['T','S','L','A','R']
Types_List = ['None','Living', 'Bath', 'CLoset', 'Bed', 'Kitchen', 'Dining', 'Balcony', 'Corridor']


# used for check valid value
def load_img_batch(batch_idx,attri_count,sqe_len,device):

    img = np.zeros((batch_idx.shape[0],attri_count*sqe_len,128,128,4))
    for i,j in enumerate(batch_idx):
        for s in range(j[1]):
            for m,n in enumerate(attribute_list):
                img[i,s*attri_count + m] = cv2.imread('Data/img/%s/%d/%d.png' % (n,s+1,j[0]), cv2.IMREAD_UNCHANGED)
            if j[2+s] == 1:
                img[i,s*attri_count + 5] = cv2.imread('Data/img/W/%d/%d.png' % (s+1,j[0]), cv2.IMREAD_UNCHANGED)
    
    img = torch.tensor(img).to(device).view(batch_idx.shape[0]*attri_count*sqe_len,128,128,4)/255
    return img

def load_img_batch_all(batch_idx,lens,device):

    img = np.zeros((batch_idx.shape[0],lens*128,128,4))
    # for i,j in enumerate(batch_idx): img[i*lens*128:(i+1)*lens*128] = cv2.imread('Data/img/batch/Train/%d.png' % (j), cv2.IMREAD_UNCHANGED)
    for i,j in enumerate(batch_idx): img[i] = cv2.imread('Data/img/batch/Train/%d.png' % (j), cv2.IMREAD_UNCHANGED)

    img = torch.tensor(img).to(device)/255#.view(batch_idx.shape[0]*attri_count*sqe_len,128,128,4)
    return img


def load_composed_img(batch_idx,lens,device):

    img = np.zeros((batch_idx.shape[0],lens*128,128,4))
    # for i,j in enumerate(batch_idx): img[i*lens*128:(i+1)*lens*128] = cv2.imread('Data/img/batch/Train/%d.png' % (j), cv2.IMREAD_UNCHANGED)
    for i,j in enumerate(batch_idx): img[i] = cv2.imread('Data/img/composed/%d.png' % (j), cv2.IMREAD_UNCHANGED)

    img = torch.tensor(img).to(device)/255#.view(batch_idx.shape[0]*attri_count*sqe_len,128,128,4)
    return img


def bit2int(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)
                 
def img_decode(codes,quantizer,decoder,attri_count,sqe_len):

    # img1: All attributes img
    priors = tf.one_hot(codes.cpu().numpy().astype("int32"), 32).numpy()
    quant = tf.matmul(priors.astype("float32"), quantizer.embeddings, transpose_b=True)
    quant = tf.reshape(quant, (-1, *((5,5,32))))
    gen = decoder.predict(quant)

    data = gen.reshape((attri_count,sqe_len,128,128,4))
    img_1 = data.copy().reshape((attri_count*sqe_len*128,128,4))*255
    # sqe = data.reshape((attri_count*sqe_len,128,128,4))

    for s in range(attri_count*sqe_len): cv2.imwrite('temp/%d.png' % (s), gen[s].copy()*255)

    # img2: Location + Window + Region
    # decode Attributes
    img_2 = np.zeros((128,128,4))
    gen_T, _, gen_L, _, gen_R, gen_W = data
    # decode Type
    Type_list = np.zeros(sqe_len)
    for i in range(sqe_len): 
        check = check_color(gen_T[i])
        if check in color_list_rgb:
            Type_list[i] = int(color_list_rgb.index(check))
        else:
            Type_list[i] = 0
    # print(Type_list)

    # decode Location
    Loc_list = np.sum(((gen_L[:,:,:,0]>0.9)&(gen_L[:,:,:,1]>0.9)&(gen_L[:,:,:,2]>0.9)&(gen_L[:,:,:,3]>0.9)).astype(int),axis=0)*255
    for i in range(4): img_2[:,:,i] += Loc_list

    # decode Region
    Region_list = ((gen_R[:,:,:,0]>0.9)&(gen_R[:,:,:,1]>0.9)&(gen_R[:,:,:,2]>0.9)&(gen_R[:,:,:,3]>0.9)).astype(int)
    for i in range(sqe_len):
        if 8 > Type_list[i] > 0:
            for j in range(3): img_2[:,:,j] += Region_list[i]*color_list_rgb[int(Type_list[i])][j]
            img_2[:,:,-1] += Region_list[i]*255
    
    # decode Window
    Window_list = np.sum(((gen_W[:,:,:,0]>0.9)&(gen_W[:,:,:,1]>0.9)&(gen_W[:,:,:,2]>0.9)&(gen_W[:,:,:,3]>0.9)).astype(int),axis=0)*255
    for i in range(4): img_2[:,:,i] += Window_list

    return img_1, img_2

def imgs_R_decode(nb_sample,codes,quantizer,decoder,attri_count,sqe_len):

    # img1: All attributes img
    priors = tf.one_hot(codes.cpu().numpy().astype("int32"), 32).numpy()
    quant = tf.matmul(priors.astype("float32"), quantizer.embeddings, transpose_b=True)
    quant = tf.reshape(quant, (-1, *((5,5,32))))
    gen = decoder.predict(quant)

    data = gen.reshape((nb_sample,attri_count,sqe_len,128,128,4))
    # img2: Location + Window + Region
    # decode Attributes
    imgs = np.zeros((nb_sample,128,128,4))
    gen_T = data[:,0]
    gen_R = data[:,-2]
    # decode Type
    Type_list = np.zeros((nb_sample,sqe_len))
    for r in range(nb_sample):
        for i in range(sqe_len): 
            check = check_color(gen_T[r,i])
            if check in color_list_rgb:
                Type_list[r,i] = int(color_list_rgb.index(check))
            else:
                Type_list[r,i] = 0

    # decode Region
    Region_list = ((gen_R[:,:,:,:,0]>0.9)&(gen_R[:,:,:,:,1]>0.9)&(gen_R[:,:,:,:,2]>0.9)&(gen_R[:,:,:,:,3]>0.9)).astype(int)
    for r in range(nb_sample):
        for i in range(sqe_len):
            if 8 > Type_list[r,i] > 0:
                for j in range(3): imgs[r,:,:,j] += Region_list[r,i]*color_list_rgb[int(Type_list[r,i])][j]
                imgs[r,:,:,-1] += Region_list[r,i]*255
    
    # decode Window
    # Window_list = np.sum(((gen_W[:,:,:,0]>0.9)&(gen_W[:,:,:,1]>0.9)&(gen_W[:,:,:,2]>0.9)&(gen_W[:,:,:,3]>0.9)).astype(int),axis=0)*255
    # for i in range(4): img_2[:,:,i] += Window_list

    return imgs

def vq_imgs_decode(gen,device):

    # colors_tensor = torch.tensor(color_list_rgb, dtype=torch.float32, device=device)

    gen_T = gen[:, 0]  # Shape: (nb_sample, sqe_len, 4, 128, 128)
    # gen_L, gen_A, gen_W: Other channels decoded via summation over the sequence elements.
    gen_L = torch.sum(gen[:, 2, :, :-1], dim=1)  # Shape: (nb_sample, 3, 128, 128)
    gen_A = torch.sum(gen[:, 3, :, :-1], dim=1)  # Shape: (nb_sample, 3, 128, 128)
    # gen_R: Region channel (will be used for masking)
    gen_R = gen[:, 4, :, :-1]                     # Shape: (nb_sample, sqe_len, 3, 128, 128)
    gen_W = torch.sum(gen[:, 5, :, :-1], dim=1)   # Shape: (nb_sample, 3, 128, 128)

    # Create the base image (a constant zero image, kept differentiable).
    gen_base = torch.zeros_like(gen_R[:, -1]).to(device)  # (nb_sample, 3, 128, 128)

    Type_list = torch.zeros((gen_T.shape[0], 14, 3), dtype=torch.float32).to(device)
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
    composite = torch.clamp(gen_base + region_contribution + gen_L + gen_A + gen_W, 0, 1).detach().cpu().numpy()

    return np.transpose(composite,(0,2,3,1))*255

def vecs_R_test(nb_sample,T,R,W,sqe_len):

    # decode Attributes
    imgs = np.zeros((nb_sample,128,128,4))
    Type_list = np.zeros((nb_sample,sqe_len))
    idxs = np.where(T.cpu().numpy()==1)
    for m in range(len(idxs[0])):
        Type_list[idxs[0][m],idxs[1][m]] = idxs[-1][m]+1
    # print(Type_list)
    R = R.cpu().numpy()
    W = W.cpu().numpy()
    gen_R = R.dot(1 << np.arange(R.shape[-1] - 1, -1, -1))
    gen_w = W.dot(1 << np.arange(W.shape[-1] - 1, -1, -1))
    # print(gen_R[0])

    unioned_rooms = None
    # decode Region
    # Region_list = ((gen_R[:,:,:,:,0]>0.9)&(gen_R[:,:,:,:,1]>0.9)&(gen_R[:,:,:,:,2]>0.9)&(gen_R[:,:,:,:,3]>0.9)).astype(int)
    for r in range(nb_sample):
        for o in room_order_reverse:
            for i in range(sqe_len):
                if Type_list[r,i] == o and not np.array_equal(gen_R[r,i],np.zeros((10,2))):
                    corners = gen_R[r, i][~np.all(gen_R[r, i] == 0, axis=1)]
                    if len(corners) > 3:
                        try:
                            # room_back = MultiPoint(corners).minimum_rotated_rectangle
                            # coords = np.array(room_back.exterior.coords[:-1])
                            cv2.fillPoly(imgs[r], [np.array(Polygon(corners).exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], color_list_rgb[o]+[1])
                            # cv2.putText(imgs[r],Types_List[o], (int(room_back.centroid.x),int(room_back.centroid.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 20, 20, 255), 1)
                        except:
                            continue
    return imgs*255

def vecs_R_decode_1024(nb_sample,T,R,W,sqe_len,bound,vec,Text):

    # decode Attributes
    imgs = np.zeros((nb_sample,1024,1024,4))
    Type_list = np.zeros((nb_sample,sqe_len))
    idxs = np.where(T.cpu().numpy()==1)
    for m in range(len(idxs[0])):
        Type_list[idxs[0][m],idxs[1][m]] = idxs[-1][m]+1
    R = R.cpu().numpy()
    W = W.cpu().numpy()
    gen_R = R.dot(1 << np.arange(R.shape[-1] - 1, -1, -1))*8
    gen_w = W.dot(1 << np.arange(W.shape[-1] - 1, -1, -1))*8

    gen_R_distri = np.zeros_like(gen_R)

    # print(Type_list)
    # print(gen_R[0])
    rotate_degree = np.arccos(vec.dot(np.array([0,1])))*180/np.pi
    out_bound = Polygon([[0,0],[0,1023],[1023,1023],[1023,0]]).difference(Polygon(bound))
    unioned_rooms = None

    # decode Region
    # Region_list = ((gen_R[:,:,:,:,0]>0.9)&(gen_R[:,:,:,:,1]>0.9)&(gen_R[:,:,:,:,2]>0.9)&(gen_R[:,:,:,:,3]>0.9)).astype(int)
    for r in range(nb_sample):
        rooms = []
        types = []
        cv2.fillPoly(imgs[r], [np.array(Polygon(bound).buffer(4).exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], [c for c in color_list_rgb[9]]+[1])
        cv2.fillPoly(imgs[r], [np.array(Polygon(bound).exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], [c for c in color_list_rgb[1]]+[1])
        for o in room_order_reverse:
            for i in range(sqe_len):
                if Type_list[r,i] == o and not np.array_equal(gen_R[r,i],np.zeros((20,2))):
                    corners,pts = del_non(gen_R[r,i],Polygon(bound))#gen_R[r, i][~np.all(gen_R[r, i] == 0, axis=1)]
                    if len(corners) > 3:
                        try:
                            room_back = orient_room(corners,rotate_degree)
                            gen_R_distri[r,i,:len(corners)] = np.array(pts)
                            # coords = np.array(room_back.exterior.coords[:-1])
                            cv2.fillPoly(imgs[r], [np.array(room_back.exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], [c for c in color_list_rgb[o]]+[1])
                            rooms.append(room_back)
                            types.append(Types_List[o])
                        except:
                            continue
        if Text: 
            for ro in range(len(rooms)): cv2.putText(imgs[r],types[ro], (int(rooms[ro].centroid.x),int(rooms[ro].centroid.y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (1, 0.3, 0.3, 1), 4)
        
    imgs = np.clip(imgs, 0, 1)
    return imgs,gen_R_distri

def vecs_R_decode_raw_1024(nb_sample,T,R,W,sqe_len,bound,vec,Text):

    # decode Attributes
    imgs = np.zeros((nb_sample,1024,1024,4))
    Type_list = np.zeros((nb_sample,sqe_len))
    idxs = np.where(T.cpu().numpy()==1)
    for m in range(len(idxs[0])):
        Type_list[idxs[0][m],idxs[1][m]] = idxs[-1][m]+1
    R = R.cpu().numpy()
    W = W.cpu().numpy()
    gen_R = R.dot(1 << np.arange(R.shape[-1] - 1, -1, -1))*8
    gen_w = W.dot(1 << np.arange(W.shape[-1] - 1, -1, -1))*8

    gen_R_distri = np.zeros_like(gen_R)

    # print(Type_list)
    # print(gen_R[0])
    rotate_degree = np.arccos(vec.dot(np.array([0,1])))*180/np.pi
    out_bound = Polygon([[0,0],[0,1023],[1023,1023],[1023,0]]).difference(Polygon(bound))
    unioned_rooms = None

    # decode Region
    # Region_list = ((gen_R[:,:,:,:,0]>0.9)&(gen_R[:,:,:,:,1]>0.9)&(gen_R[:,:,:,:,2]>0.9)&(gen_R[:,:,:,:,3]>0.9)).astype(int)
    for r in range(nb_sample):
        rooms = []
        types = []
        cv2.fillPoly(imgs[r], [np.array(Polygon(bound).buffer(4).exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], [c for c in color_list_rgb[9]]+[1])
        cv2.fillPoly(imgs[r], [np.array(Polygon(bound).exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], [c for c in color_list_rgb[1]]+[1])
        for o in room_order_reverse:
            for i in range(sqe_len):
                if Type_list[r,i] == o and not np.array_equal(gen_R[r,i],np.zeros((20,2))):
                    corners = gen_R[r, i][~np.all(gen_R[r, i] == 0, axis=1)]#del_non(gen_R[r,i],Polygon(bound))#gen_R[r, i][~np.all(gen_R[r, i] == 0, axis=1)]
                    if len(corners) > 3:
                        try:
                            room_back = Polygon(sort_points_clockwise(corners))#orient_room(corners,rotate_degree)
                            gen_R_distri[r,i,:len(corners)] = np.array(corners)
                            room = room_back.difference(out_bound)
                            if room.geom_type == 'MultiPolygon':
                                room = room.geoms[find_max(room.geoms)]
                            if room.geom_type == 'GeometryCollection':
                                room = room.geoms[0]
                            # coords = np.array(room_back.exterior.coords[:-1])
                            cv2.fillPoly(imgs[r], [np.array(room.exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], [0]*4)
                            cv2.fillPoly(imgs[r], [np.array(room.exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], [c for c in color_list_rgb[o]]+[1])
                            rooms.append(room_back)
                            types.append(Types_List[o])
                        except:
                            continue
        if Text: 
            for ro in range(len(rooms)): cv2.putText(imgs[r],types[ro], (int(rooms[ro].centroid.x),int(rooms[ro].centroid.y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (1, 0.3, 0.3, 1), 4)
        
    np.clip(imgs, 0, 1, out=imgs)
    return imgs,gen_R_distri

def vecs_R_decode(nb_sample,T,R,W,sqe_len,bound,vec,bound_img):

    # decode Attributes
    imgs = np.zeros((nb_sample,128,128,4))
    Type_list = np.zeros((nb_sample,sqe_len))
    idxs = np.where(T.cpu().numpy()==1)
    for m in range(len(idxs[0])):
        Type_list[idxs[0][m],idxs[1][m]] = idxs[-1][m]+1
    R = R.cpu().numpy()
    W = W.cpu().numpy()
    gen_R = R.dot(1 << np.arange(R.shape[-1] - 1, -1, -1))
    gen_w = W.dot(1 << np.arange(W.shape[-1] - 1, -1, -1))

    gen_R_distri = np.zeros_like(gen_R)

    # print(Type_list)
    # print(gen_R[0])
    rotate_degree = np.arccos(vec.dot(np.array([0,1])))*180/np.pi
    out_bound = Polygon([[0,0],[0,127],[127,127],[127,0]]).difference(Polygon(bound))
    unioned_rooms = None
    # decode Region
    # Region_list = ((gen_R[:,:,:,:,0]>0.9)&(gen_R[:,:,:,:,1]>0.9)&(gen_R[:,:,:,:,2]>0.9)&(gen_R[:,:,:,:,3]>0.9)).astype(int)
    for r in range(nb_sample):
        for o in room_order_reverse:
            for i in range(sqe_len):
                if Type_list[r,i] == o and not np.array_equal(gen_R[r,i],np.zeros((20,2))):
                    corners,pts = del_non(gen_R[r,i],Polygon(bound))#gen_R[r, i][~np.all(gen_R[r, i] == 0, axis=1)]
                    if len(corners) > 3:
                        try:
                            room_back = orient_room(corners,rotate_degree)
                            gen_R_distri[r,i,:len(corners)] = np.array(pts)
                            # coords = np.array(room_back.exterior.coords[:-1])
                            cv2.fillPoly(imgs[r], [np.array(room_back.exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], [c for c in color_list_rgb[o]]+[1])
                            cv2.putText(imgs[r],Types_List[o], (int(room_back.centroid.x),int(room_back.centroid.y)), cv2.FONT_HERSHEY_SIMPLEX, 3, (1, 0.1, 0.1, 1), 5)
                        except:
                            continue
        # imgs[r,:,:,-1] += bound_img
    np.clip(imgs, 0, 1, out=imgs)
    return imgs,gen_R_distri

def vecs_R_decode_post_1024(nb_sample,T,R,W,sqe_len,bound,vec,align_r=25,align_b=100,Text=False):

    # decode Attributes
    imgs = np.zeros((nb_sample,1024,1024,4))
    Type_list = np.zeros((nb_sample,sqe_len))
    idxs = np.where(T.cpu().numpy()==1)
    for m in range(len(idxs[0])):
        Type_list[idxs[0][m],idxs[1][m]] = idxs[-1][m]+1
    R = R.cpu().numpy()
    W = W.cpu().numpy()
    gen_R = R.dot(1 << np.arange(R.shape[-1] - 1, -1, -1))*8
    gen_w = W.dot(1 << np.arange(W.shape[-1] - 1, -1, -1))*8

    gen_R_distri = np.zeros_like(gen_R)

    # print(Type_list)
    # print(gen_R[0])
    rotate_degree = np.arccos(vec.dot(np.array([0,1])))*180/np.pi
    out_bound = Polygon([[0,0],[0,1023],[1023,1023],[1023,0]]).difference(Polygon(bound))
    bound_geo = Polygon(bound)
    # decode Region
    # Region_list = ((gen_R[:,:,:,:,0]>0.9)&(gen_R[:,:,:,:,1]>0.9)&(gen_R[:,:,:,:,2]>0.9)&(gen_R[:,:,:,:,3]>0.9)).astype(int)
    for r in range(nb_sample):
        unioned_rooms = None
        rooms = []
        types = []
        living = None
        cv2.fillPoly(imgs[r], [np.array(bound_geo.buffer(15).exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], [0.5]*3+[1])
        cv2.fillPoly(imgs[r], [np.array(bound_geo.exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], [0]*4)
        for o in room_order:#room_order_reverse
            for i in range(sqe_len):
                if Type_list[r,i] == o and not np.array_equal(gen_R[r,i],np.zeros((20,2))):
                    corners,pts = del_non(gen_R[r,i],bound_geo)#gen_R[r, i][~np.all(gen_R[r, i] == 0, axis=1)]
                    if len(corners) > 3:
                        room_back = orient_room(corners,rotate_degree)
                        gen_R_distri[r,i,:len(corners)] = np.array(pts)
                        coords = np.array(room_back.exterior.coords[:-1])
                        if unioned_rooms:
                            unioned_lines = []

                            if unioned_rooms.geom_type == 'MultiPolygon':
                                for g in unioned_rooms.geoms:
                                    unioned_lines.append(g.exterior)

                            else:
                                unioned_lines = [unioned_rooms.exterior]

                            for t in range(4):

                                pt1,pt2 = coords[t],coords[(t+1)%4]
                                for unioned_line in unioned_lines:
                                    pt1,pt2 = align_target(pt1,pt2,unioned_line,align_r)
                                pt1,pt2 = align_target(pt1,pt2,bound,align_b)

                                coords[t],coords[(t+1)%4] = pt1,pt2
                            room = Polygon(coords).minimum_rotated_rectangle.difference(unioned_rooms)
                            room = room.difference(out_bound)
                            if room.geom_type == 'MultiPolygon':
                                room = room.geoms[find_max(room.geoms)]
                            if room.geom_type == 'GeometryCollection':
                                room = room.geoms[0]

                            if room.geom_type == 'Polygon':
                                coorners = np.array(room.exterior.coords[:-1])
                                if 250000 > room.area > 1 and coorners.ndim>1 and coorners.shape[0]>2:

                                    if len(unioned_lines) > 1:
                                        geolis = []
                                        for l in unioned_rooms.geoms:
                                            geolis.append(l)
                                        geolis.append(room)
                                        unioned_rooms = unary_union(MultiPolygon(geolis))
                                    else:
                                        if unioned_rooms.geom_type == 'MultiPolygon':
                                            unioned_rooms = unioned_rooms.geoms[0]
                                        unioned_rooms = unary_union(MultiPolygon([room,unioned_rooms]))
                                    
                                    cv2.fillPoly(imgs[r], [coorners[:,np.newaxis,:].astype(np.int32)], [0]*4)
                                    cv2.fillPoly(imgs[r], [coorners[:,np.newaxis,:].astype(np.int32)], [c for c in color_list_rgb[o]]+[1])
                                    rooms.append(room_back)
                                    types.append(Types_List[o])
                                    if o == 1: living = room
                        else:

                            for t in range(4):

                                pt1,pt2 = coords[t],coords[(t+1)%4]
                                pt1,pt2 = align_target(pt1,pt2,bound,align_r)
                                coords[t],coords[(t+1)%4] = pt1,pt2

                            unioned_rooms = Polygon(coords).difference(out_bound)
                            if unioned_rooms.geom_type == 'MultiPolygon':
                                unioned_rooms = unioned_rooms.geoms[find_max(unioned_rooms.geoms)]
                            coorners = np.array(unioned_rooms.exterior.coords[:-1])
                            if coorners.ndim>1 and coorners.shape[0]>2:
                                cv2.fillPoly(imgs[r], [coorners[:,np.newaxis,:].astype(np.int32)], [c for c in color_list_rgb[o]]+[1])
                                # cv2.fillPoly(imgs[r], [np.array(room_back.exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], [c for c in color_list_rgb[o]]+[1])
                                rooms.append(room_back)
                                types.append(Types_List[o])

        if unioned_rooms.geom_type == 'MultiPolygon':
            remained = bound_geo
            for h in unioned_rooms.geoms:
                remained = remained.difference(h)
        else:
            remained = bound_geo.difference(unioned_rooms)

        if remained.geom_type == 'MultiPolygon':
            for g in remained.geoms:
                if g.buffer(10).intersects(living):
                    cv2.fillPoly(imgs[r], [np.array(g.exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], [0]*4)
                    cv2.fillPoly(imgs[r], [np.array(g.exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], [c for c in color_list_rgb[1]]+[1])
                else:
                    cv2.fillPoly(imgs[r], [np.array(g.exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], [0]*4)
                    cv2.fillPoly(imgs[r], [np.array(g.exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], [c for c in color_list_rgb[7]]+[1])
                    rooms.append(g)
                    types.append(Types_List[7])
        elif remained.geom_type == 'Polygon':
            if remained.buffer(10).intersects(living):
                cv2.fillPoly(imgs[r], [np.array(remained.exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], [0]*4)
                cv2.fillPoly(imgs[r], [np.array(remained.exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], [c for c in color_list_rgb[1]]+[1])
            else:
                cv2.fillPoly(imgs[r], [np.array(remained.exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], [0]*4)
                cv2.fillPoly(imgs[r], [np.array(remained.exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], [c for c in color_list_rgb[7]]+[1])
                rooms.append(remained)
                types.append(Types_List[7])
        if Text:     
            for ro in range(len(rooms)): cv2.putText(imgs[r],types[ro], (int(rooms[ro].centroid.x),int(rooms[ro].centroid.y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (1, 0.3, 0.3, 1), 4)
        
    np.clip(imgs, 0, 1, out=imgs)
    return imgs,gen_R_distri

def vecs_R_decode_post(nb_sample,T,R,W,sqe_len,bound,vec):

    # decode Attributes
    imgs = np.zeros((nb_sample,128,128,4))
    Type_list = np.zeros((nb_sample,sqe_len))
    idxs = np.where(T.cpu().numpy()==1)
    for m in range(len(idxs[0])):
        Type_list[idxs[0][m],idxs[1][m]] = idxs[-1][m]+1
    R = R.cpu().numpy()
    W = W.cpu().numpy()
    gen_R = R.dot(1 << np.arange(R.shape[-1] - 1, -1, -1))
    gen_w = W.dot(1 << np.arange(W.shape[-1] - 1, -1, -1))

    print(Type_list)
    # print(gen_R[0])
    rotate_degree = np.arccos(vec.dot(np.array([0,1])))*180/np.pi
    out_bound = Polygon([[0,0],[0,127],[127,127],[127,0]]).difference(Polygon(bound))
    unioned_rooms = None
    # decode Region
    # Region_list = ((gen_R[:,:,:,:,0]>0.9)&(gen_R[:,:,:,:,1]>0.9)&(gen_R[:,:,:,:,2]>0.9)&(gen_R[:,:,:,:,3]>0.9)).astype(int)
    for r in range(nb_sample):
        for o in room_order:
            for i in range(sqe_len):
                if Type_list[r,i] == o and not np.array_equal(gen_R[r,i],np.zeros((20,2))):
                    corners,pts = del_non(gen_R[r,i],Polygon(bound))
                    if len(corners) > 3:
                        room_back = gen_R[r, i][~np.all(gen_R[r, i] == 0, axis=1)]#orient_room(corners,rotate_degree)
                        coords = np.array(room_back.exterior.coords[:-1])
                        # cv2.fillPoly(imgs[r], [np.array(room_back.exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], color_list_rgb[o]+[255])
                        # cv2.putText(imgs[r],Types_List[o], (int(room_back.centroid.x),int(room_back.centroid.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 20, 20, 255), 1)
                        # print(room_back.area)
                        if unioned_rooms:
                            unioned_lines = []

                            if unioned_rooms.geom_type == 'MultiPolygon':
                                for g in unioned_rooms.geoms:
                                    unioned_lines.append(g.exterior)

                            else:
                                unioned_lines = [unioned_rooms.exterior]

                            for t in range(4):

                                pt1,pt2 = coords[t],coords[(t+1)%4]
                                for unioned_line in unioned_lines:
                                    pt1,pt2 = align_target(pt1,pt2,unioned_line,3)
                                pt1,pt2 = align_target(pt1,pt2,bound,3+5)

                                coords[t],coords[(t+1)%4] = pt1,pt2
                            room = Polygon(coords).minimum_rotated_rectangle.difference(unioned_rooms)
                            room = room.difference(out_bound)
                            if room.geom_type == 'MultiPolygon':
                                room = room.geoms[find_max(room.geoms)]
                            if room.geom_type == 'GeometryCollection':
                                room = room.geoms[0]
                            if 2500 > room.area > 0.1:

                                if len(unioned_lines) > 1:
                                    geolis = []
                                    for l in unioned_rooms.geoms:
                                        geolis.append(l)
                                    geolis.append(room)
                                    unioned_rooms = unary_union(MultiPolygon(geolis))
                                else:
                                    if unioned_rooms.geom_type == 'MultiPolygon':
                                        unioned_rooms = unioned_rooms.geoms[0]
                                    unioned_rooms = unary_union(MultiPolygon([room,unioned_rooms]))
                                
                                cv2.fillPoly(imgs[r], [np.array(room.exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], color_list_rgb[o]+[255])
                                cv2.putText(imgs[r],Types_List[o], (int(room.centroid.x),int(room.centroid.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 20, 20, 255), 1)
                                print(room.area)
                        else:

                            for t in range(4):

                                pt1,pt2 = coords[t],coords[(t+1)%4]
                                pt1,pt2 = align_target(pt1,pt2,bound,3)
                                coords[t],coords[(t+1)%4] = pt1,pt2

                            unioned_rooms = Polygon(coords).difference(out_bound)
                            if unioned_rooms.geom_type == 'MultiPolygon':
                                unioned_rooms = unioned_rooms.geoms[find_max(unioned_rooms.geoms)]
                            coorners = np.array(unioned_rooms.exterior.coords[:-1])
                            if coorners.ndim>1 and coorners.shape[0]>2:
                                cv2.fillPoly(imgs[r], [coorners[:,np.newaxis,:].astype(np.int32)], color_list_rgb[o]+[255])
                                cv2.putText(imgs[r],Types_List[o], (int(unioned_rooms.centroid.x),int(unioned_rooms.centroid.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 20, 20, 255), 1)
                                print(unioned_rooms.area)

    # decode Window
    # Window_list = np.sum(((gen_W[:,:,:,0]>0.9)&(gen_W[:,:,:,1]>0.9)&(gen_W[:,:,:,2]>0.9)&(gen_W[:,:,:,3]>0.9)).astype(int),axis=0)*255
    # for i in range(4): img_2[:,:,i] += Window_list

    return imgs


def del_non(sqe,bound):
    new = []
    pts = []
    for pt in sqe:
        if pt[0] == 0 or pt[1] == 0:
            continue
        elif bound.contains(Point(pt)):
            new.append(pt)
            pts.append([Point(pt).x,Point(pt).y])
    return new,pts

def find_max(multi_geo):
    areas = []
    for m in multi_geo:
        areas.append(m.area)
    idx = areas.index(max(areas))

    return idx

def orient_room(pts,rotate_degree):
    
    room = MultiPoint(pts)
    cen = room.centroid
    rotated = affinity.rotate(room, rotate_degree,cen)
    rectan = rotated.envelope
    room_back = affinity.rotate(rectan, -rotate_degree,cen)
    return room_back

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


def draw_approx_hull_polygon(img, cnts, T):

    img = np.zeros(img.shape, dtype=np.uint8)

    epsilion = img.shape[0]/128
    approxes = [cv2.approxPolyDP(cnt, epsilion, True) for cnt in cnts]
    cv2.fillPoly(img, approxes, T)

    return img,approxes


def decode_loc_square(center,vec,size):

    pt1 = (center[0] + size*(vec[1]-vec[0]),center[1] - size*(vec[1]+vec[0]))
    pt2 = (center[0] + size*(vec[1]+vec[0]),center[1] + size*(vec[1]-vec[0]))
    pt3 = (center[0] + size*(vec[0]-vec[1]),center[1] + size*(vec[1]+vec[0]))
    pt4 = (center[0] - size*(vec[1]+vec[0]),center[1] - size*(vec[1]-vec[0]))

    return [pt1,pt2,pt3,pt4,pt1]


def get_bound_pt(img):

    _,thresh = cv2.threshold(img, 120, 255, 0)
    contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    epsilion = img.shape[0]/128
    approxes = [cv2.approxPolyDP(cnt, epsilion, True) for cnt in contours]
    
    return np.squeeze(approxes[0])


import math

def sort_points_clockwise(points):
    # Calculate the centroid of the points
    centroid_x = sum(x for x, y in points) / len(points)
    centroid_y = sum(y for x, y in points) / len(points)
    
    # Function to compute the angle between the centroid and a point
    def calculate_angle(x, y):
        # Calculate the angle in radians
        angle = math.atan2(y - centroid_y, x - centroid_x)
        # Convert to degrees and adjust to ensure clockwise sorting
        return (math.degrees(angle) + 360) % 360
    
    # Sort the points based on the calculated angle
    sorted_points = sorted(points, key=lambda point: calculate_angle(point[0], point[1]), reverse=True)
    
    return sorted_points

# Example usage
points = [(1, 1), (2, 3), (3, 2), (2, 1)]
sorted_points = sort_points_clockwise(points)
print(sorted_points)


def align_target(pt1,pt2,target,gap):

    vector1to2 = pt2 - pt1
    norl_vec1to2 = vector1to2/np.linalg.norm(vector1to2)
    vector2to1 = pt1 - pt2
    norl_vec2to1 = vector2to1/np.linalg.norm(vector2to1)

    extended_pt1 = pt1 + norl_vec2to1*gap
    extended_pt2 = pt2 + norl_vec1to2*gap

    extended_pt1 = LineString([extended_pt1.tolist(),pt1.tolist()])
    extended_pt2 = LineString([extended_pt2.tolist(),pt2.tolist()])

    inter_pt1 = extended_pt1.intersection(target)
    inter_pt2 = extended_pt2.intersection(target)

    if inter_pt1.geom_type == 'Point':
        pt1 = np.array(inter_pt1.coords[:][0]).astype(np.int32)
    if inter_pt2.geom_type == 'Point':
        pt2 = np.array(inter_pt2.coords[:][0]).astype(np.int32)

    return pt1,pt2

def scale_to_img(ratio,Mx,My,geo):

    return np.array([[[x*ratio + Mx, y*ratio + My] for x, y in zip(*geo.exterior.coords.xy)]])

def get_boundbox(center,vec,size):

    pt1 = (center.x + size*(vec[1]-vec[0]),center.y - size*(vec[1]+vec[0]))
    pt2 = (center.x + size*(vec[1]+vec[0]),center.y + size*(vec[1]-vec[0]))
    pt3 = (center.x + size*(vec[0]-vec[1]),center.y + size*(vec[1]+vec[0]))
    pt4 = (center.x - size*(vec[1]+vec[0]),center.y - size*(vec[1]-vec[0]))

    return [pt1,pt2,pt3,pt4,pt1]


def euclidean_distance(set1, set2):
    """
    Compute the Euclidean distance between two sets.
    """
    return np.linalg.norm(np.array(set1) - np.array(set2))

def compute_distance_matrix(sets):
    """
    Compute the pairwise distance matrix for all sets.
    """
    N = len(sets)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            D[i, j] = euclidean_distance(sets[i], sets[j])
            D[j, i] = D[i, j]  # Distance matrix is symmetric
    return D

def most_diverse_4_sets(sets):
    """
    Find the 4 most diverse sets from a list of sets.
    """
    N = len(sets)
    if N < 4:
        raise ValueError("At least 4 sets are required.")

    # Compute the distance matrix
    D = compute_distance_matrix(sets)

    # Initialize the selected sets
    selected = []
    remaining = list(range(N))

    # Step 1: Select the two sets with the maximum distance
    max_dist = -1
    for i, j in combinations(remaining, 2):
        if D[i, j] > max_dist:
            max_dist = D[i, j]
            selected = [i, j]
    remaining = [x for x in remaining if x not in selected]

    # Step 2: Iteratively add the set that maximizes the total distance
    for _ in range(2):
        max_total_dist = -1
        best_candidate = -1
        for candidate in remaining:
            total_dist = sum(D[candidate, s] for s in selected)
            if total_dist > max_total_dist:
                max_total_dist = total_dist
                best_candidate = candidate
        selected.append(best_candidate)
        remaining.remove(best_candidate)

    # Return the selected sets
    return selected#[sets[i] for i in selected]