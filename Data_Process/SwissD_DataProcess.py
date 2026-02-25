from shapely.geometry import Polygon,MultiPolygon,LineString,Point
from shapely.plotting import plot_polygon, plot_line, plot_points
import numpy as np
import json
from shapely.ops import unary_union
from .utils import *
import cv2
import imageio
from shapely.validation import make_valid,explain_validity

class Swiss_Dewelling_Dual_Modality():

    def __init__(self,bound_len,color_list,room_types,type_index,img_res,num_room,T_dim,L_dim,S_dim,R_num,W_num,divide):

        self.bound_len = bound_len
        self.color_list = color_list
        self.room_types = room_types
        self.type_index = type_index
        self.num_room = num_room
        self.img_res = img_res
        self.T_dim = T_dim
        self.L_dim = L_dim
        self.S_dim = S_dim
        self.S_max = 2**S_dim
        self.R_num = R_num
        self.W_num = W_num
        self.divide = divide

        # vec, +1 for [end] token
        self.T_vec = np.zeros((num_room+1,T_dim))
        self.L_vec = np.zeros((num_room+1,2,L_dim))
        self.A_vec = np.zeros((num_room+1,num_room))
        self.S_vec = np.zeros((num_room+1,S_dim))
        self.R_vec = np.zeros((num_room+1,R_num,2,L_dim))
        self.W_vec = np.zeros((W_num+1,2,L_dim)) 

        # img
        self.bound_img = np.zeros((img_res,img_res,4))
        self.T_img = []
        self.L_img = []
        self.A_img = []
        self.S_img = [] 
        self.R_img = []
        self.W_img = []

        self.count = 0
        self.front_door = None
    

    def reset(self):

        self.T_vec[:] = 0
        self.L_vec[:] = 0
        self.A_vec[:] = 0
        self.S_vec[:] = 0
        self.R_vec[:] = 0
        self.W_vec[:] = 0

        self.bound_img[:] = 0
        self.T_img = []
        self.L_img = []
        self.A_img = []
        self.S_img = [] 
        self.R_img = []
        self.W_img = []

        self.front_door = None


    def transfer_polygons(self,tmp,indx,contest):

        # real-world data
        corners_house,extended_house,tpyes_house,types_names,house_centers,house_areas,front_door,doorsPt,windows\
            = parse_swiss_dewelling_polygon(tmp.index,tmp['geometry'],tmp['entity_subtype'],self.room_types,self.type_index)

        if tpyes_house != [] and len(tpyes_house) < self.num_room + 1:

            # parse boundary
            floorplan_house_polygon = MultiPolygon(corners_house)
            boundary = unary_union(floorplan_house_polygon.buffer(0.05,join_style=2))
            while boundary.geom_type == 'MultiPolygon': boundary = boundary.buffer(0.1,join_style=2)
            if list(boundary.interiors) != []:
                for inte in list(boundary.interiors):
                    boundary = unary_union(MultiPolygon([boundary,Polygon(inte).buffer(0.1,join_style=2)]))
            
            # boundary domain
            self.boundary_domainxy = boundary.bounds # (minx, miny, maxx, maxy)
            boundx = self.boundary_domainxy[2] - self.boundary_domainxy[0]
            boundy = self.boundary_domainxy[3] - self.boundary_domainxy[1]
            self.boundary = boundary

            # fit in domain
            if boundx <= self.bound_len and boundy <= self.bound_len:

                self.indx = indx

                # get adjacency as list
                ada_list = get_adjacency_graph(MultiPolygon(extended_house))

                # get bondary dominant vec
                boundary_corners = list(boundary.minimum_rotated_rectangle.exterior.coords)
                vector = find_domain_vec(boundary_corners)
                normalized_vector = vector/np.linalg.norm(vector)
                self.vec = normalized_vector

                # parse front door
                if front_door == None:
                    front_door_corners = [[0,0],[0,1],[1,1],[0,0]]
                    front_door = Polygon(front_door_corners)
                else:
                    front_door_corners = list(front_door.exterior.coords)
                frondoor_center = front_door.centroid
                new_frontdoor = Polygon(get_boundbox(frondoor_center,normalized_vector,0.75))

                # contest and boundary union
                contest_cut = contest.difference(boundary)
                contest_cut = contest_cut.difference(new_frontdoor)
                boundary_cut = boundary.difference(new_frontdoor)
                outter_boundary = self.boundary.buffer(0.4,join_style=2)
                if contest_cut.geom_type == 'MultiPolygon': contest_cut = find_maxgeo(contest_cut)
                if boundary_cut.geom_type == 'MultiPolygon': boundary_cut = find_maxgeo(boundary_cut)
                if outter_boundary.geom_type == 'MultiPolygon': outter_boundary = find_maxgeo(outter_boundary)

                # scale to img size
                self.factor = self.img_res/self.bound_len
                boudnary_center = [boundary.centroid.x*self.factor,boundary.centroid.y*self.factor]
                move_x =  self.img_res/2 - boudnary_center[0]
                move_y =  self.img_res/2 - boudnary_center[1]
                self.front_door = np.array([int(frondoor_center.x*self.factor + move_x),int(frondoor_center.y*self.factor + move_y)])
                boundary_cut_coords = scale_to_img(self.factor,move_x,move_y,boundary_cut)
                outter_boundary_coords = scale_to_img(self.factor,move_x,move_y,outter_boundary)
                contest_cut_coords = scale_to_img(self.factor,move_x,move_y,contest_cut)
                frontdoor_coords = scale_to_img(self.factor,move_x,move_y,new_frontdoor)

                # cut out of box region
                boundary_cut_coords[np.where(boundary_cut_coords>(self.img_res-1))] = int(self.img_res-1)
                boundary_cut_coords[np.where(boundary_cut_coords<0)] = 0
                contest_cut_coords[np.where(contest_cut_coords>(self.img_res-1))] = int(self.img_res-1)
                contest_cut_coords[np.where(contest_cut_coords<0)] = 0

                # init RGB and Alpha chanel
                temp_boundAlpha = np.zeros((self.img_res,self.img_res))
                temp_boundrgb = np.zeros((self.img_res,self.img_res))
                self.attri_Alpha = np.zeros((self.img_res,self.img_res))

                # site image
                # front door: [255,255,255,255], boundary: [127,127,127,255], contest :[0,0,0,255]
                temp_boundAlpha = cv2.fillPoly(temp_boundAlpha, contest_cut_coords.astype(np.int32), color=255)
                temp_boundAlpha = cv2.fillPoly(temp_boundAlpha, boundary_cut_coords.astype(np.int32), color=255)
                temp_boundrgb = cv2.fillPoly(temp_boundrgb, boundary_cut_coords.astype(np.int32), color=127)
                temp_boundAlpha = cv2.fillPoly(temp_boundAlpha, frontdoor_coords.astype(np.int32), color=255)
                temp_boundrgb = cv2.fillPoly(temp_boundrgb, frontdoor_coords.astype(np.int32), color=255)

                # Alpha in all attributes: boundary: [127], front door: [255]
                self.attri_Alpha = cv2.fillPoly(self.attri_Alpha, outter_boundary_coords.astype(np.int32), color=127)
                self.attri_Alpha = cv2.fillPoly(self.attri_Alpha, boundary_cut_coords.astype(np.int32), color=0)
                self.attri_Alpha = cv2.fillPoly(self.attri_Alpha, frontdoor_coords.astype(np.int32), color=255)
                self.inside_mask = boundary_cut_coords

                # assgin to bound_img
                self.bound_img[:,:,-1] = temp_boundAlpha
                for m in range(3): self.bound_img[:,:,m] = temp_boundrgb
                cv2.imwrite('Data/img/site_1024/%d.png' % (indx), self.bound_img)
                cv2.imwrite('Data/img/all/%d.png' % (self.count), self.bound_img)
                self.count += 1

                # save [start] token imgs for T,L,A,S,R,W
                start_fill = np.zeros((self.img_res,self.img_res,4))
                start_fill[:,:,-1] = self.attri_Alpha.copy()
                start_fill = cv2.fillPoly(start_fill, self.inside_mask.astype(np.int32), color=[255]*4)
                fill_list = ['T','L','A','S','R','W']
                for i in fill_list:
                    cv2.imwrite('Data/img/%s/0/%d.png' % (i,indx), start_fill)
                cv2.imwrite('Data/img/all/%d.png' % (self.count), start_fill)
                self.count += 1

                # real-world data to json # not used
                # file_json = 'parsed/graph/%d.json' % (indx) #'parsed/graph/%d_%s.json' % (indx, id_name)
                # floorplan_to_Json(file_json,floorplan_house_polygon,types_names,house_centers,
                #     tpyes_house,ada_list,house_areas,bound_all_corners,front_door_corners,windowsPt,doorsPt)

                # re-order rooms with livingroom first
                self.reorder(move_x,move_y,tpyes_house,house_centers,house_areas,ada_list,corners_house,windows)
                
                # save img
                self.save_img()

                return 1
            
        else:

            return 0


    def reorder(self,Mx,My,T,C,S,A,R,W):

        # Living Room First 
        # T type C location S size A adjacency R region W window

        window_count = len(W)
        self.R_region = []

        if 1 in T: # Living Room existing in layout

            living = np.where(np.array(T) == 1)[0]
            ind_living = int(living[0])
            self.ind_living = ind_living

            # vec
            self.T_vec[0,0] = 1
            Lx, Ly = int(C[ind_living][0]*self.factor + Mx), int(C[ind_living][1]*self.factor + My)
            self.L_vec[0,0],self.L_vec[0,1] = make_bit(Lx,self.L_dim),make_bit(Ly,self.L_dim)
            if S[ind_living] <= self.S_max: s = int(S[ind_living])
            else: s = int(self.S_max-1)
            self.S_vec[0] = make_bit(s,self.S_dim)
            
            for i,j in enumerate(list(R[ind_living].exterior.coords)):
                if i > self.R_num - 1: break
                x, y = int(j[0]*self.factor + Mx), int(j[1]*self.factor + My)
                if x < 0: x = 1
                if y < 0: y = 1
                if x > self.img_res-1: x = self.img_res-1
                if y > self.img_res-1: y = self.img_res-1
                # self.R_vec[0,i,0] = make_bit(x, self.L_dim)
                # self.R_vec[0,i,1] = make_bit(y, self.L_dim)

            # re-order A graph

            if ind_living != 0:
                A[np.where(A == ind_living)] = 100
                A[np.where(A < ind_living)] += 1
                A[np.where(A == 100)] = 0

            # del the rest living
            if living.shape[0] > 1: 
                del_ind = living[1:]
                for l in del_ind:
                    A[np.where(A == l)] = 100
                    A[np.where(A > ind_living)] -= 1

            for n in A:
                if n[0] < 90 and n[1] < 90:
                    self.A_vec[n[0]][n[1]] = 1
                    self.A_vec[n[1]][n[0]] = 1
            
            # img
            self.T_img.append(1)
            self.L_img.append(scale_to_img(self.factor,Mx,My,Polygon(get_boundbox(Point(C[ind_living]),self.vec,0.5))))
            self.S_img.append(scale_to_img(self.factor,self.img_res/2,self.img_res/2,Polygon(get_boundbox(Point((0,0)),self.vec,np.sqrt(s)/2))))
            self.R_img.append(scale_to_img(self.factor,Mx,My,R[ind_living]))
            self.R_region.append(R[ind_living])
            
            # parse the rest
            vindx = 1 # vec index
            ind = 0 # retrieve index
            for n in T:

                if n == 1: ind += 1

                else:

                    # vec
                    self.T_vec[vindx,int(n-1)] = 1
                    Lx, Ly = int(C[ind][0]*self.factor + Mx), int(C[ind][1]*self.factor + My)
                    self.L_vec[vindx,0],self.L_vec[vindx,1] = make_bit(Lx,self.L_dim),make_bit(Ly,self.L_dim)
                    if S[ind] <= self.S_max: s = int(S[ind])
                    else: s = int(self.S_max-1)
                    self.S_vec[vindx] = make_bit(s,self.S_dim)

                    for i,j in enumerate(list(R[ind].exterior.coords)):
                        if i > self.R_num - 1: break
                        x, y = int(j[0]*self.factor + Mx), int(j[1]*self.factor + My)
                        if x < 0: x = 1
                        if y < 0: y = 1
                        if x > self.img_res-1: x = self.img_res-1
                        if y > self.img_res-1: y = self.img_res-1
                        # self.R_vec[vindx,i,0] = make_bit(x, self.L_dim)
                        # self.R_vec[vindx,i,1] = make_bit(y, self.L_dim)

                    # img
                    self.T_img.append(n)
                    self.L_img.append(scale_to_img(self.factor,Mx,My,Polygon(get_boundbox(Point(C[ind]),self.vec,0.5))))
                    self.S_img.append(scale_to_img(self.factor,self.img_res/2,self.img_res/2,Polygon(get_boundbox(Point((0,0)),self.vec,np.sqrt(s)/2))))
                    self.R_img.append(scale_to_img(self.factor,Mx,My,R[ind]))
                    self.R_region.append(R[ind])

                    vindx += 1
                    ind += 1
        
        else: # no living room in layout

            for n in A:
                self.A_vec[n[0]][n[1]] = 1
                self.A_vec[n[1]][n[0]] = 1
            
            vindx = 0
            for n in T:

                # vec
                self.T_vec[vindx,int(n-1)] = 1
                Lx, Ly = int(C[vindx][0]*self.factor + Mx), int(C[vindx][1]*self.factor + My)
                self.L_vec[vindx,0],self.L_vec[vindx,1] = make_bit(Lx,self.L_dim),make_bit(Ly,self.L_dim)
                if S[vindx] <= self.S_max: s = int(S[vindx])
                else: s = int(self.S_max-1)

                self.S_vec[vindx] = make_bit(s,self.S_dim)
                
                for i,j in enumerate(list(R[vindx].exterior.coords)):
                    if i > self.R_num - 1: break
                    x, y = int(j[0]*self.factor + Mx), int(j[1]*self.factor + My)
                    if x < 0: x = 1
                    if y < 0: y = 1
                    if x > self.img_res-1: x = self.img_res-1
                    if y > self.img_res-1: y = self.img_res-1
                    self.R_vec[vindx,i,0] = make_bit(x, self.L_dim)
                    self.R_vec[vindx,i,1] = make_bit(y, self.L_dim)

                # img
                self.T_img.append(n)
                self.L_img.append(scale_to_img(self.factor,Mx,My,Polygon(get_boundbox(Point(C[vindx]),self.vec,0.5))))
                self.S_img.append(scale_to_img(self.factor,self.img_res/2,self.img_res/2,Polygon(get_boundbox(Point((0,0)),self.vec,np.sqrt(s)/2))))
                self.R_img.append(scale_to_img(self.factor,Mx,My,R[vindx]))
                self.R_region.append(R[vindx])

                vindx += 1
        
        room_count = len(self.T_img)

        # parse window corresponds to each room
        for i,j in enumerate(self.R_region):
            if len(self.W_img) > self.W_num: break
            cross = False
            for m,n in enumerate(W):
                window_box = Polygon(get_boundbox(n.centroid,self.vec,0.4))
                if window_box.intersects(j):
                    Lx, Ly = int(n.centroid.x*self.factor + Mx), int(n.centroid.y*self.factor + My)
                    if Lx < 0: Lx = 1
                    if Ly < 0: Ly = 1
                    if Lx > self.img_res-1: Lx = self.img_res-1
                    if Ly > self.img_res-1: Ly = self.img_res-1
                    self.W_vec[i,0],self.W_vec[i,1] = make_bit(Lx,self.L_dim),make_bit(Ly,self.L_dim)
                    self.W_img.append(scale_to_img(self.factor,Mx,My,window_box))
                    cross = True
                    break
            if cross == False: # no window room
                self.W_img.append(np.zeros(0))
            

        window_count = len(self.W_img)

        # parse A img
        for m in range(room_count):
            ada_list = []
            for n in range(self.num_room):
                if self.A_vec[m,n] == 1: ada_list.append([LineString([Polygon(self.L_img[m][0]).centroid,Polygon(self.L_img[n][0]).centroid]).buffer(0.5),n])
            self.A_img.append(ada_list)

        # parse end token
        self.T_vec[room_count,-2] = 1
        self.L_vec[room_count] = np.ones((2,self.L_dim))
        self.L_vec[room_count,:,-1] = 0
        self.A_vec[room_count] = np.array([0,1]*int(self.num_room/2))
        self.S_vec[room_count] = np.ones(self.S_dim)
        self.S_vec[room_count,-1] = 0
        self.R_vec[room_count] = np.ones((self.R_num,2,self.L_dim))
        self.R_vec[room_count,:,:,-1] = 0
        self.W_vec[window_count] = np.ones((2,self.L_dim))
        self.W_vec[window_count,:,-1] = 0

        # save [end] token imgs for T,L,A,S,R,W
        end_fill = np.zeros((self.img_res,self.img_res,4))
        end_fill[:,:,-1] = self.attri_Alpha.copy()
        end_fill = cv2.fillPoly(end_fill, self.inside_mask.astype(np.int32), color=[127]*4)
        fill_list = ['T','L','A','S','R']
        # for i in fill_list:
        #     cv2.imwrite('Data/img/%s/%d/%d.png' % (i,room_count+1,self.indx), end_fill)
        # cv2.imwrite('Data/img/W/%d/%d.png' % (window_count+1,self.indx), end_fill)
        # cv2.imwrite('Data/img/all/%d.png' % (self.count), end_fill)
        self.count += 1

    def save_img(self):

        # attri_fill = np.zeros((self.img_res,self.img_res,4))
        # attri_fill[:,:,-1] = self.attri_Alpha.copy()
        
        for m,n in enumerate(self.T_img):

            temp_room = np.zeros((128,128,3))

            cv2.fillPoly(temp_room, self.R_img[m].astype(np.int32), color=[255]*3)
            cv2.imwrite('temp.jpg',temp_room)
            temp = cv2.imread('temp.jpg')[:,:,0]

            _, thresh2 = cv2.threshold(temp, 120, 255, 0)
            contours2, _ = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            im = np.zeros(temp_room.shape, dtype=np.uint8)
            epsilion = im.shape[0]/self.divide
            approxes = [cv2.approxPolyDP(cnt, epsilion, True) for cnt in contours2]
            for i,pt in enumerate(approxes[0]):
                self.R_vec[m,i,0] = make_bit(int(pt[0,0]), self.L_dim)
                self.R_vec[m,i,1] = make_bit(int(pt[0,1]), self.L_dim)



        #     # T imgs
        #     self.fill_img(attri_fill,self.inside_mask,self.color_list[n],'T',m)

        #     # L imgs
        #     self.fill_img(attri_fill,self.L_img[m],[255]*4,'L',m)

        #     # A imgs
        #     fill = attri_fill.copy()
        #     cv2.fillPoly(fill, self.L_img[m].astype(np.int32), color=[255]*4)
        #     for i,g in enumerate(self.A_img[m]):
        #         graph_coords = g[0].exterior.coords[:-1]
        #         cv2.fillPoly(fill, [np.array(graph_coords)[:,np.newaxis,:].astype(np.int32)], color=[255]*4)
        #         cv2.fillPoly(fill, self.L_img[g[1]].astype(np.int32), color=[255]*4)
        #     cv2.imwrite('Data/img/A/%d/%d.png' % (m+1,self.indx), fill)
        #     cv2.imwrite('Data/img/all/%d.png' % (self.count), fill)
        #     self.count += 1

        #     # S imgs
        #     self.fill_img(attri_fill,self.S_img[m],[255]*4,'S',m)

        #     # R imgs
        #     self.fill_img(attri_fill,self.R_img[m],[255]*4,'R',m)

        # # W imgs
        # for m,n in enumerate(self.W_img): 
        #     if n.shape[0] != 0: self.fill_img(attri_fill,n,[255]*4,'W',m)

        # self.parse_composed_layout()

    def fill_img(self,base_img,geo,color,attri,img_indx):
        
        fill = base_img.copy()
        fill = cv2.fillPoly(fill, geo.astype(np.int32), color=color)
        cv2.imwrite('Data/img/%s/%d/%d.png' % (attri,img_indx+1,self.indx), fill)
        cv2.imwrite('Data/img/all/%d.png' % (self.count), fill)
        self.count += 1

    def parse_composed_layout(self):

        composed = np.zeros((self.img_res,self.img_res,4))
        composed[:,:,-1] = self.attri_Alpha.copy()
        
        for m,n in enumerate(self.T_img):
            cv2.fillPoly(composed, self.R_img[m].astype(np.int32), color=self.color_list[n])
            for i,g in enumerate(self.A_img[m]):
                graph_coords = g[0].exterior.coords[:-1]
                cv2.fillPoly(composed, [np.array(graph_coords)[:,np.newaxis,:].astype(np.int32)], color=[255]*4)
                cv2.fillPoly(composed, self.L_img[g[1]].astype(np.int32), color=[255]*4)
            cv2.fillPoly(composed, self.L_img[m].astype(np.int32),[255]*4)
        
        for m,n in enumerate(self.W_img): 
            if n.shape[0] != 0: cv2.fillPoly(composed, n.astype(np.int32), color=[255]*4)

        cv2.imwrite('Data/img/composed_all/%d.png' % (self.indx), composed)
