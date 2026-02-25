from shapely.geometry import Polygon,MultiPolygon,Point,LineString,MultiLineString
import numpy as np
import json
from shapely.ops import unary_union
import tensorflow
from shapely.validation import make_valid,explain_validity
import cv2

def parse_swiss_dewelling_polygon(index_list,geo_data,subtype,house_room_types,house_type_index):

    house_centers = []
    layout_house = []
    rooms_extended = []
    house_type_list = []
    house_type_names = []
    front_door = None
    doors_pt = []
    windows = []
    areas = []
    kitchen = 0
    
    for i in index_list:
        room = geo_data[i][10:-2].split(",")
        room_type = subtype[i]
        room_corners = []

        for j in room:
            xy = j.strip().split(" ")
            if xy[1][-1] == ')':
                xy[1] = xy[1][:-1]
            if xy[0][0] == '(':
                xy[0] = xy[0][1:]
            room_corners.append((float(xy[0]),float(xy[1])))
        
        rooms_poly = Polygon(room_corners)
        extended_poly = rooms_poly.buffer(0.3,join_style=2)
        
        if rooms_poly.geom_type != 'MultiPolygon' and extended_poly.geom_type != 'MultiPolygon':

            if kitchen == 1 and (room_type == 'KITCHEN_DINING' or room_type == 'KITCHEN'):
                indx_kithen = house_type_list.index(5)
                exist = layout_house[indx_kithen]
                if exist.area < rooms_poly.area:
                    layout_house[indx_kithen] = rooms_poly
                    rooms_extended[indx_kithen] = rooms_poly.buffer(0.3,join_style=2)
                    areas[indx_kithen] = rooms_poly.area
                    house_centers[indx_kithen] = [rooms_poly.centroid.x,rooms_poly.centroid.y]

            elif room_type in house_room_types and rooms_poly.area>2:
                house_type_list.append(house_type_index[house_room_types.index(room_type)])
                house_type_names.append(room_type)
                layout_house.append(rooms_poly)
                house_centers.append([rooms_poly.centroid.x,rooms_poly.centroid.y])
                rooms_extended.append(extended_poly)
                areas.append(rooms_poly.area)
                if room_type == 'KITCHEN_DINING' or room_type == 'KITCHEN':
                    kitchen = 1

            elif room_type == 'ENTRANCE_DOOR':
                front_door = rooms_poly

            elif room_type == 'DOOR':
                doors_pt.append(room_corners)

            elif room_type == 'WINDOW':
                windows.append(rooms_poly)
            
    return layout_house,rooms_extended,house_type_list,house_type_names,house_centers,areas,front_door,doors_pt,windows


def draw_approx_hull_polygon(img, cnts, T):
    # img = np.copy(img)
    img = np.zeros(img.shape, dtype=np.uint8)

    epsilion = img.shape[0]/128
    approxes = [cv2.approxPolyDP(cnt, epsilion, True) for cnt in cnts]
    #cv2.polylines(img, approxes, True, (0, 255, 0), 2)  # green
    cv2.fillPoly(img, approxes, T)

    # hulls = [cv2.convexHull(cnt) for cnt in cnts]
    # cv2.polylines(img, hulls, True, (0, 0, 255), 2)  # red

    return img,approxes


def parse_contest(index_list,geo_data):

    layout_all = []
    for i in index_list:
        room = geo_data[i][10:-2].split(",")
        room_corners = []
        for j in room:
            xy = j.strip().split(" ")
            if xy[1][-1] == ')':
                xy[1] = xy[1][:-1]
            if xy[0][0] == '(':
                xy[0] = xy[0][1:]
            room_corners.append((float(xy[0]),float(xy[1])))
        rooms_poly = Polygon(room_corners)
        layout_all.append(rooms_poly)
    floorplan_all_polygon = MultiPolygon(layout_all)
    boundary = unary_union(floorplan_all_polygon.buffer(0.01,join_style=2))
    while boundary.geom_type == 'MultiPolygon':
        boundary = boundary.buffer(0.01,join_style=2)
    return boundary


def parse_rooms(index_list,geo_data,subtype,room_types,types_index):

    rooms = []
    front_door = None
    types = []
    for i in index_list:
        room_type = subtype[i]
        room = geo_data[i][10:-2].split(",")
        room_corners = []
        for j in room:
            xy = j.strip().split(" ")
            if xy[1][-1] == ')':
                xy[1] = xy[1][:-1]
            if xy[0][0] == '(':
                xy[0] = xy[0][1:]
            room_corners.append((float(xy[0]),float(xy[1])))
        rooms_poly = Polygon(room_corners)
        if room_type in room_types and room_type not in ['TERRACE','BALCONY','LOGGIA']:
            rooms.append(rooms_poly)
            types.append(types_index[room_types.index(room_type)])
        elif room_type == 'ENTRANCE_DOOR':
            front_door=rooms_poly

    rooms_poly_segement = MultiPolygon(rooms)
    rooms_polygon = unary_union(rooms_poly_segement.buffer(0.5,join_style=2))
    if rooms_polygon.geom_type == 'MultiPolygon': rooms_polygon = union_multipoly(rooms_polygon,0.01)

    if rooms_polygon.geom_type != 'GeometryCollection' and list(rooms_polygon.interiors) != []: rooms_polygon = fillgap_poly(rooms_polygon,0.1)

    if front_door: front_door = front_door.centroid.buffer(0.75)

    return rooms_polygon, rooms_poly_segement, types, front_door


def parse_floor(index_list,geo_data,subtype,floor_types,elevation,height):

    layout_all = []
    layout_apart = []
    public = []
    front_door = []
    stairs = []
    windows = []
    balcony = []
    railings = []
    room_eleva = 100
    window_eleva = 100
    rail_eleva = 100
    room_height = 100
    window_height = 100
    rail_height = 100

    for i in index_list:
        room_type = subtype[i]
        room = geo_data[i][10:-2].split(",")
        room_corners = []
        for j in room:
            xy = j.strip().split(" ")
            if xy[1][-1] == ')':
                xy[1] = xy[1][:-1]
            if xy[0][0] == '(':
                xy[0] = xy[0][1:]
            room_corners.append((float(xy[0]),float(xy[1])))
        rooms_poly = Polygon(room_corners)
        layout_all.append(rooms_poly)
        if room_type not in ['TERRACE','BALCONY','LOGGIA']: 
            layout_apart.append(rooms_poly)
            if room_eleva == 100:
                room_eleva = elevation[i]
                room_height = height[i]
        # if room_type in room_types or room_type == 'VOID':
        #     type_list.append(room_type)
        #     loc_list.append([rooms_poly.centroid.x,rooms_poly.centroid.y])
        if room_type in floor_types: 
            public.append(rooms_poly)
        elif room_type == 'ENTRANCE_DOOR': front_door.append(rooms_poly)
        elif room_type == 'STAIRCASE': stairs.append(rooms_poly)
        elif room_type == 'WINDOW': 
            windows.append(rooms_poly)
            if window_eleva == 100:
                window_eleva = elevation[i]
                window_height = height[i]
        elif room_type in ['TERRACE','BALCONY','LOGGIA']: 
            balcony.append(rooms_poly)
        elif room_type == 'RAILING': 
            railings.append(rooms_poly)
            if rail_eleva == 100:            
                rail_eleva = elevation[i]
                rail_height = height[i]

    floorplan_all_polygon = MultiPolygon(layout_apart)
    all_polygon = MultiPolygon(layout_all) 
    public_polygon = MultiPolygon(public)
    stair_polygon = MultiPolygon(stairs)
    window_polygon = MultiPolygon(windows)
    rail_polygon = MultiPolygon(railings)

    floorplan_all_polygon = union_multipoly(floorplan_all_polygon,0.01)
    all_polygon = union_multipoly(all_polygon,0.01)
    boundary_domainxy = floorplan_all_polygon.bounds # (minx, miny, maxx, maxy)
    x = boundary_domainxy[2] - boundary_domainxy[0]
    y = boundary_domainxy[3] - boundary_domainxy[1]
    public_polygon = unary_union(public_polygon.buffer(0.3,join_style=2))
    # private_polygon = unary_union(private_polygon.buffer(0.3,join_style=2))

    if list(floorplan_all_polygon.interiors) != []: floorplan_all_polygon = fillgap_poly(floorplan_all_polygon,0.1)
    if public_polygon.geom_type == 'MultiPolygon':
        modified = []
        for m,n in enumerate(public_polygon.geoms):
            if list(n.interiors) != []: modified.append(fillgap_poly(n,0.1))
            else: modified.append(n)
        public_polygon = MultiPolygon(modified)
    elif public_polygon.geom_type == 'GeometryCollection':
        public_polygon = Point((0,0)).buffer(0.001)
    else:
        if list(public_polygon.interiors) != []: public_polygon = fillgap_poly(public_polygon,0.1)

    public_polygon = public_polygon.intersection(floorplan_all_polygon)

    if len(front_door) == 1: newfrondoor = front_door[0].centroid.buffer(0.75)
    else: newfrondoor = [i.centroid.buffer(0.75) for i in front_door]

    return all_polygon, floorplan_all_polygon, public_polygon, stair_polygon, window_polygon, rail_polygon, balcony, \
            newfrondoor, x, y, room_eleva, window_eleva, rail_eleva, room_height, window_height, rail_height

        
def union_multipoly(geo,step):
    geo = unary_union(geo.buffer(step,join_style=2))
    while geo.geom_type == 'MultiPolygon':
        geo = geo.buffer(step,join_style=2)
    
    return geo


def fillgap_poly(geo,gap):
    for inte in list(geo.interiors):
        geo = unary_union(MultiPolygon([geo,Polygon(inte).buffer(gap,join_style=2)]))
    
    return geo


def scale_to_img(ratio,Mx,My,geo):

    return np.array([[[x*ratio + Mx, y*ratio + My] for x, y in zip(*geo.exterior.coords.xy)]])


def scale_to_img_list(ratio,Mx,My,geo):
    geo_list = []
    for i in geo:
        if i.geom_type != 'GeometryCollection':
            geo_list.append(np.array([[[x*ratio + Mx, y*ratio + My] for x, y in zip(*i.exterior.coords.xy)]]))
    
    return geo_list


def scale_pt_img(ratio,Mx,My,geo):
    x = geo[0]
    y = geo[1]
    return np.array([[x*ratio + Mx, y*ratio + My]])


def find_maxgeo(geo):
    area = []
    for i in geo.geoms:
        area.append(i.area)
    
    return geo.geoms[area.index(max(area))]


def find_domain_vec(boundary_corners):

    boundary_pt = np.array(boundary_corners)[:-1]
    len = boundary_pt.shape[0]
    vec_list = []
    len_list =[]
    for i in range(len-1):

        if boundary_pt[i+1][0] >= boundary_pt[i][0]:
            vec = np.array(boundary_pt[i+1]-boundary_pt[i])
        else:
            vec = np.array(boundary_pt[i]-boundary_pt[i+1])

        vec_len = np.linalg.norm(vec)

        if i > 0:
            check = 0
            for j,k in enumerate(vec_list):
                if np.arccos(vec.dot(k)) < 0.1:
                    len_list[j] += vec_len
                    check = 1
                    break
            if check == 0:
                vec_list.append(vec/vec_len)
                len_list.append(vec_len)

        else:
            vec_list.append(vec/vec_len)
            len_list.append(vec_len)

    domain_vec = vec_list[len_list.index(max(len_list))]

    return domain_vec


def find_longest_crv(geo):
    pts = geo.exterior.coords
    count = len(pts)
    length = []
    for i in range(count-1):
        length.append((pts[i+1,0]-pts[i,0])**2+(pts[i+1,1]-pts[i,1])**2)
    maxid = length.index(max(length))
    return LineString([pts[maxid],pts[maxid+1]])


def bound_type_analy(index_list,geo_data,subtype,house_room_types,house_type_index):

    layout_all = []
    type_list = []
    house_type_list = np.zeros((14))
    area = []
    convex = []
    id = []

    for s,i in enumerate(index_list):
        room = geo_data[i][10:-2].split(",")
        room_type = subtype[i]
        room_corners = []
        for j in room:
            xy = j.strip().split(" ")
            if xy[1][-1] == ')':
                xy[1] = xy[1][:-1]
            if xy[0][0] == '(':
                xy[0] = xy[0][1:]
            room_corners.append((float(xy[0]),float(xy[1])))
        rooms_poly = Polygon(room_corners)
        # layout_all.append(rooms_poly)
        if room_type in house_room_types and rooms_poly.area > 2:
            #house_type_list.append(house_type_index[house_room_types.index(room_type)])
            # house_type_list[house_room_types.index(room_type)] += 1
            # count = int(len(room_corners)-1)
            # if count > 30: count = 30
            # layout_house[count] += 1
            boundbox = rooms_poly.minimum_rotated_rectangle
            if explain_validity(boundbox) == 'Valid Geometry' and explain_validity(rooms_poly) == 'Valid Geometry':
                type_list.append(house_type_index[house_room_types.index(room_type)])
                area.append(rooms_poly.area)
                cut = boundbox.difference(rooms_poly)
                if explain_validity(cut) != 'Valid Geometry':
                    cut = make_valid(cut).geoms[0]
                ratio = cut.area/rooms_poly.area
                convex.append(ratio)#count+
                id.append(s)

    return layout_all,type_list,house_type_list,area,convex,id


def get_adjacency_graph(extended_house):

    ada = []
    if extended_house.geom_type == 'MultiPolygon':
        geo = extended_house.geoms
        num = len(geo)
        for i in range(num):
            if i == num - 1:
                break
            else:
                for m in range(i+1,num):
                    if geo[i].intersects(geo[m]):
                        ada.append([i,m])
    
    return np.array(ada)


def floorplan_to_Json(filename,polygons,names_List,locations_List,types_List,Graph_List,area,Bound,Door,Windows,RDoors):

    rooms = polygons.geoms

    data = {
        "Edges": Graph_List.tolist(),
        "boundary_Corners": Bound,
        "Front_Door": Door,
        "Windows": Windows,
        "Room_Door": RDoors,
        "nodes": [
            {
                "name": names_List[indx],
                "id": type,
                "polygon": list(rooms[indx].exterior.coords),
                "area": area[indx],
                "location": locations_List[indx]
            }
            for indx,type in enumerate(types_List) if type != 0
        ]
    }
    with open(filename, "w") as f:
        json.dump(data, f)


def np_rotate(vec,degree):

    alpha = degree*np.pi
    x_ = vec[1]*np.sin(alpha)+vec[0]*np.cos(alpha)
    y_ = vec[1]*np.cos(alpha)-vec[0]*np.sin(alpha)
    new = np.array([x_,y_])

    return new


def get_boundbox(center,vec,size):

    pt1 = (center.x + size*(vec[1]-vec[0]),center.y - size*(vec[1]+vec[0]))
    pt2 = (center.x + size*(vec[1]+vec[0]),center.y + size*(vec[1]-vec[0]))
    pt3 = (center.x + size*(vec[0]-vec[1]),center.y + size*(vec[1]+vec[0]))
    pt4 = (center.x - size*(vec[1]+vec[0]),center.y - size*(vec[1]-vec[0]))

    return [pt1,pt2,pt3,pt4,pt1]


def ada_sparse(adacency):

    graph = np.zeros((14,14))
    for i,j in enumerate(adacency):
        for f in range(14):
            graph[i][f][f] = 1
        for m,n in enumerate(j):
            if (n[0] != 0 or n[1] != 0):
                graph[i][int(n[0])][int(n[1])] = 1
                graph[i][int(n[1])][int(n[0])] = 1
    
    return graph


def make_bit(value,dimension):

    return np.array(list(np.binary_repr(value).zfill(dimension))).astype(np.int32)


def randint_size(n, N, replace=True): 

    return np.random.choice(N, n, replace=False)


def randint_pick_attribute(valid): 

    if valid == 1 or valid == 2:

        P_T = np.array([0])
        P_LAS = 0
        P_R = np.array([0])
        P_W = np.array([0])

    elif valid == 3:

        P_T = np.array([1,2])
        P_LAS = 1
        P_R = np.array([1,2])
        P_W = np.array([1,2])

    else:

        # 1.sub-graph: partial L, A, and sub-sub T, S
        n_g = np.random.randint(1,int(valid/2+1))
        picked_g = np.random.choice(valid-1, n_g, replace=False) + 1

        # 2.partial region: partial R
        n_R = np.random.randint(1,int(valid/2))
        picked_R = np.random.choice(valid-1, n_R, replace=False) + 1

        # 3.partial window: partial W
        n_W = np.random.randint(1,int(valid/2+1))
        picked_W = np.random.choice(valid-1, n_W, replace=False) + 1

        P_LAS = picked_g[0]
        P_T = np.sort(picked_g)
        P_R = np.sort(picked_R)
        P_W = np.sort(picked_W)

    return P_T, P_LAS, P_R, P_W

