import numpy as np
import math

def build_snowflake(prefratal_level, h_max=math.inf, scale = 1, apex_angle = math.pi/3, shift = np.array([0,0])):

    #get starting level one fractal
    (nodes_data, edges_nodes, tri_nodes, tri_edges, 
    tri_types) = get_koch_seed(scale = 1, apex_angle = math.pi/3, shift = np.array([0,0]))

    init_h = get_h_max(tri_nodes,nodes_data)

    final_max_h = init_h/(2**np.floor(prefratal_level/2)) # will halve every other j
    # final_min_h = init_h/(3**prefratal_level)
    extra_refinements = np.ceil(final_max_h/h_max)


    # first refine to desired prefractal level
    for j in range(prefratal_level):
        if np.mod(j,2) == 0:
            j_even = True # only refine along dashed lines and type 8s when j is even
        else:
            if extra_refinements>0:
                extra_refinements-=1
                j_even = True
            else:
                j_even = False

        (tri_edges, tri_nodes, tri_types, edges_nodes, 
        nodes_data) =  subdiv_all(tri_edges, tri_nodes, tri_types, edges_nodes, nodes_data,
            apex_angle=apex_angle, increase_prefractal_level=True, dashed=j_even)

    # now refine to desired meshwidth, if not already achieved
    while get_h_max(tri_nodes,nodes_data) > h_max:
        (tri_edges, tri_nodes, tri_types, edges_nodes, 
        nodes_data) = subdiv_all(tri_edges, tri_nodes, tri_types, edges_nodes, nodes_data,
            apex_angle=apex_angle, increase_prefractal_level=False)

    return nodes_data, tri_nodes



def get_koch_seed(scale = 1, apex_angle = math.pi/3, shift = np.array([0,0])):
    h = math.sqrt(3/4)
    b = math.sqrt(1-h**2)
    nodes_data = shift + (scale/(4*h))*np.array([[0,0],[-1/2,h],[1/2,h],[0,2*h],
                            [-3*b,h],[-1,0],[-3*b,-h],[-b,-h],
                            [0,-2*h],[b,-h],[3*b,-h],[1,0],[3*b,h]],dtype=np.float64)

    if apex_angle != math.pi/3:
        mid_rel_dist = math.sin(apex_angle/2)/(2*(1+math.sin(apex_angle/2)))
        edge_mid = [(nodes_data[1]+nodes_data[5])/2,
                    (nodes_data[7]+nodes_data[9])/2,
                    (nodes_data[2]+nodes_data[11])/2]
        edge_dir = [(nodes_data[5]-nodes_data[1]),
                    (nodes_data[9]-nodes_data[7]),
                    (nodes_data[2]-nodes_data[11])]

        # Bring these nodes in a little if apex angle is different
        nodes_data[1] = edge_mid[0] - mid_rel_dist*edge_dir[0]
        nodes_data[5] = edge_mid[0] + mid_rel_dist*edge_dir[0]
        nodes_data[7] = edge_mid[1] - mid_rel_dist*edge_dir[1]
        nodes_data[9] = edge_mid[1] + mid_rel_dist*edge_dir[1]
        nodes_data[11] = edge_mid[2] - mid_rel_dist*edge_dir[2]
        nodes_data[2] = edge_mid[2] + mid_rel_dist*edge_dir[2]

    edges_nodes = [[0,1],[1,2],[2,0],[1,3],[3,2],
                    [5,1],[0,5],[4,1],[5,4],
                    [7,5],[6,5],[7,6],[0,7],[9,7],[8,7],[9,8],
                    [0,9],[11,9],[0,11],[10,9],[11,10],
                    [2,11],[12,11],[2,12]]
    tri_nodes = [[1,2,0],[1,3,2],[5,1,0],[5,4,1],[7,5,0],[7,6,5],
                [9,7,0],[9,8,7],[11,9,0],[11,10,9],
                [2,11,0],[2,12,11]]
    tri_edges = [[1,2,0],[3,4,1],[5,0,6],[8,7,5],[9,6,12],[11,10,9],
                [13,12,16],[15,14,13],[17,16,18],[20,19,17],[21,18,2],[23,22,21]]
    tri_types = [6,4,6,4,6,4,6,4,6,4,6,4]
    return nodes_data, edges_nodes, tri_nodes, tri_edges, tri_types

def get_h_max(tri_nodes,nodes_data):
    h_max = 0.0
    for n in range(len(tri_nodes)):
        for m in range(2):
            x = np.linalg.norm(nodes_data[tri_nodes[n][m]]-nodes_data[tri_nodes[n][m+1]])
            if x>h_max:
                h_max = x
        y = np.linalg.norm(nodes_data[tri_nodes[n][2]]-nodes_data[tri_nodes[n][0]])
        if x>h_max:
            h_max = y
    return h_max



def subdiv_all(tri_edges, tri_nodes, tri_types, edges_nodes, nodes_data,
                apex_angle=math.pi/3, increase_prefractal_level=True, dashed=True):

    subdivided_edge_indices_old = [] # 1d integer array
    subdivided_edge_nodes = [] # 2d integer array
    subdivided_edge_indices_new = [] #2d integer array
    new_tri_edges = [] #2d integer array
    new_tri_nodes = [] #2d integer array
    new_tri_types = [] # 1d integer array
    num_tris = len(tri_types)

    for n in range(num_tris):
        if tri_types[n] == 8:
            if dashed:
                (tri_edges_out_n, tri_nodes_out_n, tri_types_out_n, edges_nodes, 
                nodes_data, subdivided_edge_indices_old, subdivided_edge_nodes, 
                subdivided_edge_indices_new)  = subdiv_8(
                                                tri_edges[n], tri_nodes[n], edges_nodes, nodes_data, 
                                                subdivided_edge_indices_old, subdivided_edge_nodes,
                                                subdivided_edge_indices_new)
            else: # just replace new data with old
                tri_edges_out_n = [tri_edges[n]]
                tri_nodes_out_n = [tri_nodes[n]]
                tri_types_out_n = [8]
                            
        elif tri_types[n] == 4:
            (tri_edges_out_n, tri_nodes_out_n, tri_types_out_n, edges_nodes, 
            nodes_data, subdivided_edge_indices_old, subdivided_edge_nodes, 
            subdivided_edge_indices_new)  = subdiv_4(
                                            tri_edges[n], tri_nodes[n], edges_nodes, nodes_data, 
                                            subdivided_edge_indices_old, subdivided_edge_nodes,
                                            subdivided_edge_indices_new, apex_angle,
                                            increase_prefractal_level=increase_prefractal_level)
        elif tri_types[n] == 5:
            (tri_edges_out_n, tri_nodes_out_n, tri_types_out_n, edges_nodes, 
            nodes_data, subdivided_edge_indices_old, subdivided_edge_nodes, 
            subdivided_edge_indices_new)  =   subdiv_56(5, tri_edges[n], tri_nodes[n], edges_nodes, nodes_data, 
                                    subdivided_edge_indices_old, subdivided_edge_nodes, 
                                    subdivided_edge_indices_new, apex_angle,
                                    increase_prefractal_level=increase_prefractal_level)
        elif tri_types[n] == 6:
            (tri_edges_out_n, tri_nodes_out_n, tri_types_out_n, edges_nodes, 
            nodes_data, subdivided_edge_indices_old, subdivided_edge_nodes, 
            subdivided_edge_indices_new)  =   subdiv_56(6, tri_edges[n], tri_nodes[n], edges_nodes, nodes_data, 
                                    subdivided_edge_indices_old, subdivided_edge_nodes, 
                                    subdivided_edge_indices_new, apex_angle)
        elif tri_types[n] == 7:
            (tri_edges_out_n, tri_nodes_out_n, tri_types_out_n, edges_nodes, 
            nodes_data, subdivided_edge_indices_old, subdivided_edge_nodes, 
            subdivided_edge_indices_new)  =   subdiv_7(tri_edges[n], tri_nodes[n], edges_nodes, nodes_data, 
                                    subdivided_edge_indices_old, subdivided_edge_nodes, 
                                    subdivided_edge_indices_new, dashed)

        # concatenate these vectors of indices:
        new_tri_edges = new_tri_edges + tri_edges_out_n
        new_tri_nodes = new_tri_nodes + tri_nodes_out_n
        new_tri_types = new_tri_types + tri_types_out_n

    return (new_tri_edges, new_tri_nodes, new_tri_types, edges_nodes, nodes_data)

""" ----------------------------------------------------------------
Utility functions which are called multiple times by the subdivision routines
----------------------------------------------------------------"""

def double_split_angles(new_node, n, n_count, nodes_data,edges_nodes,tri_nodes_in,
                        new_node_index,new_edge_index, temp_edge_inds,temp_node_inds,
                        subdivided_edge_indices_old,subdivided_edge_nodes,subdivided_edge_indices_new):

    temp_node_inds.append(tri_nodes_in[n_count])

    if n not in subdivided_edge_indices_old:
        #create two new nodes along edge:
        nodes_data = np.append(nodes_data,[new_node],axis=0)
        # the above is a little hack as numpy doesn't have the same append options, in place of
        # nodes_data.append((nodes_data[edges_nodes[n][0]]+nodes_data[edges_nodes[n][1]])/2)
        temp_node_inds.append(new_node_index)

        # below needs a conditonal statement, incase edge goes against grain
        if edges_nodes[n][0] == tri_nodes_in[n_count]:
            # define two new edges, using two original nodes and one new node:
            edges_nodes.append([edges_nodes[n][0], new_node_index])
            edges_nodes.append([new_node_index, edges_nodes[n][1]])
        elif edges_nodes[n][1] == tri_nodes_in[n_count]:
            # define two new edges, using two original nodes and one new node:
            edges_nodes.append([edges_nodes[n][1], new_node_index])
            edges_nodes.append([new_node_index, edges_nodes[n][0]])
        else:
            raise Exception("Neither node of edge matches node of triangle")
        temp_edge_inds.append(new_edge_index)
        temp_edge_inds.append(new_edge_index+1)

        # make a note that this edge is no longer in use:
        subdivided_edge_indices_old.append(n)
        # and has been replaced by these two edges:
        subdivided_edge_nodes.append([edges_nodes[n][0], new_node_index, edges_nodes[n][1]])
        subdivided_edge_indices_new.append([new_edge_index,new_edge_index+1])

        new_edge_index += 2
        new_node_index += 1
    else:
        n_ = subdivided_edge_indices_old.index(n)
        temp_node_inds.append(subdivided_edge_nodes[n_][1]) # choose middle of three nodes on new edge

        # if node is in first of two subdivided edges, choose that one
        if tri_nodes_in[n_count] in edges_nodes[subdivided_edge_indices_new[n_][0]]:
            temp_edge_inds.append(subdivided_edge_indices_new[n_][0])
            temp_edge_inds.append(subdivided_edge_indices_new[n_][1])
        elif tri_nodes_in[n_count] in edges_nodes[subdivided_edge_indices_new[n_][1]]:
            temp_edge_inds.append(subdivided_edge_indices_new[n_][1])
            temp_edge_inds.append(subdivided_edge_indices_new[n_][0])
        else:
            raise Exception("Neither node of previously subdivided edge matches node of triangle")

    n_count+=1
            
    return (n, n_count, nodes_data,edges_nodes,tri_nodes_in,
                        new_node_index,new_edge_index, temp_edge_inds,temp_node_inds,
                        subdivided_edge_indices_old,subdivided_edge_nodes,subdivided_edge_indices_new)


def edge_triple_split(n,n_count,new_edge_index,new_node_index,mid_rel_dist,
                    edges_nodes,tri_nodes_in,nodes_data,
                    temp_edge_inds,temp_node_inds, subdivided_edge_indices_old,
                    subdivided_edge_nodes,subdivided_edge_indices_new):

    temp_node_inds.append(tri_nodes_in[n_count])

    if n not in subdivided_edge_indices_old:
        #create two new nodes along edge:
        edge_mid = (nodes_data[edges_nodes[n][0]]+nodes_data[edges_nodes[n][1]])/2
        edge_dir = nodes_data[edges_nodes[n][1]] - nodes_data[edges_nodes[n][0]]

        nodes_data = np.append(nodes_data,[edge_mid - mid_rel_dist*edge_dir],axis=0)
        nodes_data = np.append(nodes_data,[edge_mid + mid_rel_dist*edge_dir],axis=0)
        temp_node_inds.append(new_node_index)
        temp_node_inds.append(new_node_index+1)

        # I think the two cases below are all that will occur in practice.
        # If two edges on the same side have nodes indexed in the opposite direction,
        # that's not covered by either case, but I don't think that will happen.
        if edges_nodes[n][0] == tri_nodes_in[n_count]:
            # define three new edges, using two original nodes and two new nodes:
            edges_nodes.append([edges_nodes[n][0], new_node_index])
            edges_nodes.append([new_node_index, new_node_index+1])
            edges_nodes.append([new_node_index+1, edges_nodes[n][1]])
        elif edges_nodes[n][1] == tri_nodes_in[n_count]:
            # define three new edges, using two original nodes and two new nodes:
            edges_nodes.append([edges_nodes[n][1], new_node_index])
            edges_nodes.append([new_node_index, new_node_index+1])
            edges_nodes.append([new_node_index+1, edges_nodes[n][0]])
        else:
            raise Exception("Neither node of edge matches node of triangle")
        
        temp_edge_inds.append(new_edge_index)
        temp_edge_inds.append(new_edge_index+1)
        temp_edge_inds.append(new_edge_index+2)

        
        # make a note that this edge is no longer in use:
        subdivided_edge_indices_old.append(n)
        # and has been replaced by these three edges:
        subdivided_edge_nodes.append([edges_nodes[n][0], new_node_index, new_node_index+1, edges_nodes[n][1]])
        subdivided_edge_indices_new.append([new_edge_index,new_edge_index+1,new_edge_index+2])

        new_edge_index += 3
        new_node_index += 2
    else:
        n_ = subdivided_edge_indices_old.index(n)
        # choose middle two of four nodes on new edges

        # if node is in first of two subdivided edges, choose that one
        if tri_nodes_in[n_count] in edges_nodes[subdivided_edge_indices_new[n_][0]]:
            temp_node_inds.append(subdivided_edge_nodes[n_][1])
            temp_node_inds.append(subdivided_edge_nodes[n_][2])
            temp_edge_inds.append(subdivided_edge_indices_new[n_][0])
            temp_edge_inds.append(subdivided_edge_indices_new[n_][1])
            temp_edge_inds.append(subdivided_edge_indices_new[n_][2])
        elif tri_nodes_in[n_count] in edges_nodes[subdivided_edge_indices_new[n_][2]]:
            temp_node_inds.append(subdivided_edge_nodes[n_][2])
            temp_node_inds.append(subdivided_edge_nodes[n_][1])
            temp_edge_inds.append(subdivided_edge_indices_new[n_][2])
            temp_edge_inds.append(subdivided_edge_indices_new[n_][1])
            temp_edge_inds.append(subdivided_edge_indices_new[n_][0])
        else:
            raise Exception("Neither node of previously subdivided edge matches node of triangle")

    n_count+=1

    return n,n_count,new_edge_index,new_node_index,mid_rel_dist,edges_nodes,tri_nodes_in,nodes_data,\
                    temp_edge_inds,temp_node_inds, subdivided_edge_indices_old,\
                    subdivided_edge_nodes,subdivided_edge_indices_new



def get_new_triangle_node(x_1,x_2,apex_angle,centre_node):
    # reflect v in the line x_2-x_1, and stretch the reflection
    x_mid = (x_2+x_1)/2
    x_dir = x_2-x_1
    perp = np.array([x_dir[1],-x_dir[0]]) # get a perpendicular vector
    perp = perp/np.linalg.norm(perp) # normalise it
    if np.dot(perp,centre_node-x_mid)>0: # check it points the right direction (of a possible two)
        perp = -perp

    # length_of_new_edge = base_edge_length/2/(math.sin(apex_angle/2))
    dist_from_x_mid = np.linalg.norm(x_2-x_1)/2/math.tan(apex_angle/2)
    return x_mid + dist_from_x_mid*perp


def get_centre_node_cases56(x_1,x_2,centre_node):
    x_mid = (x_2+x_1)/2
    x_dir = x_2-x_1
    perp = np.array([x_dir[1],-x_dir[0]]) # get a perpendicular vector
    perp = perp/np.linalg.norm(perp) # normalise it
    if np.dot(perp,centre_node-x_mid)<0: # check it points the right direction (of a possible two)
        perp = -perp
    side_length = np.linalg.norm(x_dir)
    tri_height = math.sqrt(side_length**2-(side_length/2)**2)
    return x_mid + perp*tri_height/3


""" ----------------------------------------------------------------
The main subdivision routines
----------------------------------------------------------------"""

# 
def subdiv_4(tri_edges_in, tri_nodes_in, edges_nodes, nodes_data, 
            subdivided_edge_indices_old, subdivided_edge_nodes, 
            subdivided_edge_indices_new, apex_angle,
            increase_prefractal_level=True):
    new_node_index = len(nodes_data)
    new_edge_index = len(edges_nodes)
    temp_edge_inds = [] # this maps edges from the current 0:9 edges to the global (not in a coding sense) edge indices
    temp_node_inds = [] # this maps nodes from the current 0:6 nodes to the global (not in a coding sense) node indices

    n_count=0
    mid_rel_dist_new_nodes = math.sin(apex_angle/2)/(2*(1+math.sin(apex_angle/2)))
    
    for n in tri_edges_in:

        # temp_node_inds.append(tri_nodes_in[n_count])

        if n_count <2:
            mid_rel_dist = mid_rel_dist_new_nodes
        else:
            mid_rel_dist = 1/6

        # if n not in subdivided_edge_indices_old:
        (n,n_count,new_edge_index,new_node_index,mid_rel_dist,
                edges_nodes,tri_nodes_in,nodes_data,
                temp_edge_inds,temp_node_inds, subdivided_edge_indices_old,
                subdivided_edge_nodes,subdivided_edge_indices_new) = edge_triple_split(n,n_count,new_edge_index,new_node_index,mid_rel_dist,
                edges_nodes,tri_nodes_in,nodes_data,
                temp_edge_inds,temp_node_inds, subdivided_edge_indices_old,
                subdivided_edge_nodes,subdivided_edge_indices_new)


    # create new node (local index 9) in the middle
    n_9 = (nodes_data[tri_nodes_in[0]]+nodes_data[tri_nodes_in[1]] + nodes_data[tri_nodes_in[2]])/3
    nodes_data = np.append(nodes_data,[n_9],axis=0)
    temp_node_inds.append(new_node_index)
    new_node_index +=1

    # now create nine new edges in the interior, which will not be duplicates
    # creating these from the other edges is bad! and won't work in general
    edges_nodes.append([temp_node_inds[8], temp_node_inds[1]])
    temp_edge_inds.append(new_edge_index)
    new_edge_index += 1
    edges_nodes.append([temp_node_inds[1], temp_node_inds[9]])
    temp_edge_inds.append(new_edge_index)
    new_edge_index += 1
    edges_nodes.append([temp_node_inds[9], temp_node_inds[2]])
    temp_edge_inds.append(new_edge_index)
    new_edge_index += 1
    edges_nodes.append([temp_node_inds[2], temp_node_inds[4]])
    temp_edge_inds.append(new_edge_index)
    new_edge_index += 1
    edges_nodes.append([temp_node_inds[4], temp_node_inds[9]])
    temp_edge_inds.append(new_edge_index)
    new_edge_index += 1
    edges_nodes.append([temp_node_inds[9], temp_node_inds[5]])
    temp_edge_inds.append(new_edge_index)
    new_edge_index += 1
    edges_nodes.append([temp_node_inds[5], temp_node_inds[7]])
    temp_edge_inds.append(new_edge_index)
    new_edge_index += 1
    edges_nodes.append([temp_node_inds[7], temp_node_inds[9]])
    temp_edge_inds.append(new_edge_index)
    new_edge_index += 1
    edges_nodes.append([temp_node_inds[9], temp_node_inds[8]])
    temp_edge_inds.append(new_edge_index)
    new_edge_index += 1

    if increase_prefractal_level:
        # now create two new triangles on the exterior of the original triangle, to continue the fractal pattern.
        # the first:
        n_10 = get_new_triangle_node(nodes_data[temp_node_inds[1]],nodes_data[temp_node_inds[2]],apex_angle,n_9)
        nodes_data = np.append(nodes_data,[n_10],axis=0)
        temp_node_inds.append(new_node_index)
        new_node_index +=1
        edges_nodes.append([temp_node_inds[1], temp_node_inds[10]])
        temp_edge_inds.append(new_edge_index)
        new_edge_index += 1
        edges_nodes.append([temp_node_inds[10], temp_node_inds[2]])
        temp_edge_inds.append(new_edge_index)
        new_edge_index += 1

        # the second:    
        n_11 = get_new_triangle_node(nodes_data[temp_node_inds[4]],nodes_data[temp_node_inds[5]],apex_angle,n_9)
        nodes_data = np.append(nodes_data,[n_11],axis=0)
        temp_node_inds.append(new_node_index)
        new_node_index +=1
        edges_nodes.append([temp_node_inds[4], temp_node_inds[11]])
        temp_edge_inds.append(new_edge_index)
        new_edge_index += 1
        edges_nodes.append([temp_node_inds[11], temp_node_inds[5]])
        temp_edge_inds.append(new_edge_index)
        new_edge_index += 1

    # now group all edges together into triangles.
    # indices marked with # may need re-ordering later
    tri_edges_out = [[temp_edge_inds[0],temp_edge_inds[9],temp_edge_inds[8]],#
                        [temp_edge_inds[9],temp_edge_inds[10],temp_edge_inds[17]],
                        [temp_edge_inds[1],temp_edge_inds[11],temp_edge_inds[10]],
                        [temp_edge_inds[12],temp_edge_inds[13],temp_edge_inds[11]],
                        [temp_edge_inds[2],temp_edge_inds[3],temp_edge_inds[12]],
                        [temp_edge_inds[4],temp_edge_inds[14],temp_edge_inds[13]],
                        [temp_edge_inds[14],temp_edge_inds[15],temp_edge_inds[16]],
                        [temp_edge_inds[5],temp_edge_inds[6],temp_edge_inds[15]],
                        [temp_edge_inds[17],temp_edge_inds[16],temp_edge_inds[7]]]
    if increase_prefractal_level:
        # now the two new triangles on the outside
        tri_edges_out.append([temp_edge_inds[18],temp_edge_inds[19],temp_edge_inds[1]])
        tri_edges_out.append([temp_edge_inds[20],temp_edge_inds[21],temp_edge_inds[4]])

    if increase_prefractal_level:
        tri_types_out = [5,7,6,6,4,6,7,5,8,4,4]
    else:
        tri_types_out = [7,8,8,8,7,8,8,7,8]

    # indices marked with # may need re-ordering later
    tri_nodes_out = [[temp_node_inds[0], temp_node_inds[1], temp_node_inds[8]],#
                    [temp_node_inds[8], temp_node_inds[1], temp_node_inds[9]],
                    [temp_node_inds[1], temp_node_inds[2], temp_node_inds[9]],
                    [temp_node_inds[2], temp_node_inds[4], temp_node_inds[9]],
                    [temp_node_inds[2], temp_node_inds[3], temp_node_inds[4]],
                    [temp_node_inds[4], temp_node_inds[5], temp_node_inds[9]],
                    [temp_node_inds[9], temp_node_inds[5], temp_node_inds[7]],
                    [temp_node_inds[5], temp_node_inds[6], temp_node_inds[7]],
                    [temp_node_inds[8], temp_node_inds[9], temp_node_inds[7]]]
    if increase_prefractal_level:
        tri_nodes_out.append([temp_node_inds[1], temp_node_inds[10], temp_node_inds[2]])
        tri_nodes_out.append([temp_node_inds[4], temp_node_inds[11], temp_node_inds[5]])
    else:
        tri_edges_out[0] = [temp_edge_inds[8],temp_edge_inds[0],temp_edge_inds[9]]
        tri_nodes_out[0] = [temp_node_inds[8], temp_node_inds[0], temp_node_inds[1]]

    return tri_edges_out, tri_nodes_out, tri_types_out, edges_nodes, nodes_data, \
            subdivided_edge_indices_old, subdivided_edge_nodes, subdivided_edge_indices_new

# 
def subdiv_56(either_5_or_6, tri_edges_in, tri_nodes_in, edges_nodes, nodes_data, 
            subdivided_edge_indices_old, subdivided_edge_nodes, 
            subdivided_edge_indices_new, apex_angle,
            increase_prefractal_level=True):
    new_node_index = len(nodes_data)
    new_edge_index = len(edges_nodes)
    temp_edge_inds = [] # this maps edges from the current 0:9 edges to the global (not in a coding sense) edge indices
    temp_node_inds = [] # this maps nodes from the current 0:6 nodes to the global (not in a coding sense) node indices

    # deal with top edge first, which is external, so won't have been subdivided yet.
    if either_5_or_6 == 5:
            mid_rel_dist = math.sin(apex_angle/2)/(2*(1+math.sin(apex_angle/2)))
    else:
        mid_rel_dist = 1/6

    n = tri_edges_in[0]

    (n,n_count,new_edge_index,new_node_index,mid_rel_dist,
                    edges_nodes,tri_nodes_in,nodes_data,
                    temp_edge_inds,temp_node_inds, subdivided_edge_indices_old,
                    subdivided_edge_nodes,subdivided_edge_indices_new) = edge_triple_split(n,0,new_edge_index,new_node_index,mid_rel_dist,
                    edges_nodes,tri_nodes_in,nodes_data,
                    temp_edge_inds,temp_node_inds, subdivided_edge_indices_old,
                    subdivided_edge_nodes,subdivided_edge_indices_new)
    
    #create one new node along second edge:
    n = tri_edges_in[1]

    #create new node
    new_node = (2*nodes_data[tri_nodes_in[1]]+nodes_data[tri_nodes_in[2]])/3
    #can't tell if the above might choose the wrong skewing - could change to input tri nodes if it does?
    # now do all the usual stuff with it
    (n, n_count, nodes_data,edges_nodes,tri_nodes_in,
        new_node_index,new_edge_index, temp_edge_inds,temp_node_inds,
        subdivided_edge_indices_old,subdivided_edge_nodes,subdivided_edge_indices_new) \
        = double_split_angles(new_node, n, n_count, nodes_data,edges_nodes,tri_nodes_in,
        new_node_index,new_edge_index, temp_edge_inds,temp_node_inds,
        subdivided_edge_indices_old,subdivided_edge_nodes,subdivided_edge_indices_new)

    
    #create one new node along third edge:
    n = tri_edges_in[2]

    #create new node
    new_node = (2*nodes_data[tri_nodes_in[0]]+nodes_data[tri_nodes_in[2]])/3
    #can't tell if the above might choose the wrong skewing - could change to input tri nodes if it does?
    # now do all the usual stuff with it
    (n, n_count, nodes_data,edges_nodes,tri_nodes_in,
        new_node_index,new_edge_index, temp_edge_inds,temp_node_inds,
        subdivided_edge_indices_old,subdivided_edge_nodes,subdivided_edge_indices_new) \
        = double_split_angles(new_node, n, n_count, nodes_data,edges_nodes,tri_nodes_in,
        new_node_index,new_edge_index, temp_edge_inds,temp_node_inds,
        subdivided_edge_indices_old,subdivided_edge_nodes,subdivided_edge_indices_new)

    
    bary = (nodes_data[tri_nodes_in[0]]+nodes_data[tri_nodes_in[1]] + nodes_data[tri_nodes_in[2]])/3
    n_7 = get_centre_node_cases56(nodes_data[tri_nodes_in[0]],nodes_data[tri_nodes_in[1]],bary)
    nodes_data = np.append(nodes_data,[n_7],axis=0)
    temp_node_inds.append(new_node_index)
    new_node_index +=1

    #now create seven edges in the interior:
    edges_nodes.append([temp_node_inds[6], temp_node_inds[1]])
    temp_edge_inds.append(new_edge_index)
    new_edge_index += 1
    edges_nodes.append([temp_node_inds[1], temp_node_inds[7]])
    temp_edge_inds.append(new_edge_index)
    new_edge_index += 1
    edges_nodes.append([temp_node_inds[7], temp_node_inds[2]])
    temp_edge_inds.append(new_edge_index)
    new_edge_index += 1
    edges_nodes.append([temp_node_inds[2], temp_node_inds[4]])
    temp_edge_inds.append(new_edge_index)
    new_edge_index += 1
    edges_nodes.append([temp_node_inds[4], temp_node_inds[7]])
    temp_edge_inds.append(new_edge_index)
    new_edge_index += 1
    edges_nodes.append([temp_node_inds[5], temp_node_inds[7]])
    temp_edge_inds.append(new_edge_index)
    new_edge_index += 1
    edges_nodes.append([temp_node_inds[6], temp_node_inds[7]])
    temp_edge_inds.append(new_edge_index)
    new_edge_index += 1

    # by default, assume type 5
    tri_edges_out = [[temp_edge_inds[0],temp_edge_inds[7],temp_edge_inds[6]],#may need re-ordering later
                    [temp_edge_inds[7],temp_edge_inds[8],temp_edge_inds[13]],
                    [temp_edge_inds[1],temp_edge_inds[9],temp_edge_inds[8]],
                    [temp_edge_inds[9],temp_edge_inds[10],temp_edge_inds[11]],
                    [temp_edge_inds[2],temp_edge_inds[3],temp_edge_inds[10]],
                    [temp_edge_inds[11],temp_edge_inds[4],temp_edge_inds[12]],
                    [temp_edge_inds[13],temp_edge_inds[12],temp_edge_inds[5]]]

    tri_nodes_out = [[temp_node_inds[0], temp_node_inds[1], temp_node_inds[6]],#may need re-ordering later
                    [temp_node_inds[6], temp_node_inds[1], temp_node_inds[7]],
                    [temp_node_inds[1], temp_node_inds[2], temp_node_inds[7]],
                    [temp_node_inds[7], temp_node_inds[2], temp_node_inds[4]],
                    [temp_node_inds[2], temp_node_inds[3], temp_node_inds[4]],
                    [temp_node_inds[7], temp_node_inds[4], temp_node_inds[5]],
                    [temp_node_inds[6], temp_node_inds[7], temp_node_inds[5]]]

    if either_5_or_6 == 5 and increase_prefractal_level:
        n_8 = get_new_triangle_node(nodes_data[temp_node_inds[1]],nodes_data[temp_node_inds[2]],apex_angle,n_7)
        nodes_data = np.append(nodes_data,[n_8],axis=0)
        temp_node_inds.append(new_node_index)
        new_node_index +=1
        edges_nodes.append([temp_node_inds[1], temp_node_inds[8]])
        temp_edge_inds.append(new_edge_index)
        new_edge_index += 1
        edges_nodes.append([temp_node_inds[8], temp_node_inds[2]])
        temp_edge_inds.append(new_edge_index)
        new_edge_index += 1

        tri_nodes_out.append([temp_node_inds[1], temp_node_inds[8], temp_node_inds[2]])
        tri_edges_out.append([temp_edge_inds[14],temp_edge_inds[15],temp_edge_inds[1]])
        tri_types_out = [5,7,6,7,5,8,8,4]

    else:
        # adjust the ordering of these indices, so that it is consitent with triangle types
        tri_edges_out[0] = [temp_edge_inds[6],temp_edge_inds[0],temp_edge_inds[7]]
        tri_nodes_out[0] = [temp_node_inds[6],temp_node_inds[0],temp_node_inds[1]]
        tri_types_out = [7,8,8,8,7,8,8]

    return tri_edges_out, tri_nodes_out, tri_types_out, edges_nodes, nodes_data, \
        subdivided_edge_indices_old, subdivided_edge_nodes, subdivided_edge_indices_new

# 
def subdiv_7(tri_edges_in, tri_nodes_in, edges_nodes, nodes_data, 
            subdivided_edge_indices_old, subdivided_edge_nodes, 
            subdivided_edge_indices_new, dashed=True):
    new_node_index = len(nodes_data)
    new_edge_index = len(edges_nodes)
    temp_edge_inds = [] # this maps edges from the current edges to the global (not in a coding sense) edge indices
    temp_node_inds = [] # this maps nodes from the current nodes to the global (not in a coding sense) node indices

    n_count=0
    
    for n in tri_edges_in:
    
        if n_count in [0,1]:
            #create one new nodes along edge:
            if n_count == 0:
                new_node = (nodes_data[edges_nodes[n][0]]+2*nodes_data[edges_nodes[n][1]])/3
            elif n_count == 1:
                new_node = (2*nodes_data[edges_nodes[n][0]]+nodes_data[edges_nodes[n][1]])/3

            (n,n_count, nodes_data,edges_nodes,tri_nodes_in,
                new_node_index,new_edge_index, temp_edge_inds,temp_node_inds,
                subdivided_edge_indices_old,subdivided_edge_nodes,subdivided_edge_indices_new) \
                = double_split_angles(new_node, n, n_count, nodes_data,edges_nodes,tri_nodes_in,
                new_node_index,new_edge_index, temp_edge_inds,temp_node_inds,
                subdivided_edge_indices_old,subdivided_edge_nodes,subdivided_edge_indices_new)
        else:
            if dashed:
                new_node = (nodes_data[edges_nodes[n][0]]+nodes_data[edges_nodes[n][1]])/2
                (n,n_count, nodes_data,edges_nodes,tri_nodes_in,
                new_node_index,new_edge_index, temp_edge_inds,temp_node_inds,
                subdivided_edge_indices_old,subdivided_edge_nodes,subdivided_edge_indices_new) \
                = double_split_angles(new_node, n, n_count, nodes_data,edges_nodes,tri_nodes_in,
                new_node_index,new_edge_index, temp_edge_inds,temp_node_inds,
                subdivided_edge_indices_old,subdivided_edge_nodes,subdivided_edge_indices_new)
            else:
                # no issue of whether subdivion has/hasn't happened, as no new nodes are created 
                temp_node_inds.append(tri_nodes_in[n_count])
                temp_edge_inds.append(tri_edges_in[2])

                n_count+=1

    # create new node in the middle
    n_mid = (nodes_data[tri_nodes_in[0]]+nodes_data[tri_nodes_in[1]] + nodes_data[tri_nodes_in[2]])/3
    nodes_data = np.append(nodes_data,[n_mid],axis=0)
    temp_node_inds.append(new_node_index)
    new_node_index +=1

    # now create new edges in the middle, which will not be duplicates
    # creating these from the other edges is bad! and won't work in general

    # create variabe 'dc', for 'dash correction'
    if dashed:
        dc = 1
    else:
        dc = 0

    edges_nodes.append([temp_node_inds[0], temp_node_inds[5+dc]])
    temp_edge_inds.append(new_edge_index)
    new_edge_index += 1
    edges_nodes.append([temp_node_inds[1], temp_node_inds[5+dc]])
    temp_edge_inds.append(new_edge_index)
    new_edge_index += 1
    edges_nodes.append([temp_node_inds[1], temp_node_inds[3]])
    temp_edge_inds.append(new_edge_index)
    new_edge_index += 1
    edges_nodes.append([temp_node_inds[3], temp_node_inds[5+dc]])
    temp_edge_inds.append(new_edge_index)
    new_edge_index += 1
    edges_nodes.append([temp_node_inds[4], temp_node_inds[5+dc]])
    temp_edge_inds.append(new_edge_index)
    if dashed:
        new_edge_index += 1
        edges_nodes.append([temp_node_inds[5], temp_node_inds[6]])
        temp_edge_inds.append(new_edge_index)

    tri_edges_out = [[temp_edge_inds[0],temp_edge_inds[6+dc],temp_edge_inds[5+dc]],
                    [temp_edge_inds[6+dc],temp_edge_inds[7+dc],temp_edge_inds[8+dc]],
                    [temp_edge_inds[1],temp_edge_inds[2],temp_edge_inds[7+dc]],
                    [temp_edge_inds[8+dc],temp_edge_inds[3],temp_edge_inds[9+dc]]]
    if dashed:
        tri_edges_out.append([temp_edge_inds[4],temp_edge_inds[11],temp_edge_inds[10]])
        tri_edges_out.append([temp_edge_inds[5],temp_edge_inds[6],temp_edge_inds[11]])
    else:
        tri_edges_out.append([temp_edge_inds[4],temp_edge_inds[5],temp_edge_inds[9]])

    tri_nodes_out = [[temp_node_inds[0], temp_node_inds[1], temp_node_inds[5+dc]],#may need re-ordering later
                    [temp_node_inds[5+dc], temp_node_inds[1], temp_node_inds[3]],
                    [temp_node_inds[1], temp_node_inds[2], temp_node_inds[3]],
                    [temp_node_inds[5+dc], temp_node_inds[3], temp_node_inds[4]]]
    if dashed:
        tri_nodes_out.append([temp_node_inds[4], temp_node_inds[5], temp_node_inds[6]])
        tri_nodes_out.append([temp_node_inds[5], temp_node_inds[0], temp_node_inds[6]])
    else:
        tri_nodes_out.append([temp_node_inds[4], temp_node_inds[0], temp_node_inds[5]])

    tri_types_out = [8,8,7,8,8]
    if dashed:
        tri_types_out.append(8)

    return tri_edges_out, tri_nodes_out, tri_types_out, edges_nodes, nodes_data, \
            subdivided_edge_indices_old, subdivided_edge_nodes, subdivided_edge_indices_new
    
# 
def subdiv_8(tri_edges_in, tri_nodes_in, edges_nodes, nodes_data, subdivided_edge_indices_old, subdivided_edge_nodes, subdivided_edge_indices_new):
    new_node_index = len(nodes_data)
    new_edge_index = len(edges_nodes)
    temp_edge_inds = [] # this maps edges from the current 0:9 edges to the global (not in a coding sense) edge indices
    temp_node_inds = [] # this maps nodes from the current 0:6 nodes to the global (not in a coding sense) node indices

    n_count=0
    
    for n in tri_edges_in:
    
        #create one new nodes along edge:
        new_node = (nodes_data[edges_nodes[n][0]]+nodes_data[edges_nodes[n][1]])/2

        (n,n_count, nodes_data,edges_nodes,tri_nodes_in,
            new_node_index,new_edge_index, temp_edge_inds,temp_node_inds,
            subdivided_edge_indices_old,subdivided_edge_nodes,subdivided_edge_indices_new) \
            = double_split_angles(new_node, n, n_count, nodes_data,edges_nodes,tri_nodes_in,
            new_node_index,new_edge_index, temp_edge_inds,temp_node_inds,
            subdivided_edge_indices_old,subdivided_edge_nodes,subdivided_edge_indices_new)

    # now create three new edges in the middle, which will not be duplicates
    # creating these from the other edges is bad! and won't work in general
    edges_nodes.append([temp_node_inds[1], temp_node_inds[3]])
    temp_edge_inds.append(new_edge_index)
    new_edge_index += 1
    edges_nodes.append([temp_node_inds[3], temp_node_inds[5]])
    temp_edge_inds.append(new_edge_index)
    new_edge_index += 1
    edges_nodes.append([temp_node_inds[5], temp_node_inds[1]])
    temp_edge_inds.append(new_edge_index)
    # new_edge_index += 1

    # now group all edges together into triangles
    tri_edges_out = []
    tri_edges_out.append([temp_edge_inds[0],temp_edge_inds[8],temp_edge_inds[5]])
    tri_edges_out.append([temp_edge_inds[1],temp_edge_inds[2],temp_edge_inds[6]])
    tri_edges_out.append([temp_edge_inds[7],temp_edge_inds[3],temp_edge_inds[4]])
    tri_edges_out.append([temp_edge_inds[6],temp_edge_inds[7],temp_edge_inds[8]])

    tri_types_out = [8,8,8,8]

    tri_nodes_out = [[temp_node_inds[0], temp_node_inds[1], temp_node_inds[5]],
                    [temp_node_inds[1], temp_node_inds[2], temp_node_inds[3]],
                    [temp_node_inds[5], temp_node_inds[3], temp_node_inds[4]],
                    [temp_node_inds[1], temp_node_inds[3], temp_node_inds[5]]]

    return tri_edges_out, tri_nodes_out, tri_types_out, edges_nodes, nodes_data, \
            subdivided_edge_indices_old, subdivided_edge_nodes, subdivided_edge_indices_new
