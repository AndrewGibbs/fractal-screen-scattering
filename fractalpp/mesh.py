import fractalpp.meshes.koch, fractalpp.meshes.cantor, math, bempp.api
import numpy as np

def koch(prefratal_level, h_max=math.inf, scale = 1, apex_angle = math.pi/3, shift = np.array([0,0])):
    K_nodes,elements_pre_np = fractalpp.meshes.koch.build_snowflake(prefratal_level, h_max, scale, apex_angle, shift)
    elements = np.transpose(elements_pre_np)
    #still need to embed in R^3 before mesh can be input into bempp mesh generator
    third_dim = np.zeros((len(K_nodes),1),dtype=np.float64)
    vertices = np.transpose(np.append(K_nodes,third_dim,axis=1))
    # now use bempp to create the mesh
    return bempp.api.Grid(vertices, elements)

def cantor(J,h_max=np.Inf,M=0,middle=1/3):
    return fractalpp.meshes.cantor.dust(J,h_max,M,middle)