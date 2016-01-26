import fiona
import numpy as np
from numpy.linalg import norm

def compute_normals_2d(datadict):
    """compute vertex normals by averaging incident edge normals"""
    
    def orthogonal(vec):
        v = np.array([-vec[1], vec[0]])
        return v/norm(v)

    def normal(p,q):
        """returns unit vector orthogonal to pq"""

        # compute vector from p to q
        pq = q-p
        # find a vector that is orthogonal to v and points counter-clockwise wrt v
        return orthogonal(pq)
        
    m,n=datadict['coords'].shape
    datadict['normals'] = np.zeros((m,n))

    for i, p in enumerate(datadict['coords']):        

        if i==0: a = datadict['coords'][-1]
        else: a = datadict['coords'][i-1]
        
        if i==m-1: b = datadict['coords'][0]
        else: b = datadict['coords'][i+1]

        p_n = normal(a,p) + normal(p,b)
        p_n = p_n/norm(p_n)
        
        datadict['normals'][i] = p_n

def add_noise(datadict, sigma, recompute_normals=True):
    """Add gaussian noise in normal direction"""
    from random import gauss

    for i,n in enumerate(datadict['normals']):
        p = datadict['coords'][i]

        datadict['coords'][i] += n * gauss(0,sigma)

    if recompute_normals:
        compute_normals_2d(datadict)

def densify(datadict):
    """Subdivision of every edge using ball fitting to interpolate"""
    from algebra import compute_radius, cos_angle
    from math import acos, cos, sin

    m,n = datadict['coords'].shape
    result = np.zeros((m*2,n))
    # result_n = np.zeros((m*2,n))
    # for i,n in enumerate(datadict['normals']):
    for i in xrange(m):
        p = datadict['coords'][i]
        if i==m-1:
            q = datadict['coords'][0]
        else:
            q = datadict['coords'][i+1]

        # r = compute_radius(p, n, q)
        new = p + (q-p)/2
        # c = p - n*r
        # theta = acos(cos_angle(p-c,q-c))/2
        # rotation = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        
        # new_vec_at_origin = rotation.dot(q-c)
        # print new_vec_at_origin
        # print rotation

        result[2*i] = p
        result[2*i+1] = new

        # result_n[2*i] = n
        # result_n[2*i+1] = new_vec_at_origin / norm(new_vec_at_origin)

    datadict['coords'] = result

def read_shp(infile, densify_n_times=0, roll=0):
    datadict = {}
    # retrieve vertices from polygon in shp file
    shp_meta = None
    with fiona.open(infile, 'r') as shp:
        shp_meta = shp.meta
        for f in shp:
            datadict['coords'] = f['geometry']['coordinates'][0]
    
    # convert to numpy array
    datadict['coords'] = np.array(datadict['coords'][:-1])

    # roll the coords array so that we can control what point is first
    if roll != None:
        datadict['coords'] = np.roll(datadict['coords'], shift=roll, axis=0)

    for i in xrange(densify_n_times):
        densify(datadict)

    compute_normals_2d(datadict)
    # return shp_meta, datadict
    return datadict