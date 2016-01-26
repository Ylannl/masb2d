from shp_reader import *
from ply_reader import *

from StringIO import StringIO

from numpy import nanmax, isnan, count_nonzero
import numpy as np

from pickle import load, dump
import random

import matplotlib.cm as cm

def pickle_out(datadict):
    dump(datadict, open('datadict.pickle', 'w'))
def pickle_in(picklename):
    return load(open(picklename+'.pickle'))

def write_ma(datadict, key=None, gradual_color=False):
    def write(ma_coords, ma_radii, filename, color):
        output = StringIO()

        inmax = nanmax(ma_radii)
        point_count = 0

        i=0
        for p, r in zip(ma_coords, ma_radii):
            if not isnan(r) or r==0 :
                red=green=blue=0

                colorval = 255
                if gradual_color:
                    colorval = int(255 * (inmax - r/inmax))                    
                
                if color == 'red':
                    red=colorval
                elif color == 'blue':
                    blue=colorval
                elif color == 'orange':
                    blue=red = colorval
                else:
                    green=colorval

                filterit = False
                if key !=None:
                    filterit = datadict[key][i]
                i+=1
                if not filterit:
                    print >>output, "{0} {1} {2} {3} {4} {5} 255".format(p[0], p[1], p[2], red,green,blue)
                    point_count += 1

        with open(filename, 'w') as f:
            f.write("COFF\n")
            f.write("{} 0 0\n".format(point_count))
            f.write(output.getvalue())

    write(datadict['ma_coords_in'], datadict['ma_radii_in'], 'ma_points_out_inner.off', 'red')
    write(datadict['ma_coords_out'], datadict['ma_radii_out'], 'ma_points_out_outer.off', 'orange')

def write_ma_plain(datadict):
    with open('ma_points_out.xyz', 'w') as f:
        for p in datadict['ma_coords']:
            if p != None:
                f.write("{0} {1} {2}\n".format(p[0], p[1], p[2]))

def write_ma_filtered(datadict, key):
    with open('ma_points_out_'+key+'.xyz', 'w') as f:
        for p, filtered in zip(datadict['ma_coords_in'], datadict[key]):
            if not isnan(np.sum(p)) and filtered == False and not count_nonzero(p)==0:
                f.write("{0} {1} {2}\n".format(p[0], p[1], p[2]))

def write_lfs(datadict, max_lfs=2):
    with open('lfs_out.off', 'w') as f:
        f.write("COFF\n")
        # max_lfs = nanmax(datadict['lfs'])
        m,n = datadict['coords'].shape
        f.write("{} 0 0\n".format(m))

        for p, lfs in zip(datadict['coords'], datadict['lfs']):
            if not isnan(lfs):
                red=green=blue=0
                colorval = 255-int(255 * (lfs/max_lfs))
                if colorval > 255: colorval = 255
                if colorval < 0: colorval = 0

                rgba = cm.gray([colorval])[0]

                f.write("{0} {1} {2} {3} {4} {5} {6}\n".format(p[0], p[1], p[2], int(rgba[0]*255),int(rgba[1]*255),int(rgba[2]*255),int(rgba[3]*255)))

def write_coord_min_radii(datadict, maxr=1):
    with open('minradii_out.off', 'w') as f:
        f.write("COFF\n")
        # max_lfs = nanmax(datadict['lfs'])
        m,n = datadict['coords'].shape
        f.write("{} 0 0\n".format(m))

        for p, r_in, r_ext in zip(datadict['coords'], datadict['ma_radii_in'], datadict['ma_radii_out']):

            r = 0
            if isnan(r_in):
                if isnan(r_ext):
                    pass
                else:
                    r = r_ext
            elif isnan(r_ext):
                if isnan(r_in):
                    pass
                else:
                    r = r_in
            else:
                r = min([r_in,r_ext])

            red=green=blue=0
            colorval = int(255 * (r/maxr))
            if colorval > 255: colorval = 255
            # colorval = 255-colorval
            green=red=blue = colorval

            f.write("{0} {1} {2} {3} {4} {5} 255\n".format(p[0], p[1], p[2], red,green,blue))

def write_coords_filtered(datadict, key, filename):
    with open(filename+'.xyz', 'w') as f, open(filename+'_c.xyz', 'w') as f_:
        # f.write("OFF\n")
        m,n = datadict['coords'].shape
        # f.write("{} 0 0\n".format(m))
        for p, dec in zip(datadict['coords'], datadict[key]):
            if not dec:
                f.write("{0} {1} {2}\n".format(p[0], p[1], p[2]))
            else:
                f_.write("{0} {1} {2}\n".format(p[0], p[1], p[2]))

def write_coords_randfrac(datadict, frac=0.1):
    with open('coords_frac'+str(frac)+'.xyz', 'w') as f, open('coords_frac'+str(frac)+'_complement.xyz', 'w') as f_:
        # f.write("OFF\n")
        m,n = datadict['coords'].shape
        # f.write("{} 0 0\n".format(m))
        for p in datadict['coords']:
            if random.random() < frac:
                f.write("{0} {1} {2}\n".format(p[0], p[1], p[2]))
            else:
                f_.write("{0} {1} {2}\n".format(p[0], p[1], p[2]))
        # print "Finished writing output!"

def write_coords_shrinkhistlen(datadict):
    l_in = [len(l) for l in datadict['ma_shrinkhist_in']]
    l_out = [len(l) for l in datadict['ma_shrinkhist_out']]
    l_max = [max(ein, eout) for ein,eout in zip(l_in, l_out)]

    ofiles = {}
    for i in xrange(min(l_max), max(l_max)+1):
        ofiles[i] = open('coords_shrink_l'+ str(i) +'.xyz', 'w')

    for l, p in zip(l_max, datadict['coords']):
        ofiles[l].write("{0} {1} {2}\n".format(p[0], p[1], p[2]))

    for of in ofiles.itervalues():
        of.close()
# def write_coords_filtered_off(datadict, key):
#     with open('coords_'+key+'.off', 'w') as f:
#         f.write("COFF\n")
#         m,n = datadict['coords'].shape
#         f.write("{} 0 0\n".format(m))
#         for p, dec in zip(datadict['coords'], datadict[key]):
#             if not dec:
#                 f.write("{0} {1} {2} {3} {4} {5} 255\n".format(p[0], p[1], p[2], 0, 0, 0))

