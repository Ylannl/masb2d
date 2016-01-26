import sys

from ma import MA
from readwrite import *

def main(args, with_visual_appeal=True):
    datadict = read_shp(args.infile, densify_n_times=args.densify, roll=args.roll)
    
    if args.noise != 0:
        add_noise(datadict, args.noise, args.recompute_normals)

    # creating shrinking ball object
    ma = MA(datadict, args.maxr, args.denoise_absmin, args.denoise_delta, args.denoise_min, args.detect_planar)

    from interface import ShinkkingBallApp
    import Tkinter as tk
    root = tk.Tk()
    sba = ShinkkingBallApp([], args.infile, args.densify, args.noise, args.denoise_absmin, master=root, is_main_window=True)
    sba.bind_ma(ma, args.draw_poly)
    root.mainloop()

    return ma

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Point approximate the Medial Axis of input polygon using Shrinking Ball algorithm.')
    parser.add_argument('--infile', help='input .shp', default='test-data/polygon_embr.shp')
    # parser.add_argument('outfile', help='output .shp')

    # group = parser.add_mutually_exclusive_group()
    parser.add_argument('--noise', help='Add gaussian noise, specify variance', default=0, type=float)
    parser.add_argument('--recompute_normals', help='recompute normals after adding noise', default=1, type=int)
    parser.add_argument('--denoise_absmin', help='denoising during construction', default=None, type=float)
    parser.add_argument('--denoise_delta', help='denoising during construction', default=None, type=float)
    parser.add_argument('--denoise_min', help='denoising during construction', default=None, type=float)
    parser.add_argument('--detect_planar', help='detect_planar during construction', default=None, type=float)
    parser.add_argument('--densify', help='densify x times', default=0, type=int)
    parser.add_argument('--roll', help='rotate input points', default=0, type=int)
    parser.add_argument('--maxr', help='maximum ball radius', default=1000, type=float)

    parser.add_argument('--draw_poly', help='draw polygon backdrop', default=1, type=int)

    args = parser.parse_args()
    main(args)