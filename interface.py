import Tkinter as tk
from pyflann import FLANN
# from pykdtree.kdtree import KDTree
import numpy as np
import math
from algebra import cos_angle

from canvasapp import CanvasApp
from interface_draw import draw
from mapperapp import MapperApp
from filterapp import FilterApp
from shrinkhistapp import ShrinkHistApp

import canvasvg

HELP = """\
a - create all MA points in one go, press twice for interior and exterior

i - interior skeleton simple
u - interior skeleton with projection
o - outer skeleton simple
p - outer skeleton with projection

c - clear skeleton overlay

b - draw all available balls
move cursor and any available medial ball corresponding to the nearest surface point is drawn
t - toggle between interior and exterior balls

z - open the LFS mapper window for lfs based decimation (use r to suffle decimation order)
f - analysis, filter and decimate window
s - analysis window for shrinking ball processes

h - print this help message
q - quit application
other keys - next step in ball shrinking algorithm
"""

class ShinkkingBallApp(CanvasApp):
    def __init__(self, sbapp_list, filename, densify, sigma_noise, denoise, **args):
        CanvasApp.__init__(self, **args)
        self.sbapp_list = sbapp_list
        self.sbapp_list.append(self)

        self.window_diagonal = math.sqrt(self.sizex**2 + self.sizey**2)
        self.toplevel.title("Shrink the balls [{}] - densify={}x, noise={}, denoise={} ".format(filename, densify, sigma_noise, denoise))
        
        self.toplevel.bind('h', self.print_help)
        
        self.toplevel.bind('a',self.ma_auto_stepper)
        self.toplevel.bind('b',self.draw_all_balls)
        self.toplevel.bind('t',self.toggle_inout)
        self.toplevel.bind('h',self.toggle_ma_stage_geom)

        self.inner_mode = True
        self.draw_stage_geom_mode = 'normal'
        
        self.toplevel.bind('i', self.draw_topo)
        self.toplevel.bind('o', self.draw_topo)
        self.toplevel.bind('u', self.draw_topo)
        self.toplevel.bind('p', self.draw_topo)
        
        self.toplevel.bind('z', self.spawn_mapperapp)
        self.toplevel.bind('f', self.spawn_filterapp)
        self.toplevel.bind('s', self.spawn_shrinkhistapp)
        
        self.toplevel.bind( '1', self.draw_normal_map_lfs )
        self.toplevel.bind( '2', self.draw_normal_map_theta )
        self.toplevel.bind( '3', self.draw_normal_map_lam )
        self.toplevel.bind( '4', self.draw_normal_map_radii )
        self.toplevel.bind( '`', self.draw_normal_map_clear )


        self.toplevel.bind('c', self.clear_overlays)
        self.canvas.pack()

        self.toplevel.bind("<Motion>", self.draw_closest_ball)
        self.toplevel.bind("<Key>", self.ma_step)
        self.toplevel.bind("<ButtonRelease>", self.ma_step)
        self.coordstext = self.canvas.create_text(self.sizex, self.sizey, anchor='se', text='')
        self.ball_info_text = self.canvas.create_text(10, self.sizey, anchor='sw', text='')
        
        self.stage_cache = {1:[], 2:[], 3:[]}
        self.topo_cache = []
        self.highlight_point_cache = []
        self.highlight_cache = []
        self.poly_cache = []
        self.normalmap_cache = []

        self.mapper_window = None
        self.plotter_window = None
        self.shrinkhist_window = None

        self.kdtree = FLANN()

    def toggle_ma_stage_geom(self, event):
        if self.draw_stage_geom_mode == 'normal':
            self.draw_stage_geom_mode = 'dontclear'
        else:
            self.draw_stage_geom_mode = 'normal'

    def spawn_shrinkhistapp(self, event):
        self.ma_ensure_complete()
        self.shrinkhist_window = ShrinkHistApp(self)

    def spawn_mapperapp(self, event):
        self.ma_ensure_complete()
        self.mapper_window = MapperApp(self)

    def spawn_filterapp(self, event):
        self.ma_ensure_complete()
        self.plot_window = FilterApp(self)

    def update_mouse_coords(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y

    def toggle_inout(self, event):
        self.inner_mode = not self.inner_mode

    def print_help(self, event):
        print HELP

    def bind_ma(self, ma, draw_poly=True):
        self.ma = ma
        self.ma_inner = True
        self.ma_complete=False
        self.ma_gen = ma.compute_balls(inner = self.ma_inner)
        minx = ma.D['coords'][:,0].min()
        miny = ma.D['coords'][:,1].min()
        maxx = ma.D['coords'][:,0].max()
        maxy = ma.D['coords'][:,1].max()

        self.set_transform(minx, maxx, miny, maxy)
        self.normal_scale = 0.02 * ( self.window_diagonal / self.scale )

        if draw_poly:
            self.draw.polygon(ma.D['coords'], fill="#eeeeee")
        for p,n in zip(ma.D['coords'], ma.D['normals']):
            self.draw.normal(p, n, s=self.normal_scale, fill='#888888', width=1)

        self.kdtree.build_index(self.ma.D['coords'], algorithm='linear')
        # self.kdtree = KDTree(self.ma.D['coords'])
        
        self.print_help(None)

        self.canvas.update_idletasks()

    def ma_ensure_complete(self):
        while self.ma_complete==False:
            self.ma_auto_stepper(None)

    def ma_auto_stepper(self, event):
        self.ma_stepper(mode='auto_step')

    def ma_step(self, event):
        self.ma_stepper(mode='onestep')

    def ma_stepper(self, mode):
        def step_and_draw():
            d = self.ma_gen.next()
            self.ma_draw_stage(d)
        
        try:
            if mode=='onestep':
                step_and_draw()
            elif mode=='auto_step':
                while True:
                    step_and_draw()
        except StopIteration:
            if not self.ma_inner:
                self.ma.compute_lfs()
                self.ma.compute_lam()
                self.ma.compute_theta()
                self.ma.compute_lam(inner="out")
                self.ma.compute_theta(inner="out")
                self.ma_complete=True
            self.ma_inner = not self.ma_inner
            self.ma_gen = self.ma.compute_balls(self.ma_inner)

    def ma_draw_stage(self, d):
        if d['stage'] == 1:
            try:
                self.stage_cache[2].remove(self.stage_cache[2][2])
            except IndexError:
                pass
            
            self.deleteCache([1,2,3])
            p,n = d['geom']
            l = self.window_diagonal # line length - depends on windows size
            i = self.draw.point(p[0],p[1], size=8, fill='red', outline='')
            j = self.draw.edge( (p[0]+n[0]*l, p[1]+n[1]*l),\
                                (p[0]-n[0]*l, p[1]-n[1]*l), width=1, fill='blue', dash=(4,2) )
            self.stage_cache[1] = [i,j]
            self.canvas.itemconfig(self.coordstext, text=d['msg'])

        elif d['stage'] == 2:
            if self.draw_stage_geom_mode == 'normal':
                self.draw.deleteItems(self.stage_cache[2])
            q,c,r = d['geom']
            i = self.draw.point(q[0],q[1], size=4, fill='blue', outline='')
            j = self.draw.point(c[0],c[1], size=r*self.scale, fill='', outline='blue')
            k = self.draw.point(c[0],c[1], size=2, fill='blue', outline='')
            self.stage_cache[2] = [i,j, k]
            self.canvas.itemconfig(self.coordstext, text=d['msg'])

    def draw_highlight_points(self, key, val, how, inner='in'):
        self.draw.deleteItems(self.highlight_cache)
        for m, v in zip(self.ma.D['ma_coords_'+inner], self.ma.D[key]):
            if not np.isnan(v):
                if how=='greater' and v>val:
                    i = self.draw.point(m[0], m[1], size=4, fill='', outline='red', width=2)
                    self.highlight_cache.append(i)
                elif how=='smaller' and v<val:
                    i = self.draw.point(m[0], m[1], size=4, fill='', outline='red', width=2)
                    self.highlight_cache.append(i)
                elif how=='equal' and v==val:
                    i = self.draw.point(m[0], m[1], size=4, fill='', outline='red', width=2)
                    self.highlight_cache.append(i)

    def draw_topo(self, event):
        if event.char in ['i','u']: inner = 'in'
        elif event.char in ['o','p']: inner = 'out'

        if event.char in ['p', 'u']: project = True
        else: project = False

        self.draw.deleteItems(self.topo_cache)
        self.ma.construct_topo_2d(inner, project)

        for start, end in self.ma.D['ma_linepieces_'+inner]:
            s_e = self.ma.D['ma_coords_'+inner][start]
            e_e = self.ma.D['ma_coords_'+inner][end]
            i = self.draw.edge(s_e, e_e, fill='blue', width=1) 
            self.topo_cache.append(i)

    def draw_all_balls(self, event):
        self.draw.deleteItems(self.highlight_cache)
        for p_i in xrange(self.ma.m):
            self.draw_medial_ball(p_i, with_points=False)

    def draw_closest_ball(self, event):
        # x,y = self.t_(self.mouse_x, self.mouse_y)
        x,y = self.t_(event.x, event.y)
        q = np.array([x,y])
        p_i = self.kdtree.nn_index(q,1)[0][0]
        # p_i = self.kdtree.query(np.array([q]),1)[1][0]

        for sbapp in self.sbapp_list:
            sbapp.highlight_single_ball(p_i)

    def highlight_single_ball(self, p_i):
        if self.inner_mode: inner='in'
        else: inner='out'

        # plot the shrink history of this ball:
        if self.shrinkhist_window is not None:
            self.shrinkhist_window.update_plot(p_i, inner)

        def get_ball_info_text(p_i):
            if not self.ma.D.has_key('lfs'): return ""
            return "lfs\t{0:.2f}\nr\t{2:.2f}\nlambda\t{1:.2f}\ntheta\t{3:.2f} ({4:.2f} deg)\nk\t{5}\nplanar\t{6:.2f} deg".format( \
                self.ma.D['lfs'][p_i], \
                self.ma.D['lam_'+inner][p_i], \
                self.ma.D['ma_radii_'+inner][p_i], \
                self.ma.D['theta_'+inner][p_i], \
                (180/math.pi) * math.acos(self.ma.D['theta_'+inner][p_i]), \
                len(self.ma.D['ma_shrinkhist_'+inner][p_i]), \
                (90/math.pi)*( math.pi - math.acos(self.ma.D['theta_'+inner][p_i]) ) )

        self.draw.deleteItems(self.highlight_point_cache)
        self.draw_medial_ball(p_i)    
        self.draw_lfs_ball(p_i)

        self.canvas.itemconfig(self.ball_info_text, text=get_ball_info_text(p_i) )

    def draw_medial_ball(self, p_i, with_points=True):
        inner = 'out'
        if self.inner_mode: inner = 'in'
        
        p1x, p1y = self.ma.D['coords'][p_i][0], self.ma.D['coords'][p_i][1]
        ma_px, ma_py = self.ma.D['ma_coords_'+inner][p_i][0], self.ma.D['ma_coords_'+inner][p_i][1]

        if not np.isnan(ma_px):
            p2x, p2y = self.ma.D['coords'][ self.ma.D['ma_f2_'+inner][p_i] ][0], self.ma.D['coords'][ self.ma.D['ma_f2_'+inner][p_i] ][1]
            r = self.ma.D['ma_radii_'+inner][p_i]

            ball = self.draw.point(ma_px, ma_py, size = r*self.scale, width=1, fill='', outline='red', dash=(4,2,1))
            if with_points:
                self.highlight_point_cache.append( self.draw.point(p1x, p1y, size = 4, fill='', outline='red', width=2) )
                self.highlight_point_cache.append( self.draw.point(p2x, p2y, size = 4, fill='', outline='purple', width=2) )
                self.highlight_point_cache.append( self.draw.point(ma_px, ma_py, size = 4, fill='', outline='blue', dash=(1), width=2) )
                self.highlight_point_cache.append( ball )
            else:
                self.highlight_cache.append( ball )

    def draw_closest_lfs_ball(self, event):
        # self.draw.deleteItems(self.highlight_cache)

        x,y = self.t_(event.x, event.y)
        q = np.array([x,y])
        p_i = self.kdtree.nn_index(q,1)[0][0]
        # p_i = self.kdtree.query(np.array([q]),1)[1][0]

        self.draw_lfs_ball(p_i)

    def draw_lfs_ball(self, p_i):
        if self.ma.D.has_key('lfs'):
            p1x, p1y = self.ma.D['coords'][p_i][0], self.ma.D['coords'][p_i][1]
            lfs = self.ma.D['lfs'][p_i]
            if not np.isnan(lfs):
                self.highlight_point_cache.append( self.draw.point(p1x, p1y, size = lfs*self.scale, fill='', outline='#888888', dash=(2,1)) )

    def draw_decimate_lfs(self, epsilon):
        self.ma.decimate_lfs(epsilon)

        dropped, total = np.count_nonzero(self.ma.D['decimate_lfs']), self.ma.m
        print 'LFS decimation e={}: {} from {} points are dropped ({:.2f}%)'.format(epsilon, dropped, total, float(dropped)/total *100)

        self.draw.deleteItems(self.poly_cache)
        i = self.draw.polygon_alternating_edge( self.ma.D['coords'][ np.invert(self.ma.D['decimate_lfs']) ], width=3 )
        self.poly_cache.extend(i)

    def draw_decimate_ballco(self, xi, k):
        self.ma.decimate_ballco(xi, k)

        dropped, total = np.count_nonzero(self.ma.D['decimate_ballco']), self.ma.m
        print 'BALLCO decimation xi={}, k={}: {} from {} points are dropped ({:.2f}%)'.format(xi, k, dropped, total, float(dropped)/total *100)

        self.draw.deleteItems(self.poly_cache)
        i = self.draw.polygon_alternating_edge( self.ma.D['coords'][ np.invert(self.ma.D['decimate_ballco']) ], width=3 )
        self.poly_cache.extend(i)

    def draw_normal_map_lfs(self, event):
        self.draw_normal_map('lfs', 40)

    def draw_normal_map_theta(self, event):
        self.draw_normal_map('theta_in', 30)

    def draw_normal_map_lam(self, event):
        self.draw_normal_map('lam_in', 30)

    def draw_normal_map_radii(self, event):
        self.draw_normal_map('ma_radii_in', 30)

    def draw_normal_map_clear(self, event):
        self.draw.deleteItems(self.normalmap_cache)

    def draw_normal_map(self, key, scale=30):
        self.draw.deleteItems(self.normalmap_cache)
        max_val = np.nanmax( self.ma.D[key] )
        for p, p_n, val in zip(self.ma.D['coords'], self.ma.D['normals'], self.ma.D[key]):
            s=scale*(val/max_val)
            i = self.draw.normal(p, p_n, s=s, width=2, fill='red' )
            self.normalmap_cache.append(i)

    def clear_overlays(self, event):
        self.draw.deleteItems(self.topo_cache)
        self.draw.deleteItems(self.highlight_cache)
        self.draw.deleteItems(self.poly_cache)

    def deleteCache(self, stages):
        for s in stages:
            self.draw.deleteItems(self.stage_cache[s])
