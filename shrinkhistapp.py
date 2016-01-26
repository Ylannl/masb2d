import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import key_press_handler

from Tkinter import *
import numpy as np
from numpy.linalg import norm

from algebra import cos_angle, compute_radius
from math import acos, pi


class ShrinkHistApp(Toplevel):
    def __init__(self, master):
        Toplevel.__init__(self)
        self.sizex = 570
        self.sizey = 500
        # position window next to main window
        self.geometry('{0}x{1}+{2}+{3}'.format(self.sizex, self.sizey, master.sizex+20, 700))
        self.master = master

        self.minsize(self.sizex, self.sizey)
        # self.resizable(0,0)

        f = plt.figure()
        self.f = f
        self.ax = f.add_subplot(111)
        self.plotline_a, = self.ax.plot([1,8],[0,180], label='separation angle')
        self.plotline_b, = self.ax.twinx().plot([1,8],[0,1], label='radius', color='red')
        self.plotline_c, = self.ax.twinx().plot([1,8],[0,1], label='lambda', color='green')
        self.ax.set_xlabel('iteration #')
        # self.ax.set_ylabel('separation angle (deg)')
        
        plt.legend([self.plotline_a, self.plotline_b, self.plotline_c], ['angle (deg)', 'radius (wrt initial)', 'lambda (wrt initial)'],loc=1)
        self.ax.xaxis.grid(True)
        self.ax.yaxis.grid(True)

        self.canvas = FigureCanvasTkAgg(f, master=self)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        self.bind('d', self.save_to_disk)
        self.bind('q', master.exit)

    def save_to_disk(self, event):
        self.f.savefig('shrinkhist.pdf', format='pdf')

    def update_plot(self, p_i, inner):

        q_indices = self.master.ma.D['ma_shrinkhist_'+inner][p_i]
        if len(q_indices) == 0: return # perhaps also clear the plot...
        q_coords = self.master.ma.D['coords'][q_indices]
        p_n = self.master.ma.D['normals'][p_i]
        
        # if not is_inner: p_n = -p_n
        p = self.master.ma.D['coords'][p_i]

        radii = [ compute_radius(p,p_n,q) for q in q_coords ]
        centers = [ p - p_n * r for r in radii ]
        thetas = [ acos(cos_angle(p-c,q-c))*(180/pi) for c, q in zip(centers, q_coords) ]
        lambdas = [ norm(p-q) for q in q_coords ]

        r_initial = radii[0]
        radii_proportional = [r/r_initial for r in radii]
        lambda_proportional = [l/lambdas[0] for l in lambdas]

        self.plotline_a.set_xdata(range(1,1+len(thetas)))
        self.plotline_b.set_xdata(range(1,1+len(thetas)))
        self.plotline_c.set_xdata(range(1,1+len(thetas)))
        self.plotline_a.set_ydata(thetas)
        self.plotline_b.set_ydata(radii_proportional)
        self.plotline_c.set_ydata(lambda_proportional)
 
        self.canvas.draw()