import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import key_press_handler

from Tkinter import *
import numpy as np
import math

# from interface_draw import draw

class FilterApp(Toplevel):
    def __init__(self, master):
        Toplevel.__init__(self)
        self.sizex = 570
        self.sizey = 700
        # position window next to main window
        self.geometry('{0}x{1}+{2}+{3}'.format(self.sizex, self.sizey, master.sizex+20, 0))
        self.master = master

        # self.title("Filterer [{}]".format(master.filename))
        self.minsize(self.sizex, self.sizey)
        # self.resizable(0,0)

        f = self.matplt()
        canvas = FigureCanvasTkAgg(f, master=self)
        canvas.show()
        canvas.get_tk_widget().grid(row=0, columnspan=3, sticky="nsew")#(side=TOP, fill=BOTH, expand=1)

        canvas.mpl_connect('motion_notify_event', self.mouse_plotclick)

        group = LabelFrame(self, text='lambda/theta filtering')
        group.grid(row=1, column=0, sticky='nsew')

        w = Label(group, text='lambda')
        w.pack()
        self.lam_check = IntVar()
        w = Checkbutton(group, text='active', variable=self.lam_check)
        w.pack()
        self.scale_lam = Scale(group, from_=0, to=200, orient=HORIZONTAL, length=150)
        self.scale_lam.set(0)
        self.scale_lam.pack()

        w = Label(group, text='theta')
        w.pack()
        self.theta_check = IntVar()
        w = Checkbutton(group, text='active', variable=self.theta_check)
        w.pack()
        self.scale_theta = Scale(group, from_=0, to=180, resolution=1, orient=HORIZONTAL, length=150)
        self.scale_theta.set(0)
        self.scale_theta.pack()

        group = LabelFrame(self, text='normal map')
        group.grid(row=1, column=1, sticky='nsew')
        self.normalmap_var = StringVar()
        self.normalmap_var.set("none")
        w = OptionMenu(group, self.normalmap_var, "none", "lfs", "theta", "radius", "lambda")
        w.pack()
        w.bind("<ButtonRelease-1>", self.draw_normal_map)

        group = LabelFrame(self, text='thetacon noise')
        group.grid(row=1, column=2, sticky='nsew')
        self.thetacon_check = IntVar()
        w = Checkbutton(group, text='active', variable=self.thetacon_check)
        w.pack()
        self.scale_thetacon_absmin = Scale(group, from_=0, to=180, resolution=1, orient=VERTICAL, length=120)
        self.scale_thetacon_absmin.set(26)
        self.scale_thetacon_absmin.pack(side=LEFT)
        self.scale_thetacon_min = Scale(group, from_=0, to=180, resolution=1, orient=VERTICAL, length=120)
        self.scale_thetacon_min.set(37)
        self.scale_thetacon_min.pack(side=LEFT)
        self.scale_thetacon_delta = Scale(group, from_=0, to=180, resolution=1, orient=VERTICAL, length=120)
        self.scale_thetacon_delta.set(45)
        self.scale_thetacon_delta.pack(side=LEFT)

        group = LabelFrame(self, text='lfs decimation')
        group.grid(row=3, column=0, sticky='nsew')
        w = Button(group, text='Show', command=self.draw_decimate_lfs )
        w.pack()
        self.scale_epsilon = Scale(group, from_=0, to=1, resolution=.02, orient=HORIZONTAL, length=150)
        self.scale_epsilon.set(0.4)
        self.scale_epsilon.pack()

        group = LabelFrame(self, text='ballco decimation')
        group.grid(row=3, column=1, sticky='nsew')
        w = Button(group, text='Show', command=self.draw_decimate_ballco )
        w.pack()
        self.scale_ballco_xi = Scale(group, from_=0, to=1, resolution=.01, orient=HORIZONTAL, length=150)
        self.scale_ballco_xi.set(0.1)
        self.scale_ballco_xi.pack()
        self.ballco_k = Spinbox(group, from_=1, to=15)
        self.ballco_k.pack()

        group = LabelFrame(self, text='radiuscon noise')
        group.grid(row=3, column=2, sticky='nsew')
        self.radiuscon_check = IntVar()
        w = Checkbutton(group, text='active', variable=self.radiuscon_check)
        w.pack()
        self.scale_radiuscon_alpha = Scale(group, from_=0, to=1, resolution=.01, orient=HORIZONTAL, length=150)
        self.scale_radiuscon_alpha.set(0.7)
        self.scale_radiuscon_alpha.pack()
        self.radiuscon_k = Spinbox(group, from_=1, to=15)
        self.radiuscon_k.pack()

        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=0)
        self.columnconfigure(0, weight=1, minsize=self.sizex/3)
        self.columnconfigure(1, weight=1, minsize=self.sizex/3)
        self.columnconfigure(2, weight=1, minsize=self.sizex/3)


        self.bind('q', master.exit)
        self.bind('s', master.spawn_shrinkhistapp)

        self.bind("<ButtonRelease-1>", self.highlight_points)#self.click_draw)

    def draw_decimate_lfs(self):
        self.master.draw_decimate_lfs( epsilon=float(self.scale_epsilon.get()) )

    def draw_decimate_ballco(self):
        self.master.draw_decimate_ballco( xi=float(self.scale_ballco_xi.get()), k=int(self.ballco_k.get()) )

    def draw_normal_map(self, event):
        # if event.widget is self.normalmap_widget:
        normalmap = self.normalmap_var.get()
        if normalmap == "lfs":
            self.master.draw_normal_map_lfs(None)
        elif normalmap == "theta":
            self.master.draw_normal_map_theta(None)
        elif normalmap == "lambda":
            self.master.draw_normal_map_lam(None)
        elif normalmap == "radius":
            self.master.draw_normal_map_radii(None)
        else:
            self.master.draw_normal_map_clear(None)

    def highlight_points(self, event):
        val_lam = float(self.scale_lam.get())
        val_theta = math.cos( float(self.scale_theta.get()) * (math.pi/180) )

        atrr_list = []
        if self.lam_check.get() == 1:
            atrr_list.append( ('lam_in', val_lam, 'smaller') )
        if self.theta_check.get() == 1:
            atrr_list.append( ('theta_in', val_theta, 'greater') )
        

        if self.radiuscon_check.get() == 1:
            val_alpha = float(self.scale_radiuscon_alpha.get())
            val_k = int(self.radiuscon_k.get())
            self.master.ma.filter_radiuscon(alpha=val_alpha, k=val_k)
            atrr_list.append( ('filter_radiuscon', True, 'equal') )

        if self.thetacon_check.get() == 1:
            val_absmin = float(self.scale_thetacon_absmin.get())
            val_min = float(self.scale_thetacon_min.get())
            val_delta = float(self.scale_thetacon_delta.get())
            self.master.ma.filter_thetacon(theta_absmin=val_absmin, theta_min=val_min, theta_delta=val_delta)
            atrr_list.append( ('filter_thetacon', True, 'equal') )

        self.master.draw.deleteItems(self.master.highlight_cache)
        for key, val, how in atrr_list:
            self.master.draw_highlight_points(key, val, how, inner='in')

    def mouse_plotclick(self, event):
        # print "x, y:", event.x, event.y
        # print "xdata, ydata:", event.xdata, event.ydata
        # print event.inaxes, self.axis_inner, self.axis_outer
        if str(event.inaxes) == str(self.axis_inner):
            self.master.inner_mode = True
        elif str(event.inaxes) == str(self.axis_outer):
            self.master.inner_mode = False
        else:
            return

        # if event.xdata is None: return        
        i=0
        while event.xdata > self.master.ma.D['bound_len'][i]:
            if i == self.master.ma.m-1: break
            i+=1

        self.master.highlight_single_ball(i)

    def matplt(self):
        fig = plt.figure()
        ax = fig.add_subplot(211)

        self.master.ma.compute_boundary_lenghts_2d()

        # inner axis
        for key in ['lfs', 'lam_in', 'ma_radii_in']:
            ax.plot(self.master.ma.D['bound_len'], self.master.ma.D[key], label=key)
            # ax.plot(spline_filter(self.master.ma.D['bound_len']), self.master.ma.D[key], label=key)
        ax.legend(loc=2, ncol=3, fontsize='small', columnspacing=1)

        ax_ = ax.twinx()
        for key in ['theta_in']:
            ax_.plot(self.master.ma.D['bound_len'], self.master.ma.D[key], label=key, color='purple')
        ax_.legend(loc=1, fontsize='small', columnspacing=1)
        ax.set_zorder(1)
        ax.set_frame_on(False)
        ax_.set_frame_on(True)

        # outer axis
        bx = fig.add_subplot(212)
        for key in ['lfs', 'lam_out', 'ma_radii_out']:
            bx.plot(self.master.ma.D['bound_len'], self.master.ma.D[key], label=key)
            # bx.plot(spline_filter(self.master.ma.D['bound_len']), self.master.ma.D[key], label=key)
        bx.legend(loc=2, ncol=3, fontsize='small', columnspacing=1)
        bx.set_ylim([0, self.master.ma.SuperR/4])

        bx_ = bx.twinx()
        for key in ['theta_out']:
            bx_.plot(self.master.ma.D['bound_len'], self.master.ma.D[key], label=key, color='purple')
        bx_.legend(loc=1, fontsize='small', columnspacing=1)
        bx.set_zorder(1)
        bx.set_frame_on(False)
        bx_.set_frame_on(True)

        self.axis_inner = ax
        self.axis_outer = bx
        # fourier analysis maybe?
        # from scipy import fft, real
        # bx = fig.add_subplot(212)
        # bx.plot(real(fft(self.ma.D['theta'])) )

        return fig
