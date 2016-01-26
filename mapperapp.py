from Tkinter import *
import numpy as np

from ma import mapper
from interface_draw import draw

class MapperApp(Toplevel):
    def __init__(self, master):
        Toplevel.__init__(self)
        self.sizex = 500
        self.sizey = 300
        # position window next to main window
        self.geometry('{0}x{1}+{2}+{3}'.format(self.sizex, self.sizey, master.sizex+20, 700))
        self.master = master
        self.draw = draw(self)

        self.title("LFS mapper")
        self.resizable(0,0)

        self.canvas = Canvas(self, bg="white", width=self.sizex, height=self.sizey)
        self.canvas.pack()

        self.bind('q', master.exit)

        self.bind("<ButtonRelease>", self.click_drag)#self.click_draw)
        self.bind("<Motion>", self.highlight_bin)


        self.draw_cache = []
        self.hist_cache = []
        self.histdec_cache = []
        self.plot_cache = []

        self.slope=0.4
        self.gen_simple_mapper()
        self.nbins=15.
        self.draw_hist()
        self.draw_mapper_plot()
        self.master.draw_normal_map('lfs')

    def t(self, x, y):
        """transform data coordinates to screen coordinates or the inverse"""
        return (x,self.sizey-y)

    def draw_hist(self):
        hist, bin_edges = np.histogram( self.master.ma.D['lfs'], bins=self.nbins )

        w = self.sizex / self.nbins
        s = self.sizey / hist.max()

        for i, h in enumerate(hist):
            topleft = i*w, h*s
            bottomright = (i+1)*w, 0
            self.hist_cache.append( (self.draw.rectangle(topleft, bottomright), bin_edges[i]) )

        # draw the e=0.4 line, that is the max value recommended by Dey:
        e = self.draw.edge( (0,0), (self.sizex, self.sizex*0.4), fill='#333333', dash=(2,2) )

    def highlight_bin(self, event):
        x,y = self.t(event.x, event.y)
        if x < 0: return
        if x > self.sizex: return

        i1,b1 = self.hist_cache[0]
        i2,b2 = self.hist_cache[1]

        w = self.sizex / self.nbins
        binnr, bin_edge1 = self.hist_cache[int(x / w)]
        bin_edge2 = bin_edge1 + (b2-b1)

        for i, o in self.hist_cache:
            if i == binnr:
                self.canvas.itemconfig(i, fill="blue")
            else:
                self.canvas.itemconfig(i, fill="")

        # find indices corresponding to this bin (or lfs)
        ind = np.where(np.logical_and(self.master.ma.D['lfs']>=bin_edge1, self.master.ma.D['lfs']<=bin_edge2))[0]
        # ind = np.where(self.master.ma.D['lfs']<=bin_edge2)[0]
        
        # draw lfs for those coords
        # self.master.draw.deleteItems(self.master.highlight_cache)
        # for p_i in ind:
        #     self.master.draw_lfs_ball(p_i)

    def draw_mapper_plot(self):
        attributes = {'fill':'grey', 'outline':''}
        interval = self.sizex/4.
        p0x, p0y = 15, 15*self.slope
        p1x, p1y = interval, interval*self.slope
        p2x, p2y = 2*interval, 2*interval*self.slope
        p3x, p3y = 3*interval, 3*interval*self.slope
        p4x, p4y = 4*interval-15, 4*interval*self.slope

        w = 5
        h = self.draw.rectangle((p0x-w, p0y+w), (p0x+w, p0y-w), tags="plot_point", **attributes)
        i = self.draw.rectangle((p1x-w, p1y+w), (p1x+w, p1y-w), tags="plot_point", **attributes)
        j = self.draw.rectangle((p2x-w, p2y+w), (p2x+w, p2y-w), tags="plot_point", **attributes)
        k = self.draw.rectangle((p3x-w, p3y+w), (p3x+w, p3y-w), tags="plot_point", **attributes)
        k = self.draw.rectangle((p4x-w, p4y+w), (p4x+w, p4y-w), tags="plot_point", **attributes)
        self.draw_mapper_plot_edges()

    def get_plot_coords(self):
        def calc_center(bbox):
            tlx, tly, brx, bry = bbox
            w = brx-tlx
            h = bry-tly
            return brx - w/2, bry - w/2

        rectangles = self.canvas.find_withtag("plot_point")

        points = [calc_center(tuple( self.canvas.coords(p) )) for p in rectangles]
        prepend = [(0, points[0][1])]
        postpend = [(self.sizex, points[-1][1])]

        return prepend + points + postpend

    def draw_mapper_plot_edges(self):
        self.draw.deleteItems(self.plot_cache)

        points = self.get_plot_coords()

        # x, y = points[0]
        # self.plot_cache.append( self.canvas.create_line(0,self.sizey,x,y) )

        for i, p1 in enumerate(points[:-1]):
            p2 = points[i+1]
            
            x1, y1 = p1
            x2, y2 = p2

            self.plot_cache.append( self.canvas.create_line(x1,y1,x2,y2) )

        # x, y = points[-1]
        # self.plot_cache.append( self.canvas.create_line(x,y,self.sizex,y))

    def click_drag(self, event):
        x,y = event.x, event.y
        self.last_clickx, self.last_clicky = x,y
        w = 15
        current_i = self.canvas.find_enclosed(x-w, y-w, x+w, y+w)
        
        # current_i = self.canvas.find_withtag('current')
        if len(current_i) != 1: 
            self.click_draw(event)
            return
        i = current_i[0]
        # print current_i
        
        if i in self.canvas.find_withtag("plot_point"):
            self.canvas.itemconfig(i, fill='red')
            self.last_click_i = i
            self.bind("<ButtonRelease>", self.click_drop)

    def click_drop(self, event):
        x,y = event.x, event.y
        i = self.last_click_i

        dx, dy = x-self.last_clickx, y-self.last_clicky
        self.canvas.move(i, dx, dy)
        self.draw_mapper_plot_edges()

        self.canvas.itemconfig(i, fill='blue')
        self.bind("<ButtonRelease>", self.click_drag)

        self.mapper = mapper([ self.t(p[0], p[1]) for p in self.get_plot_coords() ])
        self.master.draw_decimate_lfs(self.mapper)

    def click_draw(self, event):
        self.draw.deleteItems(self.draw_cache)
        x, y = self.t(event.x, event.y)

        self.slope = float(y)/float(x)
        e = self.draw.edge( (0,0), (self.sizex, self.sizex*self.slope) )
        self.draw_cache.append(e)
        self.draw_cache.append(self.draw.point(x,y))

        self.gen_simple_mapper()
        self.master.draw_decimate_lfs(self.mapper)
    
    def gen_simple_mapper(self):
        max_lfs = np.nanmax(self.master.ma.D['lfs'])
        self.mapper = mapper([(0,0), (max_lfs, self.slope*max_lfs)])