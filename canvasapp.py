import Tkinter as tk
import canvasvg

class CanvasApp(object):
    def __init__(self, master, is_main_window=True, size=(700,700), position=(0,0), content_zoom=0.8):
        self.master = master
        if is_main_window:
            self.toplevel = master
        else:
            self.toplevel = tk.Toplevel(self.master)

        self.sizex, self.sizey = size
        self.positionx, self.positiony = position
        self.contentzoom = content_zoom
        self.toplevel.geometry('{0}x{1}+{2}+{3}'.format(self.sizex, self.sizey, self.positionx, self.positiony))

        self.toplevel.title("")
        self.toplevel.resizable(0,0)

        self.toplevel.bind('q', self.exit)
        self.toplevel.bind('d', self.save_to_disk )

        self.canvas = tk.Canvas(self.toplevel, bg="white", width=self.sizex, height=self.sizey)
        self.canvas.pack()

        self.tx = 0
        self.ty = 0
        self.scale = 1

        self.draw = Draw(self)

    def print_p(self,x,y):
        self.draw.point(11,11)

    def save_to_disk(self, event):
        canvasvg.saveall('canvas.svg', self.canvas)

    def t(self, x, y):
        """transform data coordinates to screen coordinates"""
        x = (x * self.scale) + self.tx
        y = self.sizey - ((y * self.scale) + self.ty)
        return (x,y)

    def t_(self, x, y):
        """transform screen coordinates to data coordinates"""
        x = (x - self.tx)/self.scale
        y = (self.sizey - y - self.ty)/self.scale
        return (x,y)

    def set_transform(self, minx, maxx, miny, maxy):
        """compute parameters to transform data coordinates to screen coordinates"""
        d_x = maxx-minx
        d_y = maxy-miny
        # print d_x, d_y
        c_x = minx + (d_x)/2
        c_y = miny + (d_y)/2

        self.tx = self.sizex/2 - c_x
        self.ty = self.sizey/2 - c_y

        if d_x > d_y:
            self.scale = (self.sizex*self.contentzoom) / d_x
        else:
            self.scale = (self.sizey*self.contentzoom) / d_y

        self.tx = self.sizex/2 - c_x*self.scale
        self.ty = self.sizey/2 - c_y*self.scale

    def exit(self, event):
        print "bye bye."
        self.toplevel.quit()
        self.toplevel.destroy()

class Draw(object):
    """Convenience wrappers around tkinter drawing function to take care of coordinate transformations and other things. 
    Assumes master to have a t() function and a canvas attribute"""
    def __init__(self, master):
        self.m = master

    def point(self, x, y, size=3, **attributes):
        x,y = self.m.t(x,y)
        x-=size
        y-=size
        return self.m.canvas.create_oval(x, y, x+2*size, y+2*size, **attributes)
        
    def polygon(self, ring, **attributes):
        temp = [self.m.t(x[0],x[1]) for x in ring]
        return self.m.canvas.create_polygon(temp, **attributes)

    def polygon_alternating_edge(self, ring, **attributes):
        ring = [self.m.t(x[0],x[1]) for x in ring]

        iL = []
        color="green"
        for i, p in enumerate(ring[:-1]):
            ax, ay = p[0], p[1]
            bx, by = ring[i+1][0], ring[i+1][1]
            
            iL.append(self.m.canvas.create_line(ax, ay, bx, by, fill=color, **attributes))

            if i %2 == 0:
                color="purple"
            else:
                color="green"
        
        p1, p2 = ring[-1], ring[0]
        iL.append(self.m.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill=color, **attributes))
        return iL

    def normal(self, p, p_n, s=20, arrow=None, **attributes):
        self.point(p[0], p[1], fill='#666666', outline='')
        return self.edge(p, (p[0]+p_n[0]*s, p[1]+p_n[1]*s), arrow=arrow, **attributes )
        
    def edge(self, a, b, **attributes):
        ax, ay = self.m.t(a[0], a[1])
        bx, by = self.m.t(b[0], b[1])
        return self.m.canvas.create_line(ax, ay, bx, by, **attributes)
        
    def rectangle(self, topleft, bottomright, **attributes):
        tx,ty = self.m.t(*topleft)
        bx,by = self.m.t(*bottomright)
        return self.m.canvas.create_rectangle(tx,ty,bx,by, **attributes)

    def deleteItems(self, items):
        for i in items:
            self.m.canvas.delete(i)