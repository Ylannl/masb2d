class draw(object):
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