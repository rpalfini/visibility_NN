import math
import vis_graph_enum as vge

class param_func:
    params = {}
     # function type

    def __init__(self,ftype):
        self.ftype = ftype

    def get_params(self):
        return self.params

    def evaluate(self,x):
        return

class line(param_func):
    def __init__(self,m,b):
        self.params = {'m': m, 'b': b}
        super().__init__(vge.edge_type.line)

    def is_slope_pos(self):
        return True if self.params['m'] >= 0 else False

    def evaluate(self,x):
        return self.params['m']*x + self.params['b']
        
class circle(param_func):
    def __init__(self,x,y,r,is_slope_pos):
        self.params = {'x': x, 'y': y, 'r': r, 'is_slope_pos': is_slope_pos}
        super().__init__(vge.edge_type.line)

    def evaluate(self,x):
        y = math.sqrt(self.params['r']**2 - (x-self.params['x'])^2)
        if self.params['is_slope_pos']:
            return y + self.params['y']
        else:
            return -y + self.params['y']

class fart(param_func):
    
    def __init__(self, ftype):
        super().__init__(ftype)

        