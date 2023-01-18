import math
import vis_graph_enum as vge

class param_func:

    def __init__(self,ftype):
        self.ftype = ftype
        self.params = {}

    def view(self):
        return self.params

    def is_slope_pos(self):
        pass

    def evaluate(self):
        pass

class line(param_func):
    def __init__(self,m,b,is_pos):
        super().__init__(vge.edge_type.line)
        self.params = {'m': m, 'b': b, 'is_pos': is_pos}

    def is_slope_pos(self):
        # this function is used to check if label on circle segments should be above or below the circle center
        return self.params['is_pos']

    def evaluate(self,x):
        return self.params['m']*x + self.params['b']
        
class circle(param_func):
    def __init__(self,x,y,r,is_slope_pos):
        super().__init__(vge.edge_type.circle)
        self.params = {'x': x, 'y': y, 'r': r, 'is_slope_pos': is_slope_pos}

    def is_slope_pos(self):
        return self.params['is_slope_pos']

    def evaluate(self,x):
        # y = math.sqrt(self.params['r']**2 - (x-self.params['x'])**2)
        y = math.sqrt(math.abs(self.params['r']**2 - (x-self.params['x'])**2))
        if self.params['is_slope_pos']:
            return y + self.params['y']
        else:
            return -y + self.params['y']
        