import numpy as np
from matplotlib import pyplot as plt


class test_plot:
    
    def __init__(self):
        plt.ion()
        self.fig = plt.figure()
        # self.open_fig_second_monitor()
        self.place_figure()

    def place_figure(self):
        self.fig.canvas.manager.window.move(100,400)

    # def open_fig_second_monitor(self):
    #     plt.switch_backend('QT4Agg')

    #     # a little hack to get screen size; from here [1]
    #     mgr = plt.get_current_fig_manager()
    #     mgr.full_screen_toggle()
    #     py = mgr.canvas.height()
    #     px = mgr.canvas.width()
    #     mgr.window.close()
    #     # hack end

    #     x = [i for i in range(0,10)]
        
    #     plt.plot(x)

    #     figManager = plt.get_current_fig_manager()
    #     # if px=0, plot will display on 1st screen
    #     figManager.window.move(px, 0)
    #     figManager.window.showMaximized()
    #     figManager.window.setFocus()

    #     plt.show()

    def act_fig(self):
        plt.figure(self.fig.number)

    def plot(self,x,y):
        self.act_fig()
        plt.plot(x,y)

    def close_plot(self):
        self.act_fig()
        plt.close()




if __name__ == "__main__":
    plotter1 = test_plot()
    plotter2 = test_plot()

    range1 = np.arange(1,10,0.1)
    range2 = np.arange(10,1,-0.1)

    plotter1.plot(range1,range2)
    plotter2.plot(range1,range1)


    print('stalling')


