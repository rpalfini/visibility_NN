import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../visibility_NN")
import visibility_graph as vg

def plot_obstacles(*obstacles):
    for obstacle in obstacles:
        x_points,y_points = vg.make_circle_points(obstacle)
        plt.plot(x_points, y_points,color='black',linewidth=2)

def plot_start(point):
    plt.scatter(point[0],point[1],color='green',marker="o",linewidth=3,label="start")

def plot_end(point):
     plt.scatter(point[0],point[1],color='red',marker="o",linewidth=3,label="end")

def plot_guess(x,y):
    plt.plot(x,y,color=(0.3010, 0.7450, 0.9330),linewidth=3,label="guess")

def plot_solution(x,y):
    plt.plot(x,y,color='purple',linewidth=3,label='solution')
