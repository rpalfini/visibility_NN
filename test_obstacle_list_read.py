import visibility_graph as vg
import numpy as np


fname = "obstacle_locations.txt"

obstacle_courses = vg.read_obstacle_list(fname)

ii = 0
for key in obstacle_courses:
    course = obstacle_courses[key]
    ii += 1
    print(f'course {ii}')
    [o.view() for o in course]
