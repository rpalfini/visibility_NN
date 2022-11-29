import visibility_graph as vg
import numpy as np


# fname = "obstacle_locations_test.txt"
fname = "100_obstacle_locations_uniform.txt"

obstacle_courses = vg.read_obstacle_list(fname)

ii = 0
for key in obstacle_courses:
    course = obstacle_courses[key]
    ii += 1
    print(f'course {ii}')
    [o.view() for o in course]
    print(f'Number of obstacles = {len(obstacle_courses)}')