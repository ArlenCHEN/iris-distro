import irispy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial import HalfspaceIntersection
import mpl_toolkits.mplot3d as a3
import sys
sys.path.append("..")

import drawing
import math
from random import gauss
from scipy.spatial import KDTree
from copy import deepcopy
import os
import shutil

def generate_pts(footprints, resolution, variance, height):
    env_pts = None
    for fp in footprints:
        x_1 = fp[0]
        y_1 = fp[1]
        x_2 = fp[2]
        y_2 = fp[3]

        if x_1 == x_2:
            plane_pts = np.arange(np.min([y_1, y_2]), np.max([y_1, y_2]) + resolution, resolution)
            z_pts     = np.arange(0, height + resolution, resolution)
            pv, zv = np.meshgrid(plane_pts, z_pts)
            y_flatten = pv.flatten()[:, np.newaxis]
            z_flatten = zv.flatten()[:, np.newaxis]

            ## Add disturbance to x coordinates
            x_flatten = np.array([gauss(x_1, math.sqrt(variance)) for i in range(int(y_flatten.shape[0]))])[:, np.newaxis]

            temp_pts = np.hstack((x_flatten, y_flatten))
            temp_pts = np.hstack((temp_pts, z_flatten))

            if(env_pts is None):
                env_pts = temp_pts
            else:
                env_pts = np.vstack((env_pts, temp_pts))
        elif y_1 == y_2:
            plane_pts = np.arange(np.min([x_1, x_2]), np.max([x_1, x_2]) + resolution, resolution)
            z_pts     = np.arange(0, height + resolution, resolution)
            pv, zv = np.meshgrid(plane_pts, z_pts)
            x_flatten = pv.flatten()[:, np.newaxis]
            z_flatten = zv.flatten()[:, np.newaxis]
            
            ## Add disturbance to y coordinates
            y_flatten = np.array([gauss(y_1, math.sqrt(variance)) for i in range(int(x_flatten.shape[0]))])[:, np.newaxis]

            temp_pts = np.hstack((x_flatten, y_flatten))
            temp_pts = np.hstack((temp_pts, z_flatten))

            if(env_pts is None):
                env_pts = temp_pts
            else:
                env_pts = np.vstack((env_pts, temp_pts))
    
    return env_pts

def generate_env(footprints, height, tiny_incre):
    obstacles = []
    for fp in footprints:
        x_1 = fp[0]
        y_1 = fp[1]
        x_2 = fp[2]
        y_2 = fp[3]
        x_3 = 0
        y_3 = 0
        if x_1 == x_2:
            y_3 = y_1
            x_3 = x_1 + tiny_incre
        elif y_1 == y_2:
            x_3 = x_1
            y_3 = y_1 + tiny_incre
        
        obs_1 = np.array([[x_1, y_1, 0], 
                          [x_2, y_2, 0],
                          [x_3, y_3, 0],
                          [x_2, y_2, height]])
        obstacles.append(np.transpose(obs_1))
        obs_2 = np.array([[x_1, y_1, 0], 
                          [x_1, y_1, height],
                          [x_3, y_3, 0],
                          [x_2, y_2, height]])
        obstacles.append(np.transpose(obs_2))
    return obstacles

def pts_conversion(pts_list):
    num_pts = len(pts_list)
    pts_array = np.zeros((num_pts, 3))
    for i in range(num_pts):
        pts_array[i][0] = pts_list[i][0]
        pts_array[i][1] = pts_list[i][1]
        pts_array[i][2] = pts_list[i][2]
    
    return pts_array

def init_seed(seed_resolution, height):
    '''
    Note: Seed points must be generated with CW or C-CW order without overlapping.
    '''
    seed_outlines = np.array([[-3, 2, 3, 2],
                              [3, 2, 3, -2],
                              [3, -2, -3, -2],
                              [-3, -2, -3, 2]])
    
    seed_pts = None

    for i in range(seed_outlines.shape[0]):
        x_1 = seed_outlines[i][0]
        y_1 = seed_outlines[i][1]
        x_2 = seed_outlines[i][2]
        y_2 = seed_outlines[i][3]

        if x_1 == x_2:
            if y_1 > y_2:
                y_pts = -np.arange(y_2, y_1, seed_resolution)[:, np.newaxis]
                # y_pts = np.arange(np.min([y_1, y_2]), np.max([y_1, y_2]), seed_resolution)[:, np.newaxis]
                x_pts = x_1 * np.ones(y_pts.shape)
                
                temp_pts = np.hstack((x_pts, y_pts))

                if seed_pts is None:
                    seed_pts = temp_pts
                else:
                    seed_pts = np.vstack((seed_pts, temp_pts))
            else:
                y_pts = np.arange(y_1, y_2, seed_resolution)[:, np.newaxis]
                # y_pts = np.arange(np.min([y_1, y_2]), np.max([y_1, y_2]), seed_resolution)[:, np.newaxis]
                x_pts = x_1 * np.ones(y_pts.shape)
                
                temp_pts = np.hstack((x_pts, y_pts))

                if seed_pts is None:
                    seed_pts = temp_pts
                else:
                    seed_pts = np.vstack((seed_pts, temp_pts))
        elif y_1 == y_2:
            if x_1 > x_2:
                x_pts = -np.arange(x_2, x_1, seed_resolution)[:, np.newaxis]
                # x_pts = np.arange(np.min([x_1, x_2]), np.max([x_1, x_2]), seed_resolution)[:, np.newaxis]
                y_pts = y_1 * np.ones(x_pts.shape)

                temp_pts = np.hstack((x_pts, y_pts))

                if seed_pts is None:
                    seed_pts = temp_pts
                else:
                    seed_pts = np.vstack((seed_pts, temp_pts))
            else:
                x_pts = np.arange(x_1, x_2, seed_resolution)[:, np.newaxis]
                # x_pts = np.arange(np.min([x_1, x_2]), np.max([x_1, x_2]), seed_resolution)[:, np.newaxis]
                y_pts = y_1 * np.ones(x_pts.shape)


                temp_pts = np.hstack((x_pts, y_pts))

                if seed_pts is None:
                    seed_pts = temp_pts
                else:
                    seed_pts = np.vstack((seed_pts, temp_pts))
    z_pts = 0.5*height*np.ones((seed_pts.shape[0], 1))
    seed_pts = np.hstack((seed_pts, z_pts))

    return seed_pts

def add_dimension(env_pts, tiny_incre):
    obstacle_pts = []
    for i in range(env_pts.shape[0]):
        x = env_pts[i][0]
        y = env_pts[i][1]
        z = env_pts[i][2]

        temp_hull = np.array([x, y, z])[:, np.newaxis]
        temp_hull = np.hstack((temp_hull, np.array([x + tiny_incre, y, z])[:, np.newaxis]))
        temp_hull = np.hstack((temp_hull, np.array([x, y + tiny_incre, z])[:, np.newaxis]))
        temp_hull = np.hstack((temp_hull, np.array([x, y, z + tiny_incre])[:, np.newaxis]))

        obstacle_pts.append(temp_hull)
    
    return obstacle_pts

def generate_spheres(seed_pts, env_pts, sp_bound):
    # print("Shape of seed points: ", seed_pts.shape)
    kdtree=KDTree(env_pts)
    dist,points=kdtree.query(seed_pts,k = 1)
    # print("Shape of dist: ", dist[:, np.newaxis].shape)
    # print("Shape of neasrest points: ", env_pts[points].shape)

    radius = dist[:, np.newaxis]
    b_radius = deepcopy(radius)
    b_radius[b_radius > sp_bound] = sp_bound

    spheres = np.hstack((seed_pts, radius))
    b_spheres = np.hstack((seed_pts, b_radius))
    return spheres, b_spheres

def visualize_spheres(spheres, ax):
    for sp in spheres:
        r = sp[3]
        pi = np.pi
        cos = np.cos
        sin = np.sin
        phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
        x = sp[0] + r*sin(phi)*cos(theta)
        y = sp[1] + r*sin(phi)*sin(theta)
        z = sp[2] + r*cos(phi)

        ax.plot_surface(x, y, z,  rstride=1, cstride=1, color='c', alpha=0.4, linewidth=0)

def baseline_vol(spheres, num_forward):
    total_volume = []
    inter_volume = []
    for i in range(spheres.shape[0]):
        for j in range(1, num_forward):
            idx = (i+j) % spheres.shape[0]

            sp_curr = spheres[i, :]
            sp_next = spheres[idx, :]

            r = sp_curr[-1]
            vol = 4 * np.pi * r**3 / 3

            R = sp_next[-1]
            d = np.linalg.norm(sp_curr[0:3] - sp_next[0:3])

            if d < (R+r):
                inter_vol = np.pi * (R+r-d)**2 * (d**2 + 2*d*r - 3*r**2 + 2*d*R + 6*r*R - 3*R**2) / (12*d)
            else:
                inter_vol = 0

            inter_volume.append(inter_vol)
        
        total_volume.append(vol)
    
    v_1 = sum(total_volume)
    v_2 = sum(inter_volume)

    # print("spheres v_1: ", v_1)
    # print("spheres v_2: ", v_2)

    v = v_1 - v_2
    return v, total_volume

def test_random_obstacles_3d(is_show, pts_var, pts_reso, seed_reso, h, new_path): 
    ## Environment height (lower bound is 0)
    height = h

    ## Amount for dimension increase
    tiny_incre = 0.001

    ## Environment points variance
    pts_variance = pts_var

    ## Environment points resolution
    pts_resolution = pts_reso

    ## Seed points resolution
    seed_resolution = seed_reso

    ## Bounding box half size
    bounding_half_size = 1.0 + 0.1

    num_forward = 5

    ## Set the obstacles as convex hulls(False) or points(True)
    is_obs_pts = True

    is_bounds_global = False

    ## Set the value of the bound
    sp_bound = height / 2

    ## Set up the figure
    fig = plt.figure()
    ax = a3.Axes3D(fig)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-1, 2)

    ## Define the environment's footprints. All footprints are either horizontal or vertical
    footprints = np.array([[-2, 1, 2, 1],
                          [-2, 1, -2, -1],
                          [-2, -1, 2, -1],
                          [2, -1, 2, 1],
                          [-4, 3, 4, 3],
                          [4, 3, 4, -3],
                          [4, -3, -4, -3],
                          [-4, -3, -4, 3]])
    
    ## Generate seed points
    seed_pts = init_seed(seed_resolution, height)

    ## Generate environmental convex hulls
    env_hulls = generate_env(footprints, height, tiny_incre)

    ## Generate environmental points
    env_pts = generate_pts(footprints, pts_resolution, pts_variance, height)

    ## Generate spheres
    spheres, b_spheres = generate_spheres(seed_pts, env_pts, sp_bound)

    # print("Spheres: ", spheres)

    ## Visualize the spheres
    # visualize_spheres(spheres, ax)

    ## Compute volume of baseline method
    b_vol, b_vol_list = baseline_vol(spheres, num_forward)
    bb_vol, bb_vol_list = baseline_vol(b_spheres, num_forward)

    ## Make the environmental points match to the obstacle format required by IRIS
    obstacle_pts = add_dimension(env_pts, tiny_incre)
    
    ## Select the obstacle mode
    if is_obs_pts:
        obstacles = obstacle_pts
    else:
        obstacles = env_hulls
    
    total_volume = 0
    cvx_vol_list = []
    ## Apply IRIS in a for loop
    for i in range(seed_pts.shape[0]):
        seed = seed_pts[i]

        if is_bounds_global:
            bounds = irispy.Polyhedron.from_bounds([-4, -3, 0], 
                                               [4, 3, 1])
        else:
            bounds = irispy.Polyhedron.from_bounds([seed[0] - bounding_half_size, seed[1] - bounding_half_size, 0], 
                                               [seed[0] + bounding_half_size, seed[1] + bounding_half_size, height])
        
        region, debug = irispy.inflate_region(obstacles, seed, bounds=bounds, return_debug_data=True)
    
        pts_array = pts_conversion(region.getPolyhedron().generatorPoints())

        if pts_array.shape[0] > 3:
            hull = ConvexHull(pts_array)
            np.savetxt(new_path + '/cvx_' + str(i) + '.txt', pts_array, delimiter = ' ')
            volume = hull.volume
            # print("Volume: ", volume)
            cvx_vol_list.append(volume)
            total_volume += volume

            # drawing.draw_convhull(pts_array, ax, edgecolor='k', facecolor=(0.4, 0.7, 1.0), alpha=0.3)
            # final_poly = debug.polyhedron_history[-1]
            # final_poly.draw(ax)
        else:
            print(str(i) + "-th convex hull, no enough points!")

    print("Ground-truth volume: ", 40*height)
    print("Baseline volume: ", b_vol)
    print("Bounded baseline volume: ", bb_vol)
    print("Total volume: ", total_volume)
    print(" ")

    temp_data = np.array([40*height, b_vol, bb_vol, total_volume])

    # debug.animate(pause=2.5, show=show)
    # debug.show_results(show=show)

    ## Visualize the environmental points
    ax.scatter(env_pts[:, 0], env_pts[:, 1], env_pts[:, 2], s=5, c = 'g')

    ## Visualize the seed points
    # ax.scatter(seed_pts[0, 0], seed_pts[0, 1], seed_pts[0, 2], s = 380, c = 'green', marker = '+')
    # ax.scatter(seed_pts[1, 0], seed_pts[1, 1], seed_pts[1, 2], s = 380, c = 'blue', marker = 'o')
    ax.scatter(seed_pts[:, 0], seed_pts[:, 1], seed_pts[:, 2], s = 80, c = 'red', marker = '<')

    # for obs in debug.getObstacles():
    #     drawing.draw_convhull(obs.T, ax, edgecolor='k', facecolor='k', alpha=0.1)

    ## Visualize the environmental convex hulls (Note: here is the environmental obstacels, not the extrated convex hulls)
    for hull in env_hulls:
        drawing.draw_convhull(hull.T, ax, edgecolor='k', facecolor='k', alpha=0.1)

    if is_show:
        ax.axis('off')

        ## Top view
        # ax.view_init(azim=90, elev=90)
        plt.show()

    b_vol_array = np.array(b_vol_list)
    bb_vol_array = np.array(bb_vol_list)
    cvx_vol_array = np.array(cvx_vol_list)

    vol_array = np.hstack((b_vol_array, bb_vol_array))
    vol_array = np.hstack((vol_array, cvx_vol_array))

    return temp_data, vol_array, seed_pts.shape[0]

if __name__ == '__main__':
    
    var_enable = True
    pts_reso_enable = False
    seed_reso_enable = False
    h_enable = False

    counter = 0
    data = None
    is_show = True

    shutil.rmtree('./data')
    os.mkdir('./data')
    
    if var_enable:
        var_vol_array = None
        # pts_var_list = [0.01, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.3]
        pts_var_list = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]
        for i in range(len(pts_var_list)):
            pts_var = pts_var_list[i]
            pts_reso = 0.2
            seed_reso = 1.5
            h = 1.0
            
            # pts_var = 0.01
            # pts_reso = 0.2
            # seed_reso = 1.5
            # h = 1.0

            print("In pts_var_list", i)
            print("Counter", counter)
            print("pts variance: ", pts_var)
            print("pts resolution: ", pts_reso)
            print("seed resolution: ", seed_reso)
            print("height: ", h)
            
            new_path = './data/' + str(counter)
            
            try:
                os.mkdir(new_path)
            except OSError:
                print ("Creation of the directory %s failed" % new_path)
            else:
                print ("Successfully created the directory %s " % new_path)

            temp_data, vol_array, num_seeds = test_random_obstacles_3d(is_show, pts_var, pts_reso, seed_reso, h, new_path)
            print("Number of seed points: ", num_seeds)
            
            if var_vol_array is None:
                var_vol_array = vol_array
            else:
                var_vol_array = np.vstack((var_vol_array, vol_array))

            temp_data = np.concatenate([[pts_var_list[i]], temp_data])[:, np.newaxis]

            if data is None:
                data = temp_data
            else:
                data = np.vstack((data, temp_data))

            counter += 1
        np.save("./var_vol_array.npy", var_vol_array)

    if pts_reso_enable:
        pts_reso_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
        for i in range(len(pts_reso_list)):
            pts_var = 0.08
            pts_reso = pts_reso_list[i]
            seed_reso = 1.5
            h = 1.0
            
            print("In pts_reso_list", i)
            print("Counter", counter)
            print("pts variance: ", pts_var)
            print("pts resolution: ", pts_reso)
            print("seed resolution: ", seed_reso)
            print("height: ", h)
            
            new_path = './data/' + str(counter)
            
            try:
                os.mkdir(new_path)
            except OSError:
                print ("Creation of the directory %s failed" % new_path)
            else:
                print ("Successfully created the directory %s " % new_path)

            temp_data, vol_array, num_seeds = test_random_obstacles_3d(is_show, pts_var, pts_reso, seed_reso, h, new_path)

            temp_data = np.concatenate([[pts_reso_list[i]], temp_data])[:, np.newaxis]

            if data is None:
                data = temp_data
            else:
                data = np.vstack((data, temp_data))

            counter += 1
    
    if seed_reso_enable:
        seed_reso_list = [0.7, 0.8, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0]
        for i in range(len(seed_reso_list)):
            
            pts_var = 0.08
            pts_reso = 0.3
            seed_reso = seed_reso_list[i]
            h = 1.0
            
            print("In seed_reso_list", i)
            print("Counter", counter)
            print("pts variance: ", pts_var)
            print("pts resolution: ", pts_reso)
            print("seed resolution: ", seed_reso)
            print("height: ", h)
            
            new_path = './data/' + str(counter)
            
            try:
                os.mkdir(new_path)
            except OSError:
                print ("Creation of the directory %s failed" % new_path)
            else:
                print ("Successfully created the directory %s " % new_path)

            temp_data, vol_array, num_seeds = test_random_obstacles_3d(is_show, pts_var, pts_reso, seed_reso, h, new_path)

            temp_data = np.concatenate([[seed_reso_list[i]], temp_data])[:, np.newaxis]

            if data is None:
                data = temp_data
            else:
                data = np.vstack((data, temp_data))

            counter += 1

    if h_enable:
        h_list = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]
        for i in range(len(h_list)):
            pts_var = 0.08
            pts_reso = 0.3
            seed_reso = 1.5
            h = h_list[i]
            
            print("In h_list", i)
            print("Counter", counter)
            print("pts variance: ", pts_var)
            print("pts resolution: ", pts_reso)
            print("seed resolution: ", seed_reso)
            print("height: ", h)
            
            new_path = './data/' + str(counter)
            
            try:
                os.mkdir(new_path)
            except OSError:
                print ("Creation of the directory %s failed" % new_path)
            else:
                print ("Successfully created the directory %s " % new_path)

            temp_data, vol_array, num_seeds = test_random_obstacles_3d(is_show, pts_var, pts_reso, seed_reso, h, new_path)

            temp_data = np.concatenate([[h_list[i]], temp_data])[:, np.newaxis]

            if data is None:
                data = temp_data
            else:
                data = np.vstack((data, temp_data))

            counter += 1

    np.save("./data.npy", data)
