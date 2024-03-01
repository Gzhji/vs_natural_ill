import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')

import argparse
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib.image import imread
import os
import json
import math
import copy
import random
#import pymesh
import numpy as np
import open3d as o3d
import shutil
from pathlib import Path
#from sympy import *
#from sympy.geometry import Point, Segment, Polygon, Triangle
from utils_app.zfm_transform import Point2D, Transformation2D, Transformation3D, TransformationSpherical
from utils_app.zfm_data import load_floor_data
#from datasets.adobe_reader import AdobeReader
from shapely.geometry import Point, Polygon
from imutils import perspective



def get_mesh_info(floor_tex_path):
    mesh = o3d.io.read_triangle_mesh(floor_tex_path)
    vertices = np.asarray(mesh.vertices)
    return vertices

def attach_wall(vertices):
    rmin, rmax = vertices.min(0), vertices.max(0)
    wall_ids = list(range(len(vertices)))
    return rmin, rmax, wall_ids

def sort_coordinates(list_of_xy_coords):
    cx, cy = list_of_xy_coords.mean(0)
    x, y = list_of_xy_coords.T
    angles = np.arctan2(x - cx, y - cy)
    indices = np.argsort(-angles)
    return list_of_xy_coords[indices]

def euclidian_dist(p1,p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def cal_slope(p1,p2):
    slope = (p1[1] - p2[1]) / (p1[0] - p2[0])
    return slope

def get_angle(p1, p2):
    slope = cal_slope(p1, p2)
    angle_radians = math.atan(slope)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

def cal_intercept(dis, slope):
    ang = np.arctan(slope)
    y = math.sin(ang) * dis
    x = math.cos(ang) * dis
    return x, y

def Sort_by_Cootdinates(pts, index = 0):
    sorted_points = pts[pts[:, index].argsort()]
    smallest_points = sorted_points[:2]
    return smallest_points

def cal_ang(p1, p2, p3):
    #return p2 angle
    a = math.sqrt((p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2)
    b = math.sqrt((p1[0] - p3[0]) ** 2 + (p1[1] - p3[1]) ** 2)
    c = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    B = math.degrees(math.acos((b**2 - a**2 - c**2)/(-2*a*c)))
    return B


def floor_angle(floor_tex_path):
    mesh = o3d.io.read_triangle_mesh(floor_tex_path)
    flr_verts = np.asarray(mesh.vertices)
    flr_verts = remove_duplicates(flr_verts)

    reduced_flr_verts = np.empty((0, 3))
    num_flr_verts = flr_verts.shape[0]
    for p in range(num_flr_verts):
        try:
            next_index = (p + 1) % num_flr_verts  # Loop back to start if out of range
            next_next_index = (p + 2) % num_flr_verts
            slope_0 = cal_slope(flr_verts[p], flr_verts[next_index])
            slope_1 = cal_slope(flr_verts[next_index], flr_verts[next_next_index])
            slope_diff = math.atan(slope_0) * 180 / math.pi - math.atan(slope_1) * 180 / math.pi

            if abs(slope_diff) > 10:
                reduced_flr_verts = np.vstack((reduced_flr_verts, flr_verts[next_index]))

        except Exception as e:
            print(f"Error in processing vertex {p}: {e}")

    reduced_flr_verts = np.unique(reduced_flr_verts, axis=0)
    ordered_coords = sort_coordinates(reduced_flr_verts[:, :2])
    reduced_flr_verts[:, :2] = ordered_coords

    num_flr_verts = reduced_flr_verts.shape[0]

    # Then, calculate floor_edges
    if num_flr_verts > 0:  # Ensure there are vertices to avoid IndexError
        floor_edges = [euclidian_dist(reduced_flr_verts[i], reduced_flr_verts[(i + 1) % num_flr_verts]) for i in
                       range(num_flr_verts)]
    else:
        floor_edges = []  # No vertices to calculate edges from

    # Find the longest floor edge
    longest_edge_index = np.argmax(floor_edges)
    p1 = reduced_flr_verts[longest_edge_index]
    p2 = reduced_flr_verts[(longest_edge_index + 1) % num_flr_verts]
    p3 = [p2[0], p1[1], 0]

    flr_edge_ang = cal_ang(p1[:2], p2[:2], p3[:2]) - 90
    if longest_edge_index <= 6:
        adjusted_index = longest_edge_index - 1
    else:
        adjusted_index = 0
    return flr_edge_ang, tuple(p1[:2]), tuple(p2[:2]), floor_edges, reduced_flr_verts, adjusted_index


def remove_duplicates(flr_verts):
    # Identify unique elements based on the first column, preserving order
    _, idx = np.unique(flr_verts[:, 0], return_index=True)
    unique_idxs = np.sort(idx)

    # Reconstruct the vertices array with unique first column and zeros for the third column
    unique_verts = flr_verts[unique_idxs, :2]  # Keep first two columns of unique rows
    zeros_col = np.zeros((len(unique_idxs), 1))  # Create a column of zeros

    # Concatenate to form the new vertices array
    new_list = np.concatenate((unique_verts, zeros_col), axis=1)
    return new_list


def array2pt(vert):
    vert1, vert2 = vert[0][:2], vert[1][:2]
    p1 = (vert1[0], vert1[1])
    p2 = (vert2[0], vert2[1])
    return p1, p2


def facad_vertices(floor_tex_path):
    _, _, _, _, flr_verts, i = floor_angle(floor_tex_path)
    try:
        f_vert = (flr_verts[i+1][:2], flr_verts[i+2][:2])
    except:
        f_vert = (flr_verts[i+1][:2], flr_verts[0][:2])
    return f_vert


def edge_for_furn(floor_tex_path):
    _, _, _, _, flr_verts, i = floor_angle(floor_tex_path)
    flr_vert = [flr_verts[i][:2], flr_verts[i+1][:2]]
    flr_vert = array2pt(flr_vert)
    return flr_vert

def sort_furn_pt(pts):
    # Convert to numpy array for easier manipulation
    pts = np.array(pts)
    # Calculate centroid
    centroid = np.mean(pts, axis=0)
    # Function to categorize points as above or below centroid
    def is_above_centroid(point):
        return point[1] > centroid[1]

    # Separate points based on their position relative to the centroid
    above = [pt for pt in pts if is_above_centroid(pt)]
    below = [pt for pt in pts if not is_above_centroid(pt)]

    # Sort each group by x coordinate to determine left-right orientation
    above_sorted = sorted(above, key=lambda x: x[0])
    below_sorted = sorted(below, key=lambda x: x[0])

    # Arrange points in the desired order: [top_left, top_right, bottom_left, bottom_right]
    # Assuming 'above' contains the top points and 'below' contains the bottom points
    if len(above_sorted) > 1 and len(below_sorted) > 1:
        top_left, top_right = above_sorted[0], above_sorted[-1]
        bottom_left, bottom_right = below_sorted[0], below_sorted[-1]
    else:
        # Fallback in case the above/below separation doesn't work as expected
        # This can happen if points are collinear or if it's a degenerate case
        print("Fallback to default sorting due to unexpected point distribution.")
        sorted_by_y = sorted(pts, key=lambda x: x[1], reverse=True)
        top_left, top_right = sorted(sorted_by_y[:2], key=lambda x: x[0])
        bottom_left, bottom_right = sorted(sorted_by_y[2:], key=lambda x: x[0])
    return np.array([top_left, top_right, bottom_left, bottom_right])

def find_mid(input_list):
    middle = float(len(input_list))/2
    if middle % 2 != 0:
        return input_list[int(middle - .5)], middle
    else:
        return input_list[int(middle)], middle

def find_mid_pt(p1, p2):
    pt = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
    return pt

def dis_of_lines(p1, p2, p3, p4):
    #a, "x + ", b, "y = ", c
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b1 = (p1[0] * p2[1] - p2[0] * p1[1]) / (p1[0] - p2[0])
    b2 = (p3[0] * p4[1] - p4[0] * p3[1]) / (p3[0] - p4[0])
    #dis between two lines
    d = abs(b2 - b1) / ((m * m) - 1)
    b = abs(b1 - b2)
    return d, b, m

def furn_orient(furn_path, default_ang = 0):
    #rotate to North-South arrow
    furn_mesh = o3d.io.read_triangle_mesh(furn_path)
    furn_verts = np.asarray(furn_mesh.vertices)
    a_min, a_max, _ = attach_wall(furn_verts)

    a1 = (a_min[0], a_min[1])
    a2 = (a_max[0], a_max[1])
    a3 = (a_min[0], a_max[1])

    if a_min[1] == a_max[1]:
        angle = 0
    else:
        angle = cal_ang(a1, a2, a3)
    rotation_angle1 = np.radians(angle + default_ang)
    R1 = furn_mesh.get_rotation_matrix_from_xyz((0, 0, rotation_angle1))
    furn_mesh_r1 = furn_mesh.rotate(R1, center=(0, 0, 0))

    #rotate to bed head at the top
    furn_verts_r = np.asarray(furn_mesh_r1.vertices)
    b_min, b_max, _ = attach_wall(furn_verts_r)
    if b_min[2] < b_max[2]:
        furn_ang = 180
    else:
        furn_ang = 0
    rotation_angle2 = np.radians(furn_ang)
    R2 = furn_mesh.get_rotation_matrix_from_xyz((0, 0, rotation_angle2))
    furn_mesh_r2 = furn_mesh.rotate(R2, center=(0, 0, 0))
    # #visualization
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
    # o3d.visualization.draw_geometries([furn_mesh_r2, frame])
    bbox = o3d.geometry.AxisAlignedBoundingBox(a_min, a_max)
    aabb_pts = np.asarray(bbox.get_box_points())
    #o3d.visualization.draw_geometries([furn_mesh_r2, bbox])
    return furn_mesh_r2, aabb_pts

def furn_size(furn_path):
    furn_mesh = o3d.io.read_triangle_mesh(furn_path)
    furn_verts = np.asarray(furn_mesh.vertices)
    a_min, a_max, _ = attach_wall(furn_verts)
    bbox = o3d.geometry.AxisAlignedBoundingBox(a_min, a_max)
    aabb_pts = np.asarray(bbox.get_box_points())
    aabb_pts = aabb_pts[(aabb_pts[:, 2] == 0)]

    p0 = [aabb_pts[0][0], aabb_pts[0][1]]
    p1 = [aabb_pts[1][0], aabb_pts[1][1]]
    p2 = [aabb_pts[2][0], aabb_pts[2][1]]
    #p3 = [aabb_pts[3][0], aabb_pts[3][1]]
    furn_edge1 = euclidian_dist(p0, p1)
    furn_edge2 = euclidian_dist(p0, p2)
    if furn_edge1 > furn_edge2:
        len = furn_edge1
        wid = furn_edge2
    else:
        len = furn_edge2
        wid = furn_edge1
    return len, wid


def furn_orient(aabb_pts):
    p0 = [aabb_pts[0][0], aabb_pts[0][1]]
    p1 = [aabb_pts[1][0], aabb_pts[1][1]]
    p2 = [aabb_pts[2][0], aabb_pts[2][1]]
    #p3 = [aabb_pts[3][0], aabb_pts[3][1]]
    furn_edge1 = euclidian_dist(p0, p1)
    furn_edge2 = euclidian_dist(p0, p2)
    if furn_edge1 > furn_edge2:
        longside_slop = cal_slope(p0, p1)
    else:
        longside_slop = cal_slope(p0, p2)
    return longside_slop

def aabb_orient(furn_path):
    #rotate to North-South arrow
    try:
        furn_mesh = o3d.io.read_triangle_mesh(furn_path)
    except:
        return None
    furn_verts = np.asarray(furn_mesh.vertices)
    a_min, a_max, _ = attach_wall(furn_verts)
    bbox = o3d.geometry.AxisAlignedBoundingBox(a_min, a_max)
    aabb_pts = np.asarray(bbox.get_box_points())
    aabb_pts = aabb_pts[(aabb_pts[:, 2] == 0)]

    points = aabb_pts[:, :2]  # Extract only the x, y coordinates
    p0, p1, p2 = points[:3]  # first three pts

    edge1_length = euclidian_dist(p0, p1)
    edge2_length = euclidian_dist(p0, p2)

    long_side_points = (p0, p1) if edge1_length > edge2_length else (p0, p2)

    if long_side_points[0][0] == long_side_points[1][0]:
        furn_ang = 90
    elif long_side_points[0][1] == long_side_points[1][1]:
        furn_ang = 0
    else:
        slope = cal_slope(*long_side_points)
        furn_ang = math.atan(slope) * 180 / math.pi

    rotation_angle = np.radians(furn_ang)
    R = furn_mesh.get_rotation_matrix_from_xyz((0, 0, rotation_angle))
    furn_mesh_r = furn_mesh.rotate(R, center=(0, 0, 0))

    return furn_mesh_r, furn_ang, aabb_pts

def save4x4matrix(ang = 0, tx = 0, ty = 0):
    matrix = np.array([[np.cos(ang), -np.sin(ang), 0, tx],
                       [np.sin(ang), np.cos(ang), 0, ty],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
    return matrix

def pt_line(pt, m, b):
    y = m * pt[0] + b
    if y > pt[1]: #pt under line
        dis = y - pt[1]
    elif y == pt[1]:
        dis = 0
    else: #pt above line
        dis = y - pt[1]
    return dis

def pt2line(p1, p2):
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - m*p1[0]
    return m, b

def pt_dis2line(pt, m, b):
    b_new = pt[1] + (1/m) * pt[0]

    x = ((-1)*(-b_new)-(1)*b)/(m*(1)-(1/m)*(-1))
    y = (m*(-b_new)-(1/m)*b)/((-1)*(1/m) - (1)*m)
    return x, y

def isRecOverlap(verts0,verts1):
    Pts0 = sort_furn_pt(verts0)
    Pts1 = sort_furn_pt(verts1)

    #first furn
    m01, a01 = pt2line(Pts0[0,0], Pts0[0,1])
    m23, a23 = pt2line(Pts0[1,0], Pts0[1,1])
    m02, a02 = pt2line(Pts0[0,0], Pts0[1,0])
    m13, a13 = pt2line(Pts0[0,1], Pts0[1,1])

    #right cornor overlapped
    if pt_line(Pts1[1,1], m01, a01) > 0 and pt_line(Pts1[1,1], m23, a23) < 0:
        # pt under the line m01,a01 and above line m23, a23
        if pt_line(Pts1[1,1], m13, a13) > 0 and pt_line(Pts1[1,1], m02, a02) < 0:
            # pt under the line m13,a13 and above line m02, a02
            # print('in the rec')
            x, y = pt_dis2line(Pts1[1,1], m02, a02)
            dis_x = x - Pts1[1,1][0]
            dis_y = y - Pts1[1,1][1]
            # print('x', x, 'y', y)
            # print('dis_x', dis_x, 'dis_y', dis_y)
        else:
            dis_x, dis_y = 0, 0

    elif pt_line(Pts1[0,1], m01, a01) > 0 and pt_line(Pts1[0,1], m23, a23) < 0:
        # pt under the line m01,a01 and above line m23, a23
        if pt_line(Pts1[0,1], m13, a13) > 0 and pt_line(Pts1[0,1], m02, a02) < 0:
            # pt under the line m13,a13 and above line m02, a02
            x, y = pt_dis2line(Pts1[0,1], m02, a02)
            dis_x = x - Pts1[0,1][0]
            dis_y = y - Pts1[0,1][1]
        else:
            dis_x, dis_y = 0, 0
    else:
        dis_x, dis_y = 0, 0
    return dis_x, dis_y

def Plot_2d_2set(set1, set2, label1, label2, export_img):
    x1 = set1[:, 0]
    y1 = set1[:, 1]
    x2 = set2[:, 0]
    y2 = set2[:, 1]

    plt.scatter(x1, y1, color='red', label=label1, alpha=0.75)
    plt.plot(x1, y1, color='red', alpha=0.75)
    plt.scatter(x2, y2, color='blue', label=label2, alpha=0.75)
    plt.plot(x2, y2, color='blue', alpha=0.75)
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.legend()
    plt.savefig(export_img, dpi=300)  # Adjust dpi for resolution
    plt.clf()


def place_furniture(table_obj_path, floor_tex_path,
                facewindow = 'True', dis2wall = 1, dis2facd = 1,
                reference_pt = 2, name = 'bed'):

    furn_mesh, _, aabb_pts = aabb_orient(table_obj_path) #get furn bounding box
    _, _, _, _, full_flr_verts, _ = floor_angle(floor_tex_path) #slop between long floor edge and y axis


    """
    00: sort floor points to find floor and facade vertices
    """
    #sort floor and facade verts based on x and y value
    facad_vert = Sort_by_Cootdinates(full_flr_verts, index=1)
    flr_vert = Sort_by_Cootdinates(full_flr_verts, index = 0)

    #compute the slope for facade and floor edge
    facad_edge_slop = cal_slope(facad_vert[0], facad_vert[1])
    flr_edge_slop = cal_slope(flr_vert[0], flr_vert[1])

    # adjust orientation based on the floor
    flr_angle = get_angle(flr_vert[0], flr_vert[1])
    add_ang = 180 if facewindow == 'True' else 0

    #rotate furn mesh
    rotation_angle = np.radians(flr_angle + add_ang)
    R = furn_mesh.get_rotation_matrix_from_xyz((0, 0, rotation_angle))
    furn_mesh_r = furn_mesh.rotate(R, center=(0, 0, 0))


    """
    01: rotate aabb pt around z axis 
    """
    #rotate furn long side pt
    #add 90 is because the rotation matrix has different default postion as mesh rotation
    furn_pt_angle = np.radians(flr_angle + add_ang + 90)
    rotation_matrix = np.array([[np.cos(furn_pt_angle), -np.sin(furn_pt_angle), 0],
                                [np.sin(furn_pt_angle), np.cos(furn_pt_angle), 0],
                                [0, 0, 1]])
    four_verts_r = (rotation_matrix @ aabb_pts.T).T

    #get upperright pt
    four_verts_r = sort_furn_pt(four_verts_r)
    furn_align_pt_1 = four_verts_r[reference_pt]
    Plot_2d_2set(set1 = aabb_pts, set2 = full_flr_verts,
                 label1 = 'aabb_pts', label2 = 'flr_vert', export_img = name + '_00-ori_aabb.png')
    Plot_2d_2set(set1 = four_verts_r, set2 = full_flr_verts,
                 label1 = 'four_verts_r', label2 = 'flr_vert', export_img = name + '_01-1fst_flr_rotate.png')


    """
    02: cal dis2wall to move along the facade edge
    """
    # compute facade length and furniture length
    facd_len = euclidian_dist(facad_vert[0], facad_vert[1])
    furn_len = euclidian_dist(aabb_pts[0], aabb_pts[2])

    #valid dis is facade lens - furn_len
    move_dis1 = facd_len - furn_len
    dis1 = move_dis1 * dis2wall #ratio dis to wall

    # compute translation vector
    x1, y1 = cal_intercept(dis1, facad_edge_slop)
    aligh_pt_1 = np.array(facad_vert[0]) + np.array([x1, y1, 0])
    furn2wall_vector = aligh_pt_1 - furn_align_pt_1

    #apply translate vector
    four_verts_2wall = four_verts_r + furn2wall_vector
    Plot_2d_2set(set1 = four_verts_2wall, set2 = full_flr_verts,
                 label1 = 'four_verts_2wall', label2 = 'flr_vert', export_img = name + '_02-four_verts_2wall.png')


    """
    03: cal dis2facad to move on the long side wall
    """
    # compute facade length and furniture length
    room_len = euclidian_dist(flr_vert[0], flr_vert[1])
    furn_wid = euclidian_dist(aabb_pts[0], aabb_pts[1])

    #valid dis is floor edge - furn_len
    move_dis2 = room_len - furn_wid
    dis2 = move_dis2 * dis2facd #ratio dis to wall

    # compute translation vector
    a1, b1 = cal_intercept(dis2, flr_edge_slop)
    aligh_pt_2 = aligh_pt_1 - np.array([a1, b1, 0])
    furn_align_pt_2 = four_verts_2wall[reference_pt]
    furn2facad_vector = [aligh_pt_2[0], aligh_pt_2[1], 0] - furn_align_pt_2

    #apply translate vector
    four_verts_2facad = four_verts_2wall + furn2facad_vector
    Plot_2d_2set(set1 = four_verts_2facad, set2 = full_flr_verts,
                 label1 = 'four_verts_2facad', label2 = 'flr_vert', export_img = name + '_03-four_verts_2facad.png')

    """
    04: apply translation to mesh 
    """
    all_trans_vector = furn2facad_vector + furn2wall_vector
    mesh_align = copy.deepcopy(furn_mesh_r).translate((all_trans_vector[0], all_trans_vector[1], 0))

    #save 4x4 matrix
    matrix = save4x4matrix(ang=rotation_angle, tx=all_trans_vector[0], ty=all_trans_vector[1])
    covers = four_verts_r + all_trans_vector
    return mesh_align, covers, matrix


def custom_draw_geometry_with_custom_fov(pcd, fov_step, n):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.mesh_show_wireframe = True
    vis.update_renderer()
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=fov_step)
    image = vis.capture_screen_float_buffer(do_render=True)
    plt.imsave(str(n) + '.png', np.asarray(image), dpi=1)
    vis.run()
    vis.destroy_window()


def export_para(iteration=1):
    if iteration == 1:
        parameters = {'couch': {'facewindow': 'False', 'dis2wall': '1.0', 'dis2facd': '0.0'},
                      'bed'  : {'facewindow': 'False', 'dis2wall': '0.0', 'dis2facd': '0.0'},
                      'table': {'facewindow': 'False', 'dis2wall': '0.0', 'dis2facd': '0.8'},
                      'chair': {'facewindow': 'True', 'dis2wall': '0.6', 'dis2facd': '0.9'},
                      'shelf': {'facewindow': 'True', 'dis2wall': '0.2', 'dis2facd': '0.9'}}
    elif iteration == 2:
        parameters = {'couch': {'facewindow': 'False', 'dis2wall': '1.0', 'dis2facd': '0.8'},
                      'bed'  : {'facewindow': 'False', 'dis2wall': '0.0', 'dis2facd': '0.3'},
                      'table': {'facewindow': 'False', 'dis2wall': '0.0', 'dis2facd': '0.0'},
                      'chair': {'facewindow': 'True', 'dis2wall': '0.7', 'dis2facd': '0.9'},
                      'shelf': {'facewindow': 'True', 'dis2wall': '0.3', 'dis2facd': '0.9'}}
    elif iteration == 3:
        parameters = {'couch': {'facewindow': 'False', 'dis2wall': '0.0', 'dis2facd': '0.1'},
                      'bed'  : {'facewindow': 'True', 'dis2wall': '0.9', 'dis2facd': '0.65'},
                      'table': {'facewindow': 'False' , 'dis2wall': '0.0', 'dis2facd': '0.9'},
                      'chair': {'facewindow': 'False', 'dis2wall': '0.95', 'dis2facd': '0.0'},
                      'shelf': {'facewindow': 'False', 'dis2wall': '0.0', 'dis2facd': '0.9'}}
    elif iteration == 4:
        parameters = {'couch': {'facewindow': 'False', 'dis2wall': '1.0', 'dis2facd': '0.0'},
                      'bed'  : {'facewindow': 'True' , 'dis2wall': '0.3', 'dis2facd': '0.05'},
                      'table': {'facewindow': 'True' , 'dis2wall': '0.05', 'dis2facd': '0.0'},
                      'chair': {'facewindow': 'False', 'dis2wall': '0.2', 'dis2facd': '0.1'},
                      'shelf': {'facewindow': 'True', 'dis2wall': '0.15', 'dis2facd': '0.95'}}
    elif iteration == 5:
        parameters = {'couch': {'facewindow': 'False', 'dis2wall': '1.0', 'dis2facd': '0.0'},
                      'bed'  : {'facewindow': 'True' , 'dis2wall': '0.8', 'dis2facd': '0.6'},
                      'table': {'facewindow': 'True' , 'dis2wall': '0.05', 'dis2facd': '0.0'},
                      'chair': {'facewindow': 'True' , 'dis2wall': '0.7', 'dis2facd': '0.9'},
                      'shelf': {'facewindow': 'True', 'dis2wall': '0.2', 'dis2facd': '0.9'}}
    elif iteration == 6:
        parameters = {'couch': {'facewindow': 'False', 'dis2wall': '1', 'dis2facd': '1.0'},
                      'bed'  : {'facewindow': 'True' , 'dis2wall': '0.0', 'dis2facd': '0.5'},
                      'table': {'facewindow': 'False', 'dis2wall': '0.0', 'dis2facd': '0.0'},
                      'chair': {'facewindow': 'False' , 'dis2wall': '0.95', 'dis2facd': '0.05'},
                      'shelf': {'facewindow': 'True', 'dis2wall': '0.2', 'dis2facd': '0.9'}}
    elif iteration == 7:
        parameters = {'couch': {'facewindow': 'True' , 'dis2wall': '0.9', 'dis2facd': '0.95'},
                      'bed'  : {'facewindow': 'True' , 'dis2wall': '0.9', 'dis2facd': '0.1'},
                      'table': {'facewindow': 'False', 'dis2wall': '0.0', 'dis2facd': '0.0'},
                      'chair': {'facewindow': 'True' , 'dis2wall': '1.0', 'dis2facd': '0.8'},
                      'shelf': {'facewindow': 'True', 'dis2wall': '0.1', 'dis2facd': '0.6'}}
    elif iteration == 8:
        parameters = {'couch': {'facewindow': 'True' , 'dis2wall': '0.85', 'dis2facd': '0.95'},
                      'bed'  : {'facewindow': 'False', 'dis2wall': '0.0', 'dis2facd': '0.5'},
                      'table': {'facewindow': 'False', 'dis2wall': '0.0', 'dis2facd': '0.0'},
                      'chair': {'facewindow': 'True' , 'dis2wall': '1.0', 'dis2facd': '0.05'},
                      'shelf': {'facewindow': 'False', 'dis2wall': '1.0', 'dis2facd': '0.9'}}
    else:
        parameters = {'couch': {'facewindow': 'False', 'dis2wall': '0.0', 'dis2facd': '0.0'},
                      'bed'  : {'facewindow': 'False', 'dis2wall': '0.0', 'dis2facd': '0.0'},
                      'table': {'facewindow': 'False', 'dis2wall': '0.0', 'dis2facd': '0.6'},
                      'chair': {'facewindow': 'True', 'dis2wall': '0.6', 'dis2facd': '0.9'},
                      'shelf': {'facewindow': 'False', 'dis2wall': '0.0', 'dis2facd': '0.9'}}

    return parameters


def show_all(couch_obj_path, bed_obj_path, floor_tex_path, table_obj_path, chair_obj_path, num = 1):

    #load spatial parameters
    param = export_para(iteration = num)
    print(param)
    bed, bed_cover, bed_matrix = place_furniture(bed_obj_path, floor_tex_path, facewindow=param['bed']['facewindow'],
                                                    dis2wall=float(param['bed']['dis2wall']), dis2facd=float(param['bed']['dis2facd']),
                                                    reference_pt=2, name = 'bed') #2x2 aabb pts, 2 refer to left-down pt in 4pts matrix
    couch, couch_cover, couch_matrix = place_furniture(couch_obj_path, floor_tex_path, facewindow=param['couch']['facewindow'],
                                                    dis2wall=float(param['couch']['dis2wall']), dis2facd=float(param['couch']['dis2facd']),
                                                    reference_pt=2, name = 'couch')
    table, table_cover, table_matrix = place_furniture(table_obj_path, floor_tex_path, facewindow=param['table']['facewindow'],
                                                    dis2wall=float(param['table']['dis2wall']), dis2facd=float(param['table']['dis2facd']),
                                                    reference_pt=2, name = 'table')
    chair, chair_cover, chair_matrix = place_furniture(chair_obj_path, floor_tex_path, facewindow=param['chair']['facewindow'],
                                                    dis2wall=float(param['chair']['dis2wall']), dis2facd=float(param['chair']['dis2facd']),
                                                    reference_pt=2, name = 'chair')



    o3d.io.write_triangle_mesh(filename = 'bed_03.obj', mesh = bed)
    o3d.io.write_triangle_mesh(filename = 'b_couch_2_159.obj', mesh=couch)
    o3d.io.write_triangle_mesh(filename = 'desk01.obj', mesh=table)
    o3d.io.write_triangle_mesh(filename = 'chair_08.obj', mesh=chair)


    flr_mesh = o3d.io.read_triangle_mesh(floor_tex_path)
    mesh = bed + flr_mesh + couch + table + bed + chair

    #save into matrix
    matrix = []
    matrix.append('couch_matrix')
    matrix.append(couch_matrix)
    matrix.append('bed_matrix')
    matrix.append(bed_matrix)
    matrix.append('table_matrix')
    matrix.append(table_matrix)
    matrix.append('chair_matrix')
    matrix.append(chair_matrix)

    return mesh, matrix



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='') #../data/zind/scenes/floormesh/mew/new+1F+living+10141318
    parser.add_argument('--bed-path', type=str, default='../00_Data/3D_Future/7ce3cd02-6734-456b-b13f-88a4a553252b')
    parser.add_argument('--couch-path', type=str, default='../00_Data/3D_Future/7c130177-89f3-4b09-8b80-694d9ef9f6c7')#
    parser.add_argument('--table-path', type=str, default='../00_Data/3D_Future/8dd92822-59f6-4010-8f52-0c812d9b0dd1')
    parser.add_argument('--chair-path', type=str, default='../00_Data/3D_Future/2e0c52c6-5abd-4c76-87a2-8112e3fc7e14')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_args()

    floor_tex_path = 'tex_floor.obj'#f'{opt.data_path}/tex_floor.obj'
    bed_obj_path = f'{opt.bed_path}/normalized_model_ro.obj'
    couch_obj_path = f'{opt.couch_path}/normalized_model_ro.obj'
    table_obj_path = f'{opt.table_path}/normalized_model_ro.obj'
    chair_obj_path = f'{opt.chair_path}/normalized_model_ro.obj'


    i = 3
    mesh, matrix = show_all(couch_obj_path, bed_obj_path, floor_tex_path, table_obj_path, chair_obj_path, num=i)
    custom_draw_geometry_with_custom_fov(mesh, -30.0, i)

