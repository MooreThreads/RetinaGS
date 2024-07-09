
import os
import torch
import torch.distributed as dist
import math
import numba
import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull, Delaunay
from abc import ABC, abstractmethod
from scene.cameras import Camera
from typing import NamedTuple, Dict, List
import pickle
import logging

class SpaceBox:
    """ 
        A box in continuous 3d space
        positive/negative directions of axes are not defined
        .. code-block:: none

                                              z axis
                                 x axis           
                                 /            |
                                /             |
                  (x1, y0, z1) + -----------  + (x1, y1, z1)
                              /|            / |
                             / |           /  |
               (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                            |  /          |  /
                            | /           | /
             y axis ------- + ----------- + (x0, y1, z0)
                (x0, y0, z0)
        """
    def __init__(self, range_low, range_up) -> None:
        assert len(range_low)==3 and len(range_up)==3
        x0, y0, z0 = range_low
        x1, y1, z1 = range_up
        self.vertexs = np.array([
            (x0, y0, z0),
            (x0, y0, z1), 
            (x0, y1, z1), 
            (x0, y1, z0), 
            (x1, y0, z0), 
            (x1, y0, z1), 
            (x1, y1, z1), 
            (x1, y1, z0)], dtype=np.float32)
        self.edges_start = np.array([
            (x0, y0, z0),
            (x0, y0, z1),
            (x0, y1, z0),
            (x0, y1, z1),   # parallel with x axis
            (x0, y0, z0),
            (x0, y0, z1),
            (x1, y0, z0),
            (x1, y0, z1),   # parallel with y axis
            (x0, y0, z0),
            (x0, y1, z0),
            (x1, y0, z0),
            (x1, y1, z0),   # parallel with z axis
        ], dtype=np.float32)
        self.edges_end = np.array([
            (x1, y0, z0),
            (x1, y0, z1),
            (x1, y1, z0),
            (x1, y1, z1),   # parallel with x axis
            (x0, y1, z0),
            (x0, y1, z1),
            (x1, y1, z0),
            (x1, y1, z1),   # parallel with y axis
            (x0, y0, z1),
            (x0, y1, z1),
            (x1, y0, z1),
            (x1, y1, z1),   # parallel with z axis
        ], dtype=np.float32)
        self.range_low = np.array(range_low, dtype=np.float32)
        self.range_up = np.array(range_up, dtype=np.float32)
        self.center = (self.range_low + self.range_up)/2


class ViewFrustum:
    """
    A view frustum in continuous 3d space
    """
    def __init__(self, msg: Camera, z_near=None, z_far=None) -> None:
        tanHalfFovY = math.tan((msg.FoVy / 2))
        tanHalfFovX = math.tan((msg.FoVx / 2))
        self.z_near = max(0.01, 0.01 if z_near is None else z_near)
        self.z_far = min(1e6, 1e6 if z_far is None else z_far)
        # Stacked Inequalities of the form Ax + b <= 0 in format [A; b]
        halfspaces = [
            (1, 0, -tanHalfFovX, 0),    # x <= z*tanHalfFovX
            (-1,0, -tanHalfFovX, 0),    # x >= z*-tanHalfFovX      
            (0, 1, -tanHalfFovY, 0),    # y <= z*tanHalfFovY
            (0,-1, -tanHalfFovY, 0),    # y >= z*-tanHalfFovY 
            (-1, 0, 0, self.z_near),    # x >= z_near  
            (1, 0, 0, -self.z_far),     # x <= z_far
        ]

        self.halfspaces = np.array(halfspaces, dtype=np.float32)
        vertexs_near = np.array([
            (tanHalfFovX, tanHalfFovY, 1),
            (tanHalfFovX, -tanHalfFovY, 1),
            (-tanHalfFovX, tanHalfFovY, 1),
            (-tanHalfFovX, -tanHalfFovY, 1),
        ], dtype=np.float32) * self.z_near
        vertexs_far = np.array([
            (tanHalfFovX, tanHalfFovY, 1),
            (tanHalfFovX, -tanHalfFovY, 1),
            (-tanHalfFovX, tanHalfFovY, 1),
            (-tanHalfFovX, -tanHalfFovY, 1),
        ], dtype=np.float32) * self.z_far
        self.vertexs = np.concatenate((vertexs_near, vertexs_far), axis=0)


def is_overlapping_SpaceBox_View(s:SpaceBox, v:Camera, z_far=1e6, z_near=0.01):
    vf = ViewFrustum(v, z_far=z_far, z_near=z_near)

    box_vertexs = np.concatenate((s.vertexs, np.ones((8, 1), dtype=np.float32)), axis=1)
    box_vertexs = np.dot(box_vertexs, v.world_view_transform.cpu().numpy())[:, :3]
    
    Minkowski_diff = np.reshape(box_vertexs, (8,1,3)) - np.reshape(vf.vertexs, (1,8,3))
    # print(Minkowski_diff.shape)
    Minkowski_diff = np.reshape(Minkowski_diff, (-1, 3))

    _hull = ConvexHull(Minkowski_diff)
    # print(_vertexs)
    # _vertexs = _hull.points[ConvexHull(Minkowski_diff).vertices, :]
    # tri = Delaunay(_vertexs)
    # find = tri.find_simplex(np.array([(0,0,0),], dtype=np.float32))
    # print(find)

    distances = np.dot(_hull.equations, np.array((0, 0, 0, 1.0)).reshape((4,1)))
    in_hull = np.all(distances <= 0)

    # assert in_hull == (find[0] >= 0)

    return in_hull


@numba.jit(nopython=True)
def accum_on_3Dgrid(grid: np.ndarray, value:np.ndarray, index:np.ndarray):
    """
    scatter points into a 3d array
    pytorch index_add can not guarantee a correct result
    """
    x_max, y_max, z_max = grid.shape
    for _i, v in enumerate(value):
        x, y, z = index[_i]
        if (0 <= x < x_max) and (0 <= y < y_max) and (0 <= z < z_max):
            grid[x, y, z] += v
    return grid        


class Grid3DSpace:
    """
    A 3d grid representing equally divided space
    """
    def __init__(self, range_low: list, range_up: list, voxel_size: list) -> None:
        assert len(range_low) == len(range_up) == len(voxel_size) == 3
        self.range_low = np.array(range_low)
        self.range_up = np.array(range_up)
        self.voxel_size = np.array(voxel_size)
        self.grid_size = tuple(
            max(int((self.range_up[i]-self.range_low[i])//self.voxel_size[i]), 1) for i in range(3)
        )
        self.load_cnt:np.ndarray = np.zeros(self.grid_size, dtype=float)

        self.range_low_gpu = torch.tensor(self.range_low, device='cuda', dtype=torch.float)
        self.voxel_size_gpu = torch.tensor(self.voxel_size, device='cuda', dtype=torch.float)
    
    def accum_load(self, load_np:np.ndarray, position:np.ndarray):
        position_int = ((position - self.range_low) // self.voxel_size).astype(int)
        accum_on_3Dgrid(grid=self.load_cnt, value=load_np, index=position_int)
        return self.load_cnt
    
    def accum_load_grid(self, load_np:np.ndarray):
        self.load_cnt += load_np

    def clean_load(self):
        self.load_cnt *= 0


class BoxinGrid3D:
    def __init__(self, range_low:list, range_up:list) -> None:
        assert len(range_low) == len(range_up) == 3
        self.range_low = np.array(tuple(int(v) for v in range_low))   # tuple of int
        self.range_up = np.array(tuple(int(v) for v in range_up))     # tuple of int
        assert (0 <= self.range_low[0] < self.range_up[0]) and (0 <= self.range_low[1] < self.range_up[1]) and (0 <= self.range_low[2] < self.range_up[2])

    def __str__(self) -> str:
        return f'BoxinGrid3D(low={self.range_low}, up={self.range_up})'


def overlap_BoxinGrid3D(box1:BoxinGrid3D, box2:BoxinGrid3D):
    max_x_low = max(box1.range_low[0], box2.range_low[0]) 
    max_y_low = max(box1.range_low[1], box2.range_low[1]) 
    max_z_low = max(box1.range_low[2], box2.range_low[2]) 

    min_x_up = min(box1.range_up[0], box2.range_up[0])
    min_y_up = min(box1.range_up[1], box2.range_up[1]) 
    min_z_up = min(box1.range_up[2], box2.range_up[2])  

    if (min_x_up > max_x_low) and (min_y_up > max_y_low) and (min_z_up > max_z_low):
        return BoxinGrid3D(
            range_low=(max_x_low, max_y_low, max_z_low), range_up=(min_x_up, min_y_up, min_z_up)
        )
    else:
        return None


class BvhTreeNodeon3DGrid:
    """
    BvhTreeNode on 3D Grid but not 3d space 
    """
    def __init__(self, grid:Grid3DSpace, range_low:list, range_up:list, level:int, path:str, split_orders:list=[0,1,2]) -> None:
        assert (0<=range_low[0]<range_up[0]) and (0<=range_low[1]<range_up[1]) and (0<=range_low[2]<range_up[2])
        self.grid = grid
        self.range_low = tuple(int(v) for v in range_low)   # tuple of int
        self.range_up = tuple(int(v) for v in range_up)     # tuple of int
        self.level:int = level
        self.path:str = path    # '01' string encoding the path from root, '' for root
        self.split_orders:list = split_orders
        self.split_dim:int = self.split_orders[self.level % len(self.split_orders)]   
        self.split_position_grid:int = None 
        self.left_child:BvhTreeNodeon3DGrid = None
        self.right_child:BvhTreeNodeon3DGrid = None

    def split(self, limit:int=1, invalid_split_postion:int=None, random_radius:int=10, logger:logging.Logger=None):
        if (self.range_up[self.split_dim] - self.range_low[self.split_dim]) < 2:
            return []
        # meaningless to search best split with binary-search
        # sum along split axis is of already O(N) complexity
        l0, l1, l2 = self.range_low
        u0, u1, u2 = self.range_up
        sum_axis = tuple(i for i in range(3) if i!=self.split_dim)
        load_array = self.grid.load_cnt[l0:u0, l1:u1, l2:u2].sum(axis=sum_axis, keepdims=False)
        all_load = np.sum(load_array)
        load_cum = np.cumsum(load_array)
        load_inbalance = np.absolute(2*load_cum - all_load)
        split_point = np.argmin(load_inbalance)

        pre_split_position_grid = self.range_low[self.split_dim] + split_point + 1
        if pre_split_position_grid == invalid_split_postion:
            offset, random_radius = 0, max(random_radius, 1)
            while offset == 0:
                offset = np.random.randint(-random_radius, random_radius+1)
            split_point += offset    
            if logger is not None:
                logger.info('solve invalid_split_postion with offset {}'.format(offset))    

        limit = np.clip(limit, 0, len(load_inbalance)//2)
        split_point = np.clip(split_point, limit, len(load_inbalance)-1-limit)

        self.split_position_grid = self.range_low[self.split_dim] + split_point + 1 

        left_range_low, left_range_up = [v for v in self.range_low], [v for v in self.range_up]
        left_range_up[self.split_dim] = self.range_low[self.split_dim] + split_point + 1
        self.left_child = BvhTreeNodeon3DGrid(grid=self.grid, range_low=left_range_low, range_up=left_range_up, level=self.level+1, path=self.path+'0', split_orders=self.split_orders)
        
        right_range_low, right_range_up = [v for v in self.range_low], [v for v in self.range_up]
        right_range_low[self.split_dim] = self.range_low[self.split_dim] + split_point + 1
        self.right_child = BvhTreeNodeon3DGrid(grid=self.grid, range_low=right_range_low, range_up=right_range_up, level=self.level+1, path=self.path+'1', split_orders=self.split_orders)
        return [self.left_child, self.right_child]

    def in_right(self, position3D):
        # False for left, True for right
        if self.split_position_grid is None:
            raise RuntimeError

        pos_np = np.array(position3D).reshape(-1)[:3]
        split_world = self.split_position_grid * self.grid.voxel_size[self.split_dim] + self.grid.range_low[self.split_dim]
        return pos_np[self.split_dim] > split_world

    def get_split_position_in_world(self):
        if self.split_position_grid is None:
            raise RuntimeError
        split_world = self.split_position_grid * self.grid.voxel_size[self.split_dim] + self.grid.range_low[self.split_dim]
        return split_world

    def overlap(self, box:BoxinGrid3D):
        # overlap 
        max_x_low = max(self.range_low[0], box.range_low[0]) 
        max_y_low = max(self.range_low[1], box.range_low[1]) 
        max_z_low = max(self.range_low[2], box.range_low[2]) 

        min_x_up = min(self.range_up[0], box.range_up[0])
        min_y_up = min(self.range_up[1], box.range_up[1]) 
        min_z_up = min(self.range_up[2], box.range_up[2])  

        if (min_x_up > max_x_low) and (min_y_up > max_y_low) and (min_z_up > max_z_low):
            return BoxinGrid3D(
                range_low=(max_x_low, max_y_low, max_z_low), range_up=(min_x_up, min_y_up, min_z_up)
            )
        else:
            return None

    @property
    def all_load(self):
        l0, l1, l2 = self.range_low
        u0, u1, u2 = self.range_up
        return self.grid.load_cnt[l0:u0, l1:u1, l2:u2].sum()
    
    @property
    def is_leaf(self):
        return (self.left_child is None) and (self.right_child is None)

    @property
    def size(self):
        h = self.range_up[0] - self.range_low[0]
        w = self.range_up[1] - self.range_low[1]
        l = self.range_up[2] - self.range_low[2]
        return h*w*l
    
    @property
    def center_in_world(self):
        center_grid = (np.array(self.range_low, dtype=float) + np.array(self.range_up, dtype=float)) /2
        center_world = center_grid * self.grid.voxel_size + self.grid.range_low
        return center_world

    @property
    def range_low_in_world(self):
        return np.array(self.range_low, dtype=float)*self.grid.voxel_size + self.grid.range_low

    @property
    def range_up_in_world(self):
        return np.array(self.range_up, dtype=float)*self.grid.voxel_size + self.grid.range_low

    def __str__(self) -> str:
        return 'BvhTreeNodeon3DGrid(low={}, up={}, level={}, size={}, load={})'.format(
            self.range_low, self.range_up, self.level, self.size, self.all_load
        )


def build_BvhTree_on_3DGrid(grid:Grid3DSpace, max_depth:int, split_orders=[0,1,2], example_path2bvh_nodes={}, logger:logging.Logger=None):
    grid_size = grid.grid_size
    root = BvhTreeNodeon3DGrid(grid=grid, range_low=(0,0,0), range_up=grid_size, level=0, path='', split_orders=split_orders)
    q = [root]
    path2node = {}

    while len(q) > 0:
        node:BvhTreeNodeon3DGrid = q.pop(0)
        if (node is not None) and (node.level < max_depth):
            if node.path in example_path2bvh_nodes:
                example_node:BvhTreeNodeon3DGrid = example_path2bvh_nodes[node.path]
                old_split_postion = example_node.split_position_grid
                if logger is not None:
                    logger.info('find old split at position {}'.format(old_split_postion))
            else:
                old_split_postion = None

            new_nodes = node.split(limit=2**(max_depth - node.level), invalid_split_postion=old_split_postion, logger=logger)
            if logger is not None:
                logger.info('find new split at position {}'.format(node.split_position_grid))
            for _n in new_nodes:
                if _n is not None:
                    q.append(_n)
        if node is not None:
            path2node[node.path] = node 

    return path2node


def save_BvhTree_on_3DGrid(path2node:dict, file_path:str):
    save_dict = {}
    for path, node in path2node.items():
        assert isinstance(node, BvhTreeNodeon3DGrid)
        save_dict[path] = [node.range_low, node.range_up, node.split_orders]
    
    root = path2node['']
    assert isinstance(root, BvhTreeNodeon3DGrid)
    save_dict['grid_info'] = [root.grid.range_low, root.grid.range_up, root.grid.grid_size] 
    with open(file_path, 'wb') as f:
        pickle.dump(save_dict, f)


def load_BvhTree_on_3DGrid(file_path:str):
    with open(file_path, 'rb') as f:
        save_dict:dict = pickle.load(f)

    grid_range_low, grid_range_up, grid_size = save_dict.pop('grid_info')
    path2node_info_dict:Dict[str, List] = save_dict
    return grid_range_low, grid_range_up, grid_size, path2node_info_dict


def build_BvhTree_on_3DGrid_with_info(path2node_info:dict, grid:Grid3DSpace):
    path2node:Dict[str, BvhTreeNodeon3DGrid] = {}
    # create nodes
    for path in path2node_info:
        range_low, range_up, split_orders = path2node_info[path]
        level = len(path)
        path2node[path] = BvhTreeNodeon3DGrid(grid, range_low, range_up, level, path, split_orders)
    # bind nodes
    for path in path2node:
        node = path2node[path]
        if (path+'0') in path2node and (path+'1') in path2node:
            assert isinstance(node, BvhTreeNodeon3DGrid)
            split_dim = node.split_dim
            node.left_child = path2node[path+'0']
            node.right_child = path2node[path+'1']
            node.split_position_grid = node.right_child.range_low[split_dim]
            assert node.left_child.range_up[split_dim] == node.right_child.range_low[split_dim]
    return path2node


def format_BvhTree_on_3DGrid(root:BvhTreeNodeon3DGrid):
    assert root is not None
    root_strs = ['    '*root.level + '|---' + str(root) + '\n']
    l_node, r_node = root.left_child, root.right_child

    l_strs = [] if l_node is None else format_BvhTree_on_3DGrid(l_node)
    r_strs = [] if r_node is None else format_BvhTree_on_3DGrid(r_node)
    return root_strs + l_strs + r_strs


def gather_space_from_BvhTree(block:BoxinGrid3D, root:BvhTreeNodeon3DGrid):
    """
    gather space from leaf-nodes in BvhTree 
    input: BoxinGrid3D, BvhTreeNodeon3DGrid
    output: dict(BvhTreeNodeon3DGrid.path: overlapped BoxinGrid3D)
    """
    # actually, as nodes in tree is just box
    # overlapped BoxinGrid3D of inputs is not necessary
    # is this function necessary? we won't have too many leaf nodes

    q = [root]
    ret = {}
    while len(q) > 0:
        node:BvhTreeNodeon3DGrid = q.pop(0)
        if (node is None):
            continue
        
        _overlap:BoxinGrid3D = node.overlap(block)
        if _overlap is None:
            continue
            
        if node.is_leaf:
            ret[node.path] = _overlap
        else:
            q.append(node.left_child)
            q.append(node.right_child)    

    return ret
