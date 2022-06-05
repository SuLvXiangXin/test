import numpy as np
import nibabel as nib
from utils import interpolate_linear, ndmesh, edges2vertices, arr, table, table2, \
    tetra_edges2vertices, array_split, split_data, get_data_cord, smoothing
import pickle


def tetra_Cases(vertices, cords, threshold, center):
    """
    在单个 N 维 tetrahedron 中寻找等值面
    :param vertices: 顶点强度：[n,]
    :param cords: 顶点坐标：[D, n]
    :param threshold: 阈值
    :return: 三维时：1个或多个等值面三角形顶点坐标 [[[x11, y11, z11], [x12, y12, z12], [x13, y13, z13]], ...]
    """
    # 数据维度
    dim = 3
    # 计算各顶点关于 threshold 的相对位置，转化为二进制编码
    state = (vertices > threshold).astype(int)

    # 二进制转十进制
    state = (state * arr[:len(state)]).sum()

    # 查表得到连接点所在边
    edges = table2[dim][state]

    # 初始化找到的等值面列表
    locations = []
    for i in range(0, len(edges), dim):
        # 由每条边查表可知其顶点，通过两顶点插值计算 threshold 坐标位置
        locations.append([interpolate_linear(vertices[tetra_edges2vertices[edges[i + j]]],
                                             cords[:, tetra_edges2vertices[edges[i + j]]], threshold, center) for j in
                          range(dim)])
        locations[-1].append((255, 100, 100))
    return locations


def find_tetra3d(data, cord_matrix, threshold, center):
    """
    在给定单个三维 cube 中分割出 tetrahedron 并寻找等值面
    :param data: 强度信息 [Nx, Ny, Nz]
    :param cord_matrix: 各点坐标 [3, Nx, Ny, Nz]
    :param threshold: 强度阈值
    :return: 1个或多个等值面三角形顶点坐标 [[[x11, y11, z11], [x12, y12, z12], [x13, y13, z13]], ...]
    """
    # 按顺序取 6 个tetrahedrons 中各顶点的值和坐标
    vertices = data[[0, 0, 0, 0, -1, -1, -1, -1], [0, -1, -1, 0, 0, -1, -1, 0], [0, 0, -1, -1, 0, 0, -1, -1]]
    cords = cord_matrix[:, [0, 0, 0, 0, -1, -1, -1, -1], [0, -1, -1, 0, 0, -1, -1, 0], [0, 0, -1, -1, 0, 0, -1, -1]]

    tetra_vertices = [vertices[[0, 1, 3, 7]], vertices[[0, 1, 4, 7]], vertices[[1, 4, 5, 7]], vertices[[1, 5, 6, 7]],
                      vertices[[1, 2, 6, 7]], vertices[[1, 2, 3, 7]]]
    tetra_cords = [cords[:, [0, 1, 3, 7]], cords[:, [0, 1, 4, 7]], cords[:, [1, 4, 5, 7]], cords[:, [1, 5, 6, 7]],
                   cords[:, [1, 2, 6, 7]], cords[:, [1, 2, 3, 7]]]
    # 调用函数在每个tetrahedron中得到等值面
    locations = []
    for i in range(6):
        # if tetra_vertices[i].max() <= threshold or tetra_vertices[i].min() > threshold:
        #     continue
        locations.extend(tetra_Cases(tetra_vertices[i], tetra_cords[i], threshold, center))
    return locations


def find_locations_in_tetra(data, cord_matrix, threshold, center):
    """
    分不同维度在 tetrahedron 中寻找等值面
    :param data: N-D numpy.ndarray
    :param cord_matrix: 坐标矩阵 (N+1)-D numpy.ndarray [3,Nx,Ny,Nz]
    :param threshold: 阈值
    :return: Ex: 当data为3维时， 1个或多个等值面三角形顶点坐标 [[[x11, y11, z11], [x12, y12, z12], [x13, y13, z13]], ...]
    """
    # 数据维度
    dim = len(data.shape)
    if dim == 3:
        return find_tetra3d(data, cord_matrix, threshold, center)
    else:
        raise NotImplementedError


def MarchingTetras(data, cord_matrix=None, threshold=200., min_cube_size=5, divide=2, center=None):
    """
    Implements "Marching Tetras"
    由八爪树算法寻找等值面
    :param data: N-D numpy.ndarray
    :param cord_matrix: 坐标矩阵 (N+1)-D numpy.ndarray [3,Nx,Ny,Nz]
    :param threshold: 阈值
    :param min_cube_size:  一个 cube 最小像素级尺度(八爪树切分的最高分辨率)
    :param divide: 每次沿各方向切分次数
    :return: Ex: 当data为3维时， 等值面三角形顶点坐标 [[[x11, y11, z11], [x12, y12, z12], [x13, y13, z13]], ...]
    """
    # 初始化当前 cube 的等值面列表
    locations_all = []

    # 数据维度大小
    size = np.array(data.shape)

    # 所有值均大于或小于 threshold， 说明此 cube 完全在等值面外或内，直接跳过
    if data.max() <= threshold or data.min() > threshold:
        return []
    if cord_matrix is None:
        # 获得坐标矩阵，cord_matrix[:,x,y,z]= [x,y,z], 便于拆分后记住坐标
        cord_matrix = np.array(ndmesh(*[np.arange(s) for s in size]))
    if center is None:
        # 获取中心点
        center = size / 2

    # 达到最高清晰度时停止递归
    if min(size) / divide <= min_cube_size:
        # 网格坐标上限，即各方向需要切分的次数+1
        dims = np.ceil(size / (min_cube_size + 1)).astype(int)

        # 初始化正在处理的 cube 在切分网格中坐标
        cord = np.zeros_like(dims)

        # 由 numpy.ndarray 转换为列表
        steps = dims.tolist()

        # 切分数据
        data = split_data(data, steps)
        # 按相同方式切分坐标矩阵，使坐标矩阵与数据保持同步
        cord_matrix = split_data(cord_matrix, [0] + steps, dim=1)  # 由于第一维是坐标，不做切分
        while (cord < dims).all():  # 循环直到处理到网格坐标上限
            # 获取当前需要处理的数据以及坐标矩阵
            data_ = get_data_cord(data, cord)
            cord_matrix_ = get_data_cord(cord_matrix, cord)

            # 在此 cube 中寻找等值面
            locations_found = find_locations_in_tetra(data_, cord_matrix_, threshold, center)

            # 若找到，加入等值面三角形列表
            if locations_found:
                locations_all.extend(locations_found)

            # 更新当前 cube 的网格坐标
            for i in range(len(size) - 1, -1, -1):
                if cord[i] + 1 < dims[i]:
                    cord[i] += 1
                    break
                elif i - 1 < 0:
                    # 达到最高位
                    cord[i] += 1
                    break
                else:
                    # 进位
                    cord[i] = 0
                    cord[i - 1] += 1
                    if cord[i - 1] < dims[i - 1]:
                        break
        return locations_all

    # 八爪树算法切分数据

    # 网格坐标上限，即各方向需要切分的次数+1
    divide_dims = np.array([divide] * len(size))

    # 初始化正在处理的 cube 在切分网格中坐标
    divide_cord = np.zeros_like(divide_dims)

    # 各维度切分次数
    steps = [divide for i in range(len(size))]

    # 切分数据以及坐标矩阵
    data = split_data(data, steps)
    cord_matrix = split_data(cord_matrix, [0] + steps, dim=1)  # 由于第一维是坐标，不做切分

    while (divide_cord < divide_dims).all():  # 循环直到处理到网格坐标上限
        # 获取当前需要处理的数据以及坐标矩阵
        data_ = get_data_cord(data, divide_cord)
        cord_matrix_ = get_data_cord(cord_matrix, divide_cord)

        # 递归调用 MarchingTetra 来处理当前 cube 寻找等值面
        new_location = MarchingTetras(data_, cord_matrix_, threshold, min_cube_size, divide, center)

        # 若找到，加入等值面三角形列表
        if new_location:
            locations_all.extend(new_location)

        # 更新当前 cube 的网格坐标
        for i in range(len(size) - 1, -1, -1):
            if divide_cord[i] + 1 < divide_dims[i]:
                divide_cord[i] += 1
                break
            elif i - 1 < 0:
                # 达到最高位
                divide_cord[i] += 1
                break
            else:
                # 进位
                divide_cord[i] = 0
                divide_cord[i - 1] += 1
                if divide_cord[i - 1] < divide_dims[i - 1]:
                    break
    return locations_all





def MarchingTetrasSmooth(data, cord_matrix=None, threshold=200., min_cube_size=5, divide=2, center=None):
    """
    MarchingTetras with smoothing
    """
    locations_all = MarchingTetras(data, cord_matrix, threshold, min_cube_size, divide, center)
    locations_all = smoothing(locations_all, 0.33)
    locations_all = smoothing(locations_all, -0.34)
    return locations_all


if __name__ == '__main__':
    # ###### 1D
    # img_data = np.array([2, 4.5, 6, 7, 8, 4, 8, 7, 6, 4.5, 2])
    # locations_all = MarchingCube(img_data, threshold=5, min_cube_size=2)
    # print(locations_all)
    # ######

    # ###### 2D
    # img_data = np.abs(np.arange(5) - 2).reshape(1, -1) * np.abs(np.arange(5) - 2).reshape(-1, 1)  # 从左向右波浪形
    # # img_data = np.array([list(range(5)) for i in range(5)])  # 从内向外漩涡形
    # locations_all = MarchingCube(img_data, threshold=1.5, min_cube_size=1)
    # print(locations_all)
    ######

    # ###### 3D
    img = nib.load('image_lr.nii.gz')
    img_data = img.get_fdata()
    # locations_all = MarchingTetrasSmooth(img_data, threshold=200, min_cube_size=10)
    # print(locations_all)
    ######
    # # for demo
    path = 'demoMTthreshold.pkl'
    step = 50
    locations_list = []
    for i in range(0, 1400, step):
        locations = MarchingTetras(img_data, threshold=i, min_cube_size=1)
        locations_list.append(locations)
    pickle.dump(locations_list, open(path, 'wb'))
    print('demoMTthreshold done!')

    path = 'demoMTsize.pkl'
    locations_list = []
    for i in range(1, 20):
        locations = MarchingTetras(img_data, threshold=200, min_cube_size=i)
        locations_list.append(locations)
    pickle.dump(locations_list, open(path, 'wb'))
    print('demoMTsize done!')

    path = 'demoMTthresholdSmooth.pkl'
    step = 50
    locations_list = []
    for i in range(0, 1400, step):
        locations = MarchingTetrasSmooth(img_data, threshold=i, min_cube_size=1)
        locations_list.append(locations)
    pickle.dump(locations_list, open(path, 'wb'))
    print('demoMCthresholdSmooth done!')

    path = 'demoMTsizeSmooth.pkl'
    locations_list = []
    for i in range(1, 20):
        locations = MarchingTetrasSmooth(img_data, threshold=200, min_cube_size=i)
        locations_list.append(locations)
    pickle.dump(locations_list, open(path, 'wb'))
    print('demoMCsizeSmooth done!')
