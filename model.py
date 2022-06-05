from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import vtkActor, vtkRenderer, vtkRenderWindow, \
    vtkRenderWindowInteractor, vtkCamera
import vtkmodules.all as vtk
import nibabel as nib
from MarchingCubes import MarchingCubes, MarchingCubesSmooth
from MarchingTetra import MarchingTetras, MarchingTetrasSmooth
from utils import logger


class vtkModel(object):
    def __init__(self, ren, iren, renWin, filename='image_lr.nii.gz', log=None):
        """

        :param ren: vtkRender
        :param iren: vtkRenderWindowInteractor
        :param renWin: vtkRenderWindow
        :param filename: 文件名
        :param log: GUI textbrowser
        """
        self.ren = ren
        self.iren = iren

        # set the camera style
        style = vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(style)
        self.renWin = renWin
        self.filename = filename

        # set logger
        if log:
            self.logger = log
        else:
            self.logger = logger()

        # read data
        self.ReadNii()

        # 最开始不设置smooth
        self.smooth = False

        # 调用算法
        self.MarchingCubes()
        # self.MarchingTetras()

        # 渲染
        self.render_from_polydata()

    def ReadNii(self):
        """
        读取 nii 数据
        :return:
        """
        self.logger.info('reading from {}'.format(self.filename))

        # 读入数据
        img = nib.load(self.filename)
        self.img_data = img.get_fdata()

        # 数据的维度，阈值区间
        self.dims = self.img_data.shape
        self.thresholdBound = (int(self.img_data.min()), int(self.img_data.max()))
        self.logger.info("data dimension is [{}]".format(','.join([str(i) for i in self.dims])))
        self.logger.info("intensity range from {} to {}".format(self.thresholdBound[0], self.thresholdBound[1]))

    def MarchingCubes(self, threshold=200, min_cube_size=10, smooth=False):
        """
        计算 marchingcubes
        :param threshold: 阈值
        :param min_cube_size: 最小 cube 大小
        :param smooth: 是否使用 smooth
        :return: 等值面三角形列表
        """
        self.logger.info('computing marchingcubes with threshold={} min_cube_size={}'.format(threshold, min_cube_size))
        if not smooth:
            locations_all = MarchingCubes(self.img_data, threshold=threshold, min_cube_size=min_cube_size)
        else:
            locations_all = MarchingCubesSmooth(self.img_data, threshold=threshold, min_cube_size=min_cube_size)

        # 根据等值面三角形列表设置 vtk渲染所需的点/面
        self.get_all_points(locations_all)
        self.get_cells(locations_all)
        return locations_all

    def MarchingTetras(self, threshold=200, min_cube_size=10, smooth=False):
        """
        计算 marchingtetras
        :param threshold: 阈值
        :param min_cube_size: 最小 cube 大小
        :param smooth: 是否使用 smooth
        :return: 等值面三角形列表
        """
        self.logger.info('computing marchingteras with threshold={} min_cube_size={}'.format(threshold, min_cube_size))
        if not smooth:
            locations_all = MarchingTetras(self.img_data, threshold=threshold, min_cube_size=min_cube_size)
        else:
            locations_all = MarchingTetrasSmooth(self.img_data, threshold=threshold, min_cube_size=min_cube_size)

        # 根据等值面三角形列表设置 vtk渲染所需的点/面
        self.get_all_points(locations_all)
        self.get_cells(locations_all)
        return locations_all

    def update_polydata(self):
        """
        更新 vtkpolydata
        :return:
        """
        # 设置点集
        self.polygonPolyData.SetPoints(self.points)
        # 设置面集
        self.polygonPolyData.SetPolys(self.cells)
        # 设置颜色
        self.polygonPolyData.GetCellData().SetScalars(self.colors)

    def render_from_polydata(self):
        """
        由 polydata 渲染
        :return:
        """
        self.polygonPolyData = vtk.vtkPolyData()
        self.update_polydata()
        # 调整相机位置位于正上方
        camera = vtkCamera()
        camera.SetPosition(0, 0, self.dims[-1] * 5)
        camera.SetFocalPoint(0, 0, 0)
        self.ren.SetActiveCamera(camera)

        # 删除之前的actor， 避免在更改时 renderer 中有多个actor
        if hasattr(self, 'actor'):
            self.ren.RemoveActor(self.actor)
        # 连接 polydata 与 mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.polygonPolyData)
        mapper.ScalarVisibilityOn()

        # 连接 mapper 与 renderer
        self.actor = vtkActor()
        self.actor.SetMapper(mapper)
        self.ren.AddActor(self.actor)
        # 设置背景颜色
        self.ren.SetBackground(vtkNamedColors().GetColor3d("SlateGray"))

    def get_all_points(self, locations_all):
        """
        由等值面三角形列表获取点集
        :param locations_all: [[[x11,y11,z11], ..., [x1n,y1n,z1n], [R1,G1,B1]],
                            ..., [[xk1,yk1,zk1], ..., [xkn,ykn,zkn], [Rk,Gk,Bk]]]
        :return:
        """
        # 初始化点集
        self.point_id = 0
        self.point_loc2id = {}
        self.points = vtkPoints()
        for locations in locations_all:
            for location in locations[:-1]:
                # 防止 location 为列表，将其转换为 hashable 对象
                if not isinstance(location, tuple):
                    location = tuple(location)
                # 若该点不在点集内，则加入点集
                if location not in self.point_loc2id:
                    self.points.InsertNextPoint(*location)
                    self.point_loc2id[location] = self.point_id
                    self.point_id += 1
        self.logger.info("Get {} points, {} triangle".format(self.point_id, len(locations_all)))

    def get_cells(self, locations_all):
        """
        由等值面三角形列表获得面集
        :param locations_all: [[[x11,y11,z11], ..., [x1n,y1n,z1n], [R1,G1,B1]], ..., [[xk1,yk1,zk1],
                            ..., [xkn,ykn,zkn], [Rk,Gk,Bk]]]
        :return:
        """
        # 初始化面集
        self.cells = vtk.vtkCellArray()
        # 初始化面集的颜色序列
        self.colors = vtk.vtkUnsignedCharArray()
        self.colors.SetNumberOfComponents(3)
        self.colors.SetName('Colors')
        for locations in locations_all:
            # 由一个三角形获得面以及其颜色
            cell, color = self.get_cell(locations)
            # 将面及颜色信息保存
            self.cells.InsertNextCell(cell)
            self.colors.InsertNextTuple3(*color)

    def get_cell(self, locations):
        """
        由一个三角形获得面及颜色
        :param locations: [[x1,y1,z1], [x2,y2,z2], ..., [xn,yn,zn], [R,G,B]]
        :return:
        """
        # 初始化面
        polygon = vtk.vtkPolygon()
        # 设置面所需顶点数
        polygon.GetPointIds().SetNumberOfIds(len(locations) - 1)
        for i, location in enumerate(locations[:-1]):
            # 通过点集中的 id 与之对应
            polygon.GetPointIds().SetId(i, self.point_loc2id[location])
        return polygon, locations[-1]


if __name__ == '__main__':
    # 初始化 renderer，window，windowinteractor，并相互连接
    ren = vtkRenderer()
    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtkRenderWindowInteractor()
    renWin.SetInteractor(iren)
    model = vtkModel(ren=ren, iren=iren, renWin=renWin)
    iren.Initialize()
    iren.Start()
