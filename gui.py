import pickle
import sys
from PyQt5.QtWidgets import QAction
from vtkmodules.vtkRenderingCore import vtkRenderer
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import os
from model import vtkModel
from utils import logger, helpInformation


class MainWindow(QtWidgets.QMainWindow):
    """
    主窗口
    """

    def __init__(self):
        super(MainWindow, self).__init__()
        # 设置窗口布局
        self.setup_window()
        # 初始化 vtk 模型
        self.init_vtk()
        # 展示
        self.show_all()

    def show_all(self):
        """
        展示界面
        :return:
        """
        self.iren.Initialize()
        self.iren.Start()
        self.show()

    def init_vtk(self):
        """
        初始化 vtk 模型
        :return:
        """
        self.logger2.info("Initialize vtk model...")
        # 初始化 renderer，window，并连接
        self.ren = vtkRenderer()
        self.renWin = self.vtkWidget.GetRenderWindow()
        self.renWin.AddRenderer(self.ren)
        self.iren = self.vtkWidget
        # 初始化 vtk 模型
        self.model = vtkModel(ren=self.ren, iren=self.iren, renWin=self.renWin
                              , log=self.logger2)

    def setup_window(self):
        """
        设置主窗口
        :return:
        """
        # 设置窗口大小
        self.resize(1400, 800)
        # 主窗口
        self.centralWidget = QtWidgets.QWidget(self)
        # 主窗口中的主 layout
        self.MainLayout = QtWidgets.QHBoxLayout(self.centralWidget)
        self.setCentralWidget(self.centralWidget)

        # vtk 对应 layout
        self.vtkLayout = QtWidgets.QHBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor()
        # 设置大小
        self.vtkWidget.setFixedSize(800, 800)

        # 控制按钮对应布局，先上下后左右
        self.inputlayout = QtWidgets.QVBoxLayout()
        self.inputlayout1 = QtWidgets.QHBoxLayout()
        self.inputlayout2 = QtWidgets.QHBoxLayout()

        # 阈值标签及滑块
        self.ThreshholdLabel = QtWidgets.QLabel("Threshold")
        self.ThresholdSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.ThresholdSlider.setRange(0, 1300)
        self.ThresholdSlider.setSingleStep(50)
        self.ThresholdSlider.setValue(200)
        self.ThresholdSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.ThresholdSlider.setTickInterval(50)
        self.ThresholdSlider.valueChanged.connect(self.SliderValueChanged)
        self.ThresholdValueMin = QtWidgets.QLabel("0")
        self.ThresholdValueMax = QtWidgets.QLabel("1300")

        # 清晰度标签及滑块
        self.ResolutionLabel = QtWidgets.QLabel("Resolution")
        self.ResolutionSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.ResolutionSlider.setRange(1, 21)
        self.ResolutionSlider.setSingleStep(2)
        self.ResolutionSlider.setValue(10)
        self.ResolutionSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.ResolutionSlider.setTickInterval(2)
        self.ResolutionSlider.valueChanged.connect(self.SliderValueChanged)
        self.ResolutionValueMin = QtWidgets.QLabel("1")
        self.ResolutionValueMax = QtWidgets.QLabel("21")

        # 日志窗口
        self.TextBrowser1 = QtWidgets.QTextBrowser()
        self.TextBrowser1.setText(helpInformation)
        self.TextBrowser2 = QtWidgets.QTextBrowser()
        # logger用于同时在 terminal 和 窗口中输出
        self.logger2 = logger(self.TextBrowser2)

        # 将以上布局分层次装入主窗口
        self.inputlayout1.addWidget(self.ThreshholdLabel)
        self.inputlayout1.addWidget(self.ThresholdValueMin)
        self.inputlayout1.addWidget(self.ThresholdSlider)
        self.inputlayout1.addWidget(self.ThresholdValueMax)
        self.inputlayout2.addWidget(self.ResolutionLabel)
        self.inputlayout2.addWidget(self.ResolutionValueMin)
        self.inputlayout2.addWidget(self.ResolutionSlider)
        self.inputlayout2.addWidget(self.ResolutionValueMax)
        self.inputlayout.addWidget(self.TextBrowser1)
        self.inputlayout.addLayout(self.inputlayout1)
        self.inputlayout.addLayout(self.inputlayout2)
        self.inputlayout.addWidget(self.TextBrowser2)
        self.MainLayout.addWidget(self.vtkWidget)
        self.MainLayout.addLayout(self.inputlayout)

        # 定义菜单栏
        self.menuBar().setNativeMenuBar(False)
        self.loadMenu = self.menuBar().addMenu('File')
        self.demoMenu = self.menuBar().addMenu('Demo')
        self.operationMenu = self.menuBar().addMenu('Operation')
        self.resetMenu = self.menuBar().addMenu('Reset')
        self.helpMenu = self.menuBar().addMenu('Help')
        self.loadMenu.addAction(QAction('Load', self, triggered=self.loadfile))
        self.demoMenu.addAction(
            QAction('MC varying threshold', self, triggered=lambda: self.Demothreshold(smooth=False, tetras=False)))
        self.demoMenu.addAction(
            QAction('MC varying cubesize', self, triggered=lambda: self.DemoCubesize(smooth=False, tetras=False)))
        self.demoMenu.addAction(QAction('MCSmooth varying threshold', self,
                                        triggered=lambda: self.Demothreshold(smooth=True, tetras=False)))
        self.demoMenu.addAction(
            QAction('MCSmooth varying cubesize', self, triggered=lambda: self.DemoCubesize(smooth=True, tetras=False)))
        self.demoMenu.addAction(
            QAction('MT varying threshold', self, triggered=lambda: self.Demothreshold(smooth=False, tetras=True)))
        self.demoMenu.addAction(
            QAction('MT varying cubesize', self, triggered=lambda: self.DemoCubesize(smooth=False, tetras=True)))
        self.demoMenu.addAction(
            QAction('MTSmooth varying threshold', self, triggered=lambda: self.Demothreshold(smooth=True, tetras=True)))
        self.demoMenu.addAction(
            QAction('MTSmooth varying cubesize', self, triggered=lambda: self.DemoCubesize(smooth=True, tetras=True)))
        self.operationMenu.addAction(QAction('MarhingCubesSmooth', self, triggered=self.MCSmooth))
        self.operationMenu.addAction(QAction('MarchingCubesUnSmooth', self, triggered=self.MCUnSmooth))
        self.operationMenu.addAction(QAction('MarhingTetrasSmooth', self, triggered=self.MTSmooth))
        self.operationMenu.addAction(QAction('MarchingTetrasUnSmooth', self, triggered=self.MTUnSmooth))
        self.operationMenu.addAction(QAction('Rotate', self, triggered=self.Rotate))
        self.resetMenu.addAction(QAction('Reset model', self, triggered=self.Reset))
        self.resetMenu.addAction(QAction('clear output', self, triggered=self.clearText))
        self.helpMenu.addAction(QAction('help', self, triggered=self.help))

    def MCSmooth(self):
        """
        对 MarchingCubes 模型进行 Smooth 操作
        :return:
        """
        self.logger2.info('MarchingCubes Smoothing...')
        self.model.smooth = True
        self.model.MarchingCubes(threshold=self.ThresholdSlider.value(), min_cube_size=self.ResolutionSlider.value(),
                                 smooth=self.model.smooth)
        self.model.update_polydata()

    def MCUnSmooth(self):
        """
        undo MarchingCubesSmooth 操作
        :return:
        """
        self.logger2.info('MarchingCubes UnSmoothing...')
        self.model.smooth = False
        self.model.MarchingCubes(threshold=self.ThresholdSlider.value(), min_cube_size=self.ResolutionSlider.value(),
                                 smooth=self.model.smooth)
        self.model.update_polydata()

    def MTSmooth(self):
        """
        对 MarchingTetras 模型进行 Smooth 操作
        :return:
        """
        self.logger2.info('MarchingTetras Smoothing...')
        self.model.smooth = True
        self.model.MarchingTetras(threshold=self.ThresholdSlider.value(), min_cube_size=self.ResolutionSlider.value(),
                                  smooth=self.model.smooth)
        self.model.update_polydata()

    def MTUnSmooth(self):
        """
        undo MarchingTetrasSmooth 操作
        :return:
        """
        self.logger2.info('MarchingTetras UnSmoothing...')
        self.model.smooth = False
        self.model.MarchingTetras(threshold=self.ThresholdSlider.value(), min_cube_size=self.ResolutionSlider.value(),
                                  smooth=self.model.smooth)
        self.model.update_polydata()

    def help(self):
        """
        输出帮助信息
        :return:
        """
        self.logger2.info('************* Help ****************')
        self.TextBrowser2.append(helpInformation)

    def clearText(self):
        """
        清空输出日志
        :return:
        """
        self.TextBrowser2.clear()

    def SliderValueChanged(self, value):
        """
        当滑块值改变时重新计算 marchingcubes
        :param value:
        :return:
        """
        self.logger2.info('changing to threshold={}, resolution={}'.format(self.ThresholdSlider.value(),
                                                                           self.ResolutionSlider.value()))
        self.model.MarchingCubes(threshold=self.ThresholdSlider.value(), min_cube_size=self.ResolutionSlider.value(),
                                 smooth=self.model.smooth)
        self.model.update_polydata()

    def Reset(self):
        """
        重新计算，将模型恢复到默认状态
        :return:
        """
        self.logger2.info('************* Reset model ****************')
        self.logger2.info("Reset algorithm to default parameters!")
        self.model.MarchingCubes()
        self.model.update_polydata()
        self.show_all()

    def loadfile(self):
        """
        导入 nii 文件
        :return:
        """
        filename, ok = QtWidgets.QFileDialog.getOpenFileName(self, 'Load .nii', '')
        if ok:
            # 若与之前的文件名不同，则读入
            if self.model.filename != filename:
                self.model.filename = filename
                self.model.ReadNii()

            self.thresholdRange = (max(self.model.thresholdBound[0], 0), self.model.thresholdBound[1])
            # 初始化模型参数
            threshold = self.thresholdRange[0] + (self.thresholdRange[1] - self.thresholdRange[0]) / 5
            min_cube_size = max(1, min(self.model.dims) // 10)
            # 根据数据属性重新设置滑块参数
            self.ThresholdSlider.setRange(*self.thresholdRange)
            self.ThresholdSlider.setValue(int(sum(self.thresholdRange) / 2))
            self.ThresholdSlider.setSingleStep(max(1, (self.thresholdRange[1] - self.thresholdRange[0]) // 20))
            self.ThresholdValueMin.setText("%d" % self.thresholdRange[0])
            self.ThresholdValueMax.setText("%d" % self.thresholdRange[1])
            self.ResolutionSlider.setValue(min_cube_size)
            # 重新计算 marchingcubes
            self.model.MarchingCubes(threshold=threshold, min_cube_size=min_cube_size)
            self.model.update_polydata()
            self.show_all()

    def Demothreshold(self, smooth=False, tetras=False):
        """
        demo for varying threshold， 展示等值面阈值从小变大的动态过程
        :return:
        """
        self.logger2.info('********* Display DemoMCthreshold **********')
        # 判断 path
        if tetras:
            path = 'demoMTthreshold'
        else:
            path = 'demoMCthreshold'
        if smooth:
            path += 'Smooth'
        path += '.pkl'
        step = 50
        # 需要事先运行 MarchingCubes.py,MarchingTetras.py 以生成缓存文件
        # 载入已有缓存文件
        locations_list = pickle.load(open(path, 'rb'))
        # 依次渲染展示动态结果
        for t in range(0, 1400, step):
            self.logger2.info('Threshold = {}'.format(t))
            i = t // step
            # 重新设置点集，面集
            self.model.get_all_points(locations_list[i])
            self.model.get_cells(locations_list[i])
            self.model.update_polydata()
            self.renWin.Render()

    def DemoCubesize(self, smooth=False, tetras=False):
        """
        demo for varying cube size, 展示由小到大的 cube 计算出的等值面
        :return:
        """
        self.logger2.info('********* Display DemoMCsize **********')
        if tetras:
            path = 'demoMTthreshold'
        else:
            path = 'demoMCthreshold'
        if smooth:
            path += 'Smooth'
        path += '.pkl'
        # 需要事先运行 MarchingCubes.py,MarchingTetras.py 以生成缓存文件
        # 载入已有缓存文件
        if os.path.exists(path):
            locations_list = pickle.load(open(path, 'rb'))
        else:
            locations_list = []
            for i in range(1, 20):
                locations = self.model.MarchingCubes(threshold=200, min_cube_size=i, smooth=self.model.smooth)
                locations_list.append(locations)
            pickle.dump(locations_list, open(path, 'wb'))
        # 依次渲染
        for s in range(1, 10):
            self.logger2.info('Min_cube_size = {}'.format(s))
            i = s - 1
            # 重新设置点集，面集
            self.model.get_all_points(locations_list[i])
            self.model.get_cells(locations_list[i])
            self.model.update_polydata()
            self.renWin.Render()

    def Rotate(self):
        """
        旋转360度
        :return:
        """
        self.logger2.info('********* Rotate 360 **********')
        for i in range(360):
            self.renWin.Render()
            self.ren.GetActiveCamera().Azimuth(1)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
