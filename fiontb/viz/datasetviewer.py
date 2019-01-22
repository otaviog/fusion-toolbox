from open3d import PointCloud, draw_geometries


class DatasetViewer:
    def __init__(self, dataset):
        self.dataset = dataset

    def run(self):
        pcl = PointCloud()
        #for idx in range(0, 900, 25):  # len(self.dataset)):
        for idx in [0, 1]:
            snap = self.dataset[idx]
            if True:
                for point, color in zip(snap.cam_points, snap.colors):
                    pcl.points.append(point.squeeze())
                    pcl.colors.append(color)
            print(idx)
        draw_geometries([pcl])

import vtk

class DatasetViewer2:
    def __init__(self, dataset):
        self.dataset = dataset        
        self.vtkPolyData = vtk.vtkPolyData()
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName("DepthArray")
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        # mapper.SetScalarRange(zMin, zMax)
        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)

    def run(self):
        for idx in range(0, 900, 25):  # len(self.dataset)):
            snap = self.dataset[idx]
            print(idx)
            if True:
                for k, (point, color) in enumerate(zip(snap.world_points, snap.colors)):
                    pointId = self.vtkPoints.InsertNextPoint(point[:])
                    self.vtkDepth.InsertNextValue(point[2])
                    self.vtkCells.InsertNextCell(1)
                    self.vtkCells.InsertCellPoint(pointId)
                    self.vtkPoints.SetPoint(k, point)
                    #pcl.points.append(point.squeeze())
                    #pcl.colors.append(color)
                self.vtkCells.Modified()
                self.vtkPoints.Modified()
                self.vtkDepth.Modified()

        renderer = vtk.vtkRenderer()
        renderer.AddActor(self.vtkActor)
        renderer.SetBackground(.2, .3, .4)
        renderer.ResetCamera()

        # Render Window
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)

        # Interactor
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)

        # Begin Interaction
        renderWindow.Render()
        renderWindowInteractor.Start()
        
