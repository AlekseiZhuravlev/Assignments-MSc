#!/usr/bin/env python
# This script has been tested with Python 3.7.7 and VTK 8.2.0
# ################################################################
# Import Packages
# ################################################################
import sys

from scipy import integrate
from vtkmodules.vtkChartsCore import vtkChart
from vtkmodules.vtkRenderingContext2D import vtkPen
from vtkmodules.vtkChartsCore import vtkPlotPoints
from vtkmodules.vtkViewsContext2D import vtkContextView
from vtkmodules.vtkChartsCore import vtkChartXY
from vtkmodules.vtkCommonDataModel import vtkTable
from vtkmodules.vtkCommonCore import vtkFloatArray

chart = None


class vtkTimerCallback():
    def __init__(self):
        self.timer_count = 0
        self.timeStamp = 0

    def execute(self, obj, event):

        numP = chart.GetNumberOfPlots()
        global bBox
        self.timer_count += 1

        # Execution for the first Animation
        if self.timer_count <= 100:
            numP = chart.GetNumberOfPlots()
            translation = 0.03

            val1 = boxB.GetRow(0).GetValue(0).ToFloat();
            val2 = boxB.GetRow(0).GetValue(1).ToFloat();
            val3 = boxB.GetRow(1).GetValue(0).ToFloat();
            val4 = boxB.GetRow(1).GetValue(1).ToFloat();
            val5 = boxB.GetRow(2).GetValue(0).ToFloat();
            val6 = boxB.GetRow(2).GetValue(1).ToFloat();
            val7 = boxB.GetRow(3).GetValue(0).ToFloat();
            val8 = boxB.GetRow(3).GetValue(1).ToFloat();

            boxB.SetValue(0, 0, val1 + translation)
            boxB.SetValue(0, 1, val2)
            boxB.SetValue(1, 0, val3 + translation)
            boxB.SetValue(1, 1, val4)
            boxB.SetValue(2, 0, val5 + translation)
            boxB.SetValue(2, 1, val6)
            boxB.SetValue(3, 0, val7 + translation)
            boxB.SetValue(3, 1, val8)

            line2 = chart.AddPlot(vtkChart.LINE)
            line2.SetInputData(boxB, 0, 1)
            line2.SetColor(255, 0, 0, 255)
            line2.SetWidth(3.0)
            line2.GetPen().SetLineType(vtkPen.SOLID_LINE)
            line2.SetMarkerStyle(vtkPlotPoints.CIRCLE)

            line3 = chart.AddPlot(vtkChart.LINE)
            line3.SetInputData(boxA, 0, 1)
            line3.SetColor(0, 0, 0, 255)
            line3.SetWidth(3.0)
            line3.GetPen().SetLineType(vtkPen.SOLID_LINE)
            line3.SetMarkerStyle(vtkPlotPoints.CIRCLE)

            # call the convolution Function Drawer

            convolutionDrawerBoxBox(boxB.GetRow(3).GetValue(0).ToFloat() - 0.5,
                                    self.timer_count - 1)
            for i in range(0, numP - 1):
                chart.RemovePlot(1)

        # clearing the Plot when reached the end
        if self.timer_count == 310:

            # reset the plotter
            for i in range(0, numP - 1):
                chart.RemovePlot(1)

            # reset the convolution box
            boxB.SetValue(0, 0, -2.5)
            boxB.SetValue(0, 1, 0.0)
            boxB.SetValue(1, 0, -2.5)
            boxB.SetValue(1, 1, 1)
            boxB.SetValue(2, 0, -1.5)
            boxB.SetValue(2, 1, 1.0)
            boxB.SetValue(3, 0, -1.5)
            boxB.SetValue(3, 1, 0)

            line2 = chart.AddPlot(vtkChart.LINE)
            line2.SetInputData(boxB, 0, 1)
            line2.SetColor(255, 0, 0, 255)
            line2.SetWidth(3.0)
            line2.GetPen().SetLineType(vtkPen.SOLID_LINE)
            line2.SetMarkerStyle(vtkPlotPoints.CIRCLE)

            boxA.SetValue(0, 0, 0)
            boxA.SetValue(0, 1, 0)
            boxA.SetValue(1, 0, 1)
            boxA.SetValue(1, 1, 1)
            boxA.SetValue(2, 0, 2)
            boxA.SetValue(2, 1, 0)
            boxA.SetValue(3, 0, 2)
            boxA.SetValue(3, 1, 0)

            line3 = chart.AddPlot(vtkChart.LINE)
            line3.SetInputData(boxA, 0, 1)
            line3.SetColor(0, 0, 255, 255)
            line3.SetWidth(3.0)
            line3.GetPen().SetLineType(vtkPen.SOLID_LINE)
            line3.SetMarkerStyle(vtkPlotPoints.CIRCLE)

        # Now draw the new Function
        if self.timer_count > 310 and self.timer_count < 440:
            translation = 0.03
            # get number of plots
            numP = chart.GetNumberOfPlots()
            val1 = boxB.GetRow(0).GetValue(0).ToFloat();
            val2 = boxB.GetRow(0).GetValue(1).ToFloat();
            val3 = boxB.GetRow(1).GetValue(0).ToFloat();
            val4 = boxB.GetRow(1).GetValue(1).ToFloat();
            val5 = boxB.GetRow(2).GetValue(0).ToFloat();
            val6 = boxB.GetRow(2).GetValue(1).ToFloat();
            val7 = boxB.GetRow(3).GetValue(0).ToFloat();
            val8 = boxB.GetRow(3).GetValue(1).ToFloat();

            boxB.SetValue(0, 0, val1 + translation)
            boxB.SetValue(0, 1, val2)
            boxB.SetValue(1, 0, val3 + translation)
            boxB.SetValue(1, 1, val4)
            boxB.SetValue(2, 0, val5 + translation)
            boxB.SetValue(2, 1, val6)
            boxB.SetValue(3, 0, val7 + translation)
            boxB.SetValue(3, 1, val8)

            line2 = chart.AddPlot(vtkChart.LINE)
            line2.SetInputData(boxB, 0, 1)
            line2.SetColor(255, 0, 0, 255)
            line2.SetWidth(3.0)
            line2.GetPen().SetLineType(vtkPen.SOLID_LINE)
            line2.SetMarkerStyle(vtkPlotPoints.CIRCLE)

            boxA.SetValue(0, 0, -1.0)
            boxA.SetValue(0, 1, 0)
            boxA.SetValue(1, 0, 0)
            boxA.SetValue(1, 1, 1)
            boxA.SetValue(2, 0, 1.0)
            boxA.SetValue(2, 1, 0)
            boxA.SetValue(3, 0, 1.0)
            boxA.SetValue(3, 1, 0)

            line3 = chart.AddPlot(vtkChart.LINE)
            line3.SetInputData(boxA, 0, 1)
            line3.SetColor(0, 0, 255, 255)
            line3.SetWidth(3.0)
            line3.GetPen().SetLineType(vtkPen.SOLID_LINE)
            line3.SetMarkerStyle(vtkPlotPoints.CIRCLE)

            convolutionDrawerBoxTriangle(boxB.GetRow(3).GetValue(0).ToFloat() - 0.5,
                                         self.timeStamp)
            self.timeStamp += 1
            # reset the plotter
            for i in range(0, numP - 1):
                chart.RemovePlot(1)


def computeAreaBoxTriangle(xpos):
    # evaluates the integral for shift value xpos
    left_bound = xpos - 0.5
    right_bound = xpos + 0.5

    def triangleArea(x):
        return 1 - abs(x) if abs(x) < 1 else 0

    # compute the area of the triangle from left_bound to right_bound
    result = integrate.quad(triangleArea, left_bound, right_bound)

    return result[0]


def convolutionDrawerBoxTriangle(xval, timestamp):
    boxD.SetNumberOfRows(timestamp + 1)
    boxD.SetValue(timestamp, 0, xval)
    boxD.SetValue(timestamp, 1, computeAreaBoxTriangle(xval))
    # we render points here
    points = chart.AddPlot(vtkChart.LINE)
    points.SetInputData(boxD, 0, 1)
    points.SetColor(0, 255, 0, 255)
    points.SetWidth(3.0)


def computeAreaBoxBox(xpos):
    # evaluates the integral for shift value xpos
    left_bound = xpos - 0.5
    right_bound = xpos + 0.5

    def boxArea(x):
        return 1 if abs(x) < 0.5 else 0

    # compute the area of the triangle from left_bound to right_bound
    result = integrate.quad(boxArea, left_bound, right_bound)

    return result[0]


def convolutionDrawerBoxBox(xval, timestamp):
    boxD.SetNumberOfRows(timestamp + 1)
    boxD.SetValue(timestamp, 0, xval)
    boxD.SetValue(timestamp, 1, computeAreaBoxBox(xval))
    # we render points here
    points = chart.AddPlot(vtkChart.LINE)
    points.SetInputData(boxD, 0, 1)
    points.SetColor(50, 50, 255, 255)
    points.SetWidth(3.0)


# Defining the Main Function
def main(argv):
    view = vtkContextView()
    view.GetRenderer().SetBackground(1.0, 1.0, 1.0)
    view.GetRenderWindow().SetSize(1000, 400)

    global chart
    chart = vtkChartXY()
    view.GetScene().AddItem(chart)
    chart.SetShowLegend(False)
    chart.AutoAxesOn()
    chart.SetForceAxesToBounds(False)
    table = vtkTable()

    arrX = vtkFloatArray()
    arrX.SetName('X Axis')

    arrC = vtkFloatArray()
    arrC.SetName('Y Axis')

    table.AddColumn(arrX)
    table.AddColumn(arrC)

    table.SetNumberOfRows(2)
    table.SetValue(0, 0, -2.5)
    table.SetValue(0, 1, 0)

    table.SetValue(1, 0, 2.5)
    table.SetValue(1, 1, 1.4)

    points = chart.AddPlot(vtkChart.POINTS)
    points.SetInputData(table, 0, 1)
    points.SetColor(0, 0, 0, 255)
    points.SetWidth(1.0)
    points.SetMarkerStyle(vtkPlotPoints.CROSS)

    # define the stationary (gray) box
    global boxA
    boxA = vtkTable()
    boxAx = vtkFloatArray()
    boxAx.SetName('X Axis')

    boxAy = vtkFloatArray()
    boxAy.SetName('Y Axis')

    boxA.AddColumn(boxAx)
    boxA.AddColumn(boxAy)

    boxA.SetNumberOfRows(4)
    boxA.SetValue(0, 0, -0.5)
    boxA.SetValue(0, 1, 0.0)

    boxA.SetValue(1, 0, -0.5)
    boxA.SetValue(1, 1, 1)

    boxA.SetValue(2, 0, 0.5)
    boxA.SetValue(2, 1, 1.0)
    boxA.SetValue(3, 0, 0.5)
    boxA.SetValue(3, 1, 0)

    line0 = chart.AddPlot(vtkChart.LINE)
    line0.SetInputData(boxA, 0, 1)
    line0.SetColor(50, 50, 50, 255)
    line0.SetWidth(3.0)
    line0.GetPen().SetLineType(vtkPen.SOLID_LINE)
    line0.SetMarkerStyle(vtkPlotPoints.CIRCLE)

    # define the moving (red) box
    global boxB
    boxB = vtkTable()
    boxBx = vtkFloatArray()
    boxBx.SetName('X Axis')

    boxBy = vtkFloatArray()
    boxBy.SetName('Y Axis')

    boxB.AddColumn(boxBx)
    boxB.AddColumn(boxBy)

    boxB.SetNumberOfRows(4)
    boxB.SetValue(0, 0, -2.0)
    boxB.SetValue(0, 1, 0.0)

    boxB.SetValue(1, 0, -2.0)
    boxB.SetValue(1, 1, 1)

    boxB.SetValue(2, 0, -1.0)
    boxB.SetValue(2, 1, 1.0)
    boxB.SetValue(3, 0, -1.0)
    boxB.SetValue(3, 1, 0)

    line1 = chart.AddPlot(vtkChart.LINE)
    line1.SetInputData(boxB, 0, 1)
    line1.SetColor(255, 0, 0, 255)
    line1.SetWidth(3.0)
    line1.GetPen().SetLineType(vtkPen.SOLID_LINE)
    line1.SetMarkerStyle(vtkPlotPoints.CIRCLE)

    global boxC
    boxC = vtkTable()
    boxCx = vtkFloatArray()
    boxCx.SetName('X Axis')
    boxCy = vtkFloatArray()
    boxCy.SetName('Y Axis')
    boxC.SetNumberOfRows(3)
    boxC.AddColumn(boxCx)
    boxC.AddColumn(boxCy)
    boxC.SetNumberOfRows(3)

    boxC.SetValue(0, 0, 0)
    boxC.SetValue(0, 1, 0)
    boxC.SetValue(1, 0, 0)
    boxC.SetValue(1, 1, 0)
    boxC.SetValue(2, 0, 0)
    boxC.SetValue(2, 1, 0)

    line0 = chart.AddPlot(vtkChart.LINE)
    line0.SetInputData(boxC, 0, 1)
    line0.SetColor(50, 50, 255, 255)
    line0.SetWidth(3.0)
    line0.GetPen().SetLineType(vtkPen.SOLID_LINE)
    line0.SetMarkerStyle(vtkPlotPoints.CIRCLE)

    global boxD
    boxD = vtkTable()
    boxDx = vtkFloatArray()
    boxDx.SetName('X Axis')
    boxDy = vtkFloatArray()
    boxDy.SetName('Y Axis')
    boxD.AddColumn(boxDx)
    boxD.AddColumn(boxDy)
    boxD.SetNumberOfRows(130)

    line3 = chart.AddPlot(vtkChart.LINE)
    line3.SetInputData(boxD, 0, 1)
    line3.SetColor(50, 255, 0, 255)
    line3.SetWidth(3.0)
    line3.GetPen().SetLineType(vtkPen.SOLID_LINE)
    line3.SetMarkerStyle(vtkPlotPoints.CIRCLE)

    # create timerCallBack
    cb = vtkTimerCallback()
    view.GetRenderWindow().SetMultiSamples(0)
    view.GetInteractor().Initialize()
    view.GetInteractor().AddObserver('TimerEvent', cb.execute)
    timerId = view.GetInteractor().CreateRepeatingTimer(20);
    view.GetInteractor().Start()


boxA = None
boxB = None
boxC = None
boxD = None

# Entry point
if __name__ == "__main__":
    main(sys.argv[1:])
