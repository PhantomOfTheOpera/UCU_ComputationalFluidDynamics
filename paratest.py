# trace generated using paraview version 5.9.0

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

import glob
import os

sz = 26
folder = '/home/virialdi/data/torchsim/exp_1/'

# create a new 'CSV Reader'
state_ = CSVReader(registrationName='state_*', FileName=sorted(glob.glob(os.path.join(folder, "*.csv"))))

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# Create a new 'SpreadSheet View'
spreadSheetView1 = CreateView('SpreadSheetView')
spreadSheetView1.ColumnToSort = ''
spreadSheetView1.BlockSize = 1024

# show data in view
state_Display = Show(state_, spreadSheetView1, 'SpreadSheetRepresentation')

# get layout
layout1 = GetLayoutByName("Layout #1")

# add view to a layout so it's visible in UI
AssignViewToLayout(view=spreadSheetView1, layout=layout1, hint=0)

# find view
renderView1 = FindViewOrCreate('RenderView1', viewtype='RenderView')

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Table To Structured Grid'
tableToStructuredGrid1 = TableToStructuredGrid(registrationName='TableToStructuredGrid1', Input=state_)
tableToStructuredGrid1.XColumn = 'p'
tableToStructuredGrid1.YColumn = 'p'
tableToStructuredGrid1.ZColumn = 'p'

# Properties modified on tableToStructuredGrid1
tableToStructuredGrid1.WholeExtent = [0, sz, 0, sz, 0, sz]
tableToStructuredGrid1.XColumn = 'x'
tableToStructuredGrid1.YColumn = 'y'
tableToStructuredGrid1.ZColumn = 'z'

# show data in view
tableToStructuredGrid1Display = Show(tableToStructuredGrid1, spreadSheetView1, 'SpreadSheetRepresentation')

# hide data in view
Hide(state_, spreadSheetView1)

# update the view to ensure updated data information
spreadSheetView1.Update()

# create a new 'Calculator'
calculator1 = Calculator(registrationName='Calculator1', Input=tableToStructuredGrid1)
calculator1.Function = ''

# Properties modified on calculator1
calculator1.Function = 'vx*iHat+vy*jHat+vz*kHat'

# show data in view
calculator1Display = Show(calculator1, spreadSheetView1, 'SpreadSheetRepresentation')

# hide data in view
Hide(tableToStructuredGrid1, spreadSheetView1)

# update the view to ensure updated data information
spreadSheetView1.Update()

# create a new 'Warp By Vector'
warpByVector1 = WarpByVector(registrationName='WarpByVector1', Input=calculator1)
warpByVector1.Vectors = ['POINTS', 'Result']



# show data in view
warpByVector1Display = Show(warpByVector1, spreadSheetView1, 'SpreadSheetRepresentation')

# hide data in view
Hide(calculator1, spreadSheetView1)

# update the view to ensure updated data information
spreadSheetView1.Update()

# create a new 'Glyph'
glyph1 = Glyph(registrationName='Glyph1', Input=warpByVector1,
    GlyphType='Arrow')
glyph1.OrientationArray = ['POINTS', 'Result']
glyph1.ScaleArray = ['POINTS', 'No scale array']
glyph1.ScaleFactor = 2.6
glyph1.GlyphTransform = 'Transform2'

# show data in view
glyph1Display = Show(glyph1, spreadSheetView1, 'SpreadSheetRepresentation')

# update the view to ensure updated data information
spreadSheetView1.Update()

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1368, 781)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [40.57159616829485, -54.809054348697394, -11.981459810728733]
renderView1.CameraFocalPoint = [14.187416888773438, 13.000000119209298, 12.312330663204193]
renderView1.CameraViewUp = [-0.2114617511455609, 0.25569229808908106, -0.9433479615181121]
renderView1.CameraParallelScale = 24.023262931536767

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
