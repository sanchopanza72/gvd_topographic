
datas, tests, sample codes for GVD, geodesic interpolation on topography

The scripts tests / runs with windows 11/python 3.10. 
The Natural Neighbor interpolation requires modifying part of the source VTK due to precision reason.

Please feel free to contact the author for a modified VTK wheel, and the Ordinary Kriging with geodesic distance.

Some added datasets:
b18p.vtk     topographic point cloud near Santafe, in ~0.31m resolution
i18p.vtk     1M DEM point cloud
rn18p.vtk    RBF & NN interpolated point cloud, with geodesic distance
nn18p.vtk    NN interpolated point cloud, with Euclidean distance
e_rn18p.vtk  RBF interpolated point cloud, with Euclidean distance
