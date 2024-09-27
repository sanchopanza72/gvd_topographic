import vtk
import numpy as np
import potpourri3d as p3d
from scipy.interpolate import Rbf
from sklearn.manifold import MDS, Isomap
import random
from scipy.spatial.distance import squareform,cdist,pdist
from scipy.special import xlogy
import scipy.linalg as la

def rbf00(r):
    return xlogy(r ** 2, r)  # thin plate

def rbf01(r, epsilon):
    # return np.exp(-(1.0/epsilon*r)**2) #gaussian
    return np.sqrt((1.0 / epsilon * r) ** 2 + 1)  # multiquadric

# natural neighbor geodesic
def nng(mpd: vtk.vtkPolyData):
    cids = vtk.vtkIdList()
    mpd.GetPointCells(0, cids) # cells incident to point 0
    assert (cids.GetNumberOfIds()>0)
    # check if point 0 is on the boundary
    fcs=set() # first & last facet id
    for k in range(cids.GetNumberOfIds()):
        cid=cids.GetId(k)
        pis=vtk.vtkIdList()
        mpd.GetCellPoints(cid,pis)
        v1=-1
        v2=-1
        for m in range(3):
            if pis.GetId(m) != 0: v1 = pis.GetId(m)
        for m in range(3):
            if pis.GetId(m) != 0 and pis.GetId(m) != v1: v2 = pis.GetId(m)
        cis = vtk.vtkIdList()
        mpd.GetCellEdgeNeighbors(cid, 0, v1, cis)
        if cis.GetNumberOfIds() < 1: fcs.add(cid)
        mpd.GetCellEdgeNeighbors(cid, 0, v2, cis)
        if cis.GetNumberOfIds() < 1: fcs.add(cid)

    pts = vtk.vtkPoints()
    pts.SetDataTypeToDouble()
    pts.InsertNextPoint(mpd.GetPoint(0))
    if len(fcs)==2:
        fs=list(fcs)[0] # start
        fe=list(fcs)[1] # end
        fn=-1 # next
        pis = vtk.vtkIdList()
        mpd.GetCellPoints(fs, pis)
        v1 = -1
        v2 = -1
        pset0 = set()
        pset1 = set()
        for m in range(3):
            if pis.GetId(m) != 0: v1 = pis.GetId(m)
            pset0.add(pis.GetId(m))
        for m in range(3):
            if pis.GetId(m) != 0 and pis.GetId(m) != v1: v2 = pis.GetId(m)
        cis = vtk.vtkIdList()
        mpd.GetCellEdgeNeighbors(fs, 0, v1, cis)
        if cis.GetNumberOfIds() > 0: fn = cis.GetId(0)
        mpd.GetCellEdgeNeighbors(fs, 0, v2, cis)
        if cis.GetNumberOfIds() > 0: fn = cis.GetId(0)
        mpd.GetCellPoints(fn,pis)
        for m in range(3): pset1.add(pis.GetId(m))
        vs = list(pset0 - (pset1 & pset0))[0]  # start point id
        v1 = list(pset0 - {vs, 0})[0]
        v2 = list(pset1 - {v1, 0})[0] # next point!
        pts.InsertNextPoint(mpd.GetPoint(vs))
        pts.InsertNextPoint(mpd.GetPoint(v1))
        pts.InsertNextPoint(mpd.GetPoint(v2))
        while fn!=fe:
            mpd.GetCellEdgeNeighbors(fn, 0, v2, cis)
            fn = cis.GetId(0)
            pset0.clear()
            pset0 = pset1.copy()
            pset1.clear()
            mpd.GetCellPoints(fn,pis)
            for m in range(3): pset1.add(pis.GetId(m))
            v2=list(pset1-(pset1&pset0))[0]
            pts.InsertNextPoint(mpd.GetPoint(v2))
    else:
        f0 = cids.GetId(0)  # the first cell
        pis = vtk.vtkIdList()  # point ids in the first cell
        mpd.GetCellPoints(f0, pis)
        v1 = -1
        v2 = -1
        ve = -1
        for m in range(3):
            if pis.GetId(m) != 0: v1 = pis.GetId(m)
        for m in range(3):
            if pis.GetId(m) != 0 and pis.GetId(m) != v1: ve = pis.GetId(m)
        pts.InsertNextPoint(mpd.GetPoint(v1))
        cis=vtk.vtkIdList()
        mpd.GetCellEdgeNeighbors(f0, 0, v1, cis)
        f1=cis.GetId(0)
        while f1!=f0:
            mpd.GetCellPoints(f1,pis)
            for m in range(3):
                if pis.GetId(m)!=0 and pis.GetId(m)!=v1: v2=pis.GetId(m)
            v1=v2
            pts.InsertNextPoint(mpd.GetPoint(v1))
            mpd.GetCellEdgeNeighbors(f1,0,v1,cis)
            f1=cis.GetId(0)

    ts=vtk.vtkCellArray()
    for i in range(1,pts.GetNumberOfPoints()-1):
        t=vtk.vtkTriangle()
        t.GetPointIds().SetId(0, 0)
        t.GetPointIds().SetId(1, i)
        t.GetPointIds().SetId(2, i+1)
        ts.InsertNextCell(t)
    if len(fcs)==0:
        t=vtk.vtkTriangle()
        t.GetPointIds().SetId(0, 0)
        t.GetPointIds().SetId(1, pts.GetNumberOfPoints() - 1)
        t.GetPointIds().SetId(2, 1)
        ts.InsertNextCell(t)
    npd=vtk.vtkPolyData()
    npd.SetPoints(pts)
    npd.SetPolys(ts)
    subd=vtk.vtkLinearSubdivisionFilter()
    subd.SetInputData(npd)
    subd.SetNumberOfSubdivisions(4) # greater number of subdivision than 4 would be extremely expensive
    subd.Update()
    npd.ShallowCopy(subd.GetOutput())

    vl=[]
    tN=pts.GetNumberOfPoints()
    for i in range(tN):
        cd=np.zeros(3,dtype=np.float64)
        pts.GetPoint(i,cd)
        vl.append(cd)
    pis0 = vtk.vtkIdList()
    p0 = np.zeros(3, dtype=np.float64)
    p1 = np.zeros(3, dtype=np.float64)
    p2 = np.zeros(3, dtype=np.float64)
    for i in range(npd.GetNumberOfCells()):
        npd.GetCellPoints(i,pis0)
        npd.GetPoint(pis0.GetId(0), p0)
        npd.GetPoint(pis0.GetId(1), p1)
        npd.GetPoint(pis0.GetId(2), p2)
        vl.append((p0+p1+p2)/3.0)
    VL = np.array(vl, dtype=np.float64)
    tar0=np.asarray(range(1,pts.GetNumberOfPoints()))
    tar1=np.asarray(range(pts.GetNumberOfPoints()))
    slv = p3d.PointCloudHeatSolver(VL)

    ia = vtk.vtkIntArray()
    ib = vtk.vtkIntArray()
    ia.SetName('va')
    ib.SetName('vb')
    for i in range(npd.GetNumberOfCells()):
        dis=slv.compute_distance(i+tN)
        disA=dis[tar0]
        disB=dis[tar1]
        ia.InsertNextTuple1(np.argmin(disA))
        ib.InsertNextTuple1(np.argmin(disB))

    lst_lst = []
    for i in range(tN - 1):
        al = []
        al.append(i)   # reference point id
        al.append(0.0) # reference area
        lst_lst.append(al)
    ta = 0.0 #total area
    for i in range(npd.GetNumberOfCells()):
        if ib.GetTuple1(i) != 0: continue
        t = vtk.vtkTriangle()
        t.ShallowCopy(npd.GetCell(i))
        a = t.ComputeArea()
        ta += a
        ct = ia.GetTuple1(i)
        for al in lst_lst:
            if al[0] == ct: al[1] += a
    vj=0
    for al in lst_lst:
        rv=pts.GetPoint(al[0]+1)[2]
        wj=al[1]/ta
        vj+=rv*wj
    return vj

#natural neighbor Euclidean
def nne(mpd: vtk.vtkPolyData):
    cids = vtk.vtkIdList()
    mpd.GetPointCells(0, cids)  # cells incident to point 0
    assert (cids.GetNumberOfIds() > 0)
    # check if point 0 is on the boundary
    fcs = set()  # first & last facet id
    for k in range(cids.GetNumberOfIds()):
        cid = cids.GetId(k)
        pis = vtk.vtkIdList()
        mpd.GetCellPoints(cid, pis)
        v1 = -1
        v2 = -1
        for m in range(3):
            if pis.GetId(m) != 0: v1 = pis.GetId(m)
        for m in range(3):
            if pis.GetId(m) != 0 and pis.GetId(m) != v1: v2 = pis.GetId(m)
        cis = vtk.vtkIdList()
        mpd.GetCellEdgeNeighbors(cid, 0, v1, cis)
        if cis.GetNumberOfIds() < 1: fcs.add(cid)
        mpd.GetCellEdgeNeighbors(cid, 0, v2, cis)
        if cis.GetNumberOfIds() < 1: fcs.add(cid)

    pts = vtk.vtkPoints()
    pts.SetDataTypeToDouble()
    pts.InsertNextPoint(mpd.GetPoint(0))
    if len(fcs) == 2:
        fs = list(fcs)[0]  # start
        fe = list(fcs)[1]  # end
        fn = -1  # next
        pis = vtk.vtkIdList()
        mpd.GetCellPoints(fs, pis)
        v1 = -1
        v2 = -1
        pset0 = set()
        pset1 = set()
        for m in range(3):
            if pis.GetId(m) != 0: v1 = pis.GetId(m)
            pset0.add(pis.GetId(m))
        for m in range(3):
            if pis.GetId(m) != 0 and pis.GetId(m) != v1: v2 = pis.GetId(m)
        cis = vtk.vtkIdList()
        mpd.GetCellEdgeNeighbors(fs, 0, v1, cis)
        if cis.GetNumberOfIds() > 0: fn = cis.GetId(0)
        mpd.GetCellEdgeNeighbors(fs, 0, v2, cis)
        if cis.GetNumberOfIds() > 0: fn = cis.GetId(0)
        mpd.GetCellPoints(fn, pis)
        for m in range(3): pset1.add(pis.GetId(m))
        vs = list(pset0 - (pset1 & pset0))[0]  # start point id
        v1 = list(pset0 - {vs, 0})[0]
        v2 = list(pset1 - {v1, 0})[0]  # next point!
        pts.InsertNextPoint(mpd.GetPoint(vs))
        pts.InsertNextPoint(mpd.GetPoint(v1))
        pts.InsertNextPoint(mpd.GetPoint(v2))
        while fn != fe:
            mpd.GetCellEdgeNeighbors(fn, 0, v2, cis)
            fn = cis.GetId(0)
            pset0.clear()
            pset0 = pset1.copy()
            pset1.clear()
            mpd.GetCellPoints(fn, pis)
            for m in range(3): pset1.add(pis.GetId(m))
            v2 = list(pset1 - (pset1 & pset0))[0]
            pts.InsertNextPoint(mpd.GetPoint(v2))
    else:
        f0 = cids.GetId(0)  # the first cell
        pis = vtk.vtkIdList()  # point ids in the first cell
        mpd.GetCellPoints(f0, pis)
        v1 = -1
        v2 = -1
        ve = -1
        for m in range(3):
            if pis.GetId(m) != 0: v1 = pis.GetId(m)
        for m in range(3):
            if pis.GetId(m) != 0 and pis.GetId(m) != v1: ve = pis.GetId(m)
        pts.InsertNextPoint(mpd.GetPoint(v1))
        cis = vtk.vtkIdList()
        mpd.GetCellEdgeNeighbors(f0, 0, v1, cis)
        f1 = cis.GetId(0)
        while f1 != f0:
            mpd.GetCellPoints(f1, pis)
            for m in range(3):
                if pis.GetId(m) != 0 and pis.GetId(m) != v1: v2 = pis.GetId(m)
            v1 = v2
            pts.InsertNextPoint(mpd.GetPoint(v1))
            mpd.GetCellEdgeNeighbors(f1, 0, v1, cis)
            f1 = cis.GetId(0)

    ts = vtk.vtkCellArray()
    for i in range(1, pts.GetNumberOfPoints() - 1):
        t = vtk.vtkTriangle()
        t.GetPointIds().SetId(0, 0)
        t.GetPointIds().SetId(1, i)
        t.GetPointIds().SetId(2, i + 1)
        ts.InsertNextCell(t)
    if len(fcs) == 0:
        t = vtk.vtkTriangle()
        t.GetPointIds().SetId(0, 0)
        t.GetPointIds().SetId(1, pts.GetNumberOfPoints() - 1)
        t.GetPointIds().SetId(2, 1)
        ts.InsertNextCell(t)
    npd = vtk.vtkPolyData()
    npd.SetPoints(pts)
    npd.SetPolys(ts)
    subd = vtk.vtkLinearSubdivisionFilter()
    subd.SetInputData(npd)
    subd.SetNumberOfSubdivisions(4)  # greater number of subdivision than 4 would be extremely expensive
    subd.Update()
    npd.ShallowCopy(subd.GetOutput())

    vl = []
    tN = pts.GetNumberOfPoints()
    pis0 = vtk.vtkIdList()
    p0 = np.zeros(3, dtype=np.float64)
    p1 = np.zeros(3, dtype=np.float64)
    p2 = np.zeros(3, dtype=np.float64)
    for i in range(npd.GetNumberOfCells()):
        npd.GetCellPoints(i, pis0)
        npd.GetPoint(pis0.GetId(0), p0)
        npd.GetPoint(pis0.GetId(1), p1)
        npd.GetPoint(pis0.GetId(2), p2)
        vl.append((p0 + p1 + p2) / 3.0)
    VL = np.array(vl, dtype=np.float64)
    tar0 = []
    for i in range(1, tN):
        cd = np.zeros(3, dtype=np.float64)
        pts.GetPoint(i, cd)
        tar0.append(cd)
    tar0 = np.asarray(tar0, dtype=np.float64)
    tar1 = []
    for i in range(tN):
        cd = np.zeros(3, dtype=np.float64)
        pts.GetPoint(i, cd)
        tar1.append(cd)
    tar1 = np.asarray(tar1, dtype=np.float64)

    ia = vtk.vtkIntArray()
    ib = vtk.vtkIntArray()
    ia.SetName('va')
    ib.SetName('vb')
    for i in range(npd.GetNumberOfCells()):
        disA = cdist(tar0, [VL[i]])
        disB = cdist(tar1, [VL[i]])
        ia.InsertNextTuple1(np.argmin(disA))
        ib.InsertNextTuple1(np.argmin(disB))

    lst_lst = []
    for i in range(tN - 1):
        al = []
        al.append(i)  # reference point id
        al.append(0.0)  # reference area
        lst_lst.append(al)
    ta = 0.0  # total area
    for i in range(npd.GetNumberOfCells()):
        if ib.GetTuple1(i) != 0: continue
        t = vtk.vtkTriangle()
        t.ShallowCopy(npd.GetCell(i))
        a = t.ComputeArea()
        ta += a
        ct = ia.GetTuple1(i)
        for al in lst_lst:
            if al[0] == ct: al[1] += a
    vj = 0
    for al in lst_lst:
        rv = pts.GetPoint(al[0] + 1)[2]
        wj = al[1] / ta
        vj += rv * wj
    return vj

sr = vtk.vtkPolyDataReader()
sr.SetFileName('d:/tin/sh10.vtk')
sr.Update()

pd = vtk.vtkPolyData()
pd.ShallowCopy(sr.GetOutput())

loc=vtk.vtkKdTreePointLocator()
loc.SetDataSet(pd)
loc.BuildLocator()

binSize=300
binRadiaus=200
crsSamples=50 #number of samples for cross-validation

smp=random.sample(range(pd.GetNumberOfPoints()), crsSamples)
np.savetxt('D:/tin/random_150_sites.csv',smp, fmt='%d')
smp = np.loadtxt('D:/tin/random_150_sites.csv', dtype=int)

#idw
idw_search_size = 100
idw_ref_size = 5
idw_pwr= 2.5

#idw Euclidean
smpz=[] #ground truth
simz=[] #interpolation
for i in smp:
    crd = np.zeros(3, np.float64)
    pd.GetPoint(i, crd)
    smpz.append(crd[2])
    pis = vtk.vtkIdList()
    loc.FindClosestNPoints(idw_search_size, crd, pis)

    ax = np.repeat(crd[0], idw_search_size - 1)
    ay = np.repeat(crd[1], idw_search_size - 1)
    az = np.repeat(crd[2], idw_search_size - 1)
    bx = [];    by = [];    bz = [];
    for j in range(idw_search_size):
        if pis.GetId(j) == i: continue
        crd0 = np.zeros(3, dtype=np.float64)
        pd.GetPoint(pis.GetId(j), crd0)
        bx.append(crd0[0])
        by.append(crd0[1])
        bz.append(crd0[2])
    bx=np.asarray(bx);by=np.asarray(by);bz=np.asarray(bz)
    diss = np.sqrt(np.square(ax - bx) + np.square(ay - by) + np.square(az - bz))
    sts = np.argsort(diss)

    wjs = 0
    for j in range(idw_ref_size):
        eud = diss[sts[j]]
        wj = 1 / (eud ** idw_pwr)
        wjs += wj

    vjs = 0
    for j in range(idw_ref_size):
        eud = diss[sts[j]]
        wj = 1 / (eud ** idw_pwr)
        wj = wj / wjs
        vj = wj * bz[sts[j]]
        vjs += vj
    simz.append(vjs)
#print out
dv = np.subtract(smpz, simz)
sq = np.square(dv)
mse = np.mean(sq)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(dv))
print('idw planar:', mse, rmse, mae)


#idw geodesic
simz=[] #interpolation
for i in smp:
    vlist = []

    crd=np.zeros(3,dtype=np.float64)
    pd.GetPoint(i, crd)
    vlist.append(crd)

    ids=vtk.vtkIdList()
    loc.FindClosestNPoints(idw_search_size,crd,ids) #euclidean
    bz = []
    bz.append(crd[2])
    for j in range(ids.GetNumberOfIds()):
        if ids.GetId(j)==i: continue
        crd0=np.zeros(3,dtype=np.float64)
        pd.GetPoint(ids.GetId(j),crd0)
        bz.append(crd0[2])
        vlist.append(crd0)
    V = np.array(vlist, dtype=np.float64)
    solver = p3d.PointCloudHeatSolver(V)
    diss=solver.compute_distance(0)
    sts=np.argsort(diss)

    wjs=0
    for j in range(idw_ref_size):
        egs=diss[sts[j+1]]
        wj=1/(egs**idw_pwr)
        wjs+=wj
    vjs=0
    for j in range(idw_ref_size):
        egs=diss[sts[j+1]]
        wj=1/(egs**idw_pwr)
        vj=wj/wjs*bz[sts[j+1]]
        vjs+=vj
    simz.append(vjs)
dv = np.subtract(smpz, simz)
sq = np.square(dv)
mse = np.mean(sq)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(dv))
print('idw geodesic:', mse, rmse, mae)

nnve=[]
nnvg=[]
for i in smp:
    pns = vtk.vtkPoints()
    pns.SetDataTypeToDouble()
    crd = np.zeros(3, dtype=np.float64)
    pd.GetPoint(i, crd)
    pns.InsertNextPoint(crd)

    ids = vtk.vtkIdList()
    loc.FindClosestNPoints(idw_search_size, crd, ids)  # euclidean
    for j in range(ids.GetNumberOfIds()):
        if ids.GetId(j) == i: continue
        crd0 = np.zeros(3, dtype=np.float64)
        pd.GetPoint(ids.GetId(j), crd0)
        pns.InsertNextPoint(crd0)

    npd=vtk.vtkPolyData()
    npd.SetPoints(pns)
    d2d=vtk.vtkDelaunay2D()
    d2d.SetInputData(npd)
    #d2d.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
    d2d.Update()
    tpd=vtk.vtkPolyData()
    tpd.ShallowCopy(d2d.GetOutput())
    nnve.append(nne(tpd))
    nnvg.append(nng(tpd))

#print out
dv = np.subtract(smpz, nnve)
sq = np.square(dv)
mse = np.mean(sq)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(dv))
print('nn planar:', mse, rmse, mae)

dv = np.subtract(smpz, nnvg)
sq = np.square(dv)
mse = np.mean(sq)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(dv))
print('nn geodesic:', mse, rmse, mae)


#rbf
simz=[] #interpolation
for i in smp:
    cri=np.zeros(3,dtype=np.float64)
    pd.GetPoint(i, cri)
    ids=vtk.vtkIdList()
    loc.FindClosestNPoints(binSize,cri,ids)
    gtx=[]
    gty=[]
    gtz=[] #ground truth
    for j in range(ids.GetNumberOfIds()):
        if ids.GetId(j)==i: continue
        crj=np.zeros(3,dtype=np.float64)
        pd.GetPoint(ids.GetId(j),crj)
        gtx.append(crj[0])
        gty.append(crj[1])
        gtz.append(crj[2])
    # print(i)
    func = Rbf(gtx, gty, gtz, function='thin-plate')
    simz.append(func(cri[0], cri[1]))
#print out
dv = np.subtract(smpz, simz)
sq = np.square(dv)
mse = np.mean(sq)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(dv))
print('rbf planar:', mse, rmse, mae)

#rbf geodesic
simz=[] #interpolation
for i in smp:
    vlist = []

    cri=np.zeros(3,dtype=np.float64)
    pd.GetPoint(i, cri)
    vlist.append(cri)

    ids=vtk.vtkIdList()
    loc.FindClosestNPoints(binSize,cri,ids) #euclidean
    gtz=[]
    for j in range(ids.GetNumberOfIds()):
        if ids.GetId(j)==i: continue
        crj=np.zeros(3,dtype=np.float64)
        pd.GetPoint(ids.GetId(j),crj)
        gtz.append(crj[2])
        vlist.append(crj)

    V = np.array(vlist, dtype=np.float64)
    solver = p3d.PointCloudHeatSolver(V)
    wdl=[]
    tar=np.asarray(range(len(V)))
    for j in range(len(V)):
        dist = solver.compute_distance(j)
        dfn = tar[j + 1:]
        dist1 = dist[dfn]
        wdl.extend(dist1)
    wdm = squareform(wdl)
    mds = MDS(dissimilarity='precomputed', random_state=0,normalized_stress='auto')
    tdx = mds.fit_transform(wdm)
    subc=tdx[1:,]

    func = Rbf(subc[:,0], subc[:,1], gtz, function='thin-plate')
    simz.append(func(tdx[0][0], tdx[0][1]))

dv = np.subtract(smpz, simz)
sq = np.square(dv)
mse = np.mean(sq)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(dv))
print('nn geodesic:', mse, rmse, mae)

