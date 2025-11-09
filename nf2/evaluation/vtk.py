import numpy as np
import vtk
from tvtk.api import tvtk, write_data


def save_vtk(path, coords=None, vectors={}, scalars={}, Mm_per_pix=720e-3):
    """Save numpy array as VTK file

    :param vectors: numpy array of the vector field (x, y, z, c)
    :param path: path to the target VTK file
    :param name: label of the vector field (e.g., B)
    :param Mm_per_pix: pixel size in Mm. 360e-3 for original HMI resolution. (default bin2 pixel scale)
    """
    # Unpack
    if len(vectors) > 0:
        dim = list(vectors.values())[0].shape[:-1]
    elif len(scalars) > 0:
        dim = list(scalars.values())[0].shape
    else:
        raise ValueError('No data to save')

    if coords is None:
        # Generate the grid
        pts = np.stack(np.mgrid[0:dim[0], 0:dim[1], 0:dim[2]], -1).astype(np.int64) * Mm_per_pix
        # reorder the points and vectors in agreement with VTK
        pts = pts.transpose(2, 1, 0, 3)
        pts = pts.reshape((-1, 3))
    else:
        pts = coords
        # reorder the points and vectors in agreement with VTK
        pts = pts.transpose(2, 1, 0, 3)
        pts = pts.reshape((-1, 3))

    # Create VTK StructuredGrid directly to avoid TVTK compatibility issues
    sg_native = vtk.vtkStructuredGrid()
    
    # Create and set points
    points = vtk.vtkPoints()
    for i in range(len(pts)):
        points.InsertNextPoint(pts[i])
    
    sg_native.SetDimensions(dim[0], dim[1], dim[2])
    sg_native.SetPoints(points)
    
    # Add vector data
    for v_name, v in vectors.items():
        v = v.transpose(2, 1, 0, 3)
        v = v.reshape((-1, 3))
        # Convert to plain numpy array to handle Astropy Quantities
        v = np.asarray(v, dtype=np.float32)
        arr = vtk.vtkFloatArray()
        arr.SetNumberOfComponents(3)
        arr.SetName(v_name)
        for i in range(len(v)):
            arr.InsertNextTuple(v[i])
        sg_native.GetPointData().AddArray(arr)
    
    # Add scalar data
    for s_name, s in scalars.items():
        s = s.transpose(2, 1, 0)
        s = s.reshape((-1))
        # Convert to plain numpy array to handle Astropy Quantities
        s = np.asarray(s, dtype=np.float32)
        arr = vtk.vtkFloatArray()
        arr.SetNumberOfComponents(1)
        arr.SetName(s_name)
        for i in range(len(s)):
            arr.InsertNextValue(s[i])
        sg_native.GetPointData().AddArray(arr)
    
    # Convert to TVTK for writing
    sg = tvtk.to_tvtk(sg_native)
    write_data(sg, path)
