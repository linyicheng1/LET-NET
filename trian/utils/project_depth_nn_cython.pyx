import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:,:] _project_depth_nn_cython(double[:,:] uv,
                                        double[:] z,
                                        double[:,:] depth):
    cdef int h = depth.shape[0]
    cdef int w = depth.shape[1]
    cdef int x, y
    cdef int N = len(uv)

    for idx in range(N):
        x = int(uv[idx,0])
        y = int(uv[idx,1])

        if x < 0 or y < 0 or x >= w or y >= h:
            continue

        if depth[y, x] == 0. or depth[y, x] > z[idx]:
            depth[y, x] = z[idx]
    return depth

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:,:,:] _project_image_nn_cython(double[:,:] uv,
                                            double[:] z,
                                            double[:] color,
                                            double[:,:,:] depth_image):
    cdef int h = depth_image.shape[1]
    cdef int w = depth_image.shape[2]
    cdef int x, y
    cdef int N = len(uv)

    for idx in range(N):
        x = int(uv[idx,0])
        y = int(uv[idx,1])

        if x < 0 or y < 0 or x >= w or y >= h:
            continue

        if depth_image[0,y, x] == 0. or depth_image[0,y, x] > z[idx]:
            depth_image[0,y, x] = z[idx]
            depth_image[1,y, x] = color[idx]
    return depth_image

def project_depth_nn_cython(uv, z, depth, color=None):
    if color is None:
        ret_depth = _project_depth_nn_cython(uv, z, depth)
        ret_depth = np.asarray(ret_depth)
        return ret_depth
    else:
        depth_image = np.zeros([2,depth.shape[0],depth.shape[1]],dtype=np.float64)
        ret_depth_image = _project_image_nn_cython(uv, z, color,depth_image)
        ret_depth_image = np.asarray(ret_depth_image)
        return ret_depth_image[0],ret_depth_image[1]
