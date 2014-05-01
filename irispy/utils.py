import cdd
import numpy as np

def lcon_to_vert(A, b):
    mat = cdd.Matrix([np.hstack((b[j], -A[j,:])) for j in range(A.shape[0])])
    mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat)
    ext = poly.get_generators()
    return np.hstack((np.reshape(row[1:], (-1,1)) for row in ext))