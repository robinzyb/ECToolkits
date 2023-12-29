import numpy as np


def birch_murnaghan_equation(V, V0, E0, B0, B0_prime):
    V_ratio = np.power(np.divide(V0, V), np.divide(2, 3))
    E = E0 + np.divide((9 * V0 * B0), 16) * (np.power(V_ratio - 1, 3) * B0_prime
                                             + np.power((V_ratio - 1), 2) * (6 - 4 * V_ratio))
    return E


def fit_plane_normal(xyz):
    """Three dimensional plane fitting for a group of points

    Args:
        xyz (numpy.ndarray): Your group of points, i.e., atoms in non-orthogonal cells forming a plane

    Returns:
        numpy.ndarray:
            Normalized plane's normal vector. z>0.
    """
    A = np.concatenate(
        [xyz[:, :-1], np.ones(xyz.shape[0])[:, np.newaxis]], axis=1)
    b = xyz[:, -1]

    tA = A.T
    RHS = np.matmul(tA, b)
    tAA = np.matmul(tA, A)

    x = np.matmul(np.linalg.pinv(tAA), RHS)
    pred_b = np.matmul(A, x)
    std_devi = np.sum((pred_b - b)**2)/b.shape[0]
    print("Standard deviation for plain fitting is {0:.3e}".format(std_devi))
    normal_z = np.append(x[:-1], [-1])
    return -normal_z/np.linalg.norm(normal_z)


def fit_line_vec(xyz):
    """Fit the direction vector for a group of sampling points.
    Use PCA algorithm find the direction vector for a group of points.
    The PCA method reference:
    https://math.stackexchange.com/questions/1611308/best-fit-line-with-3d-points#:~:text=In%20three%20dimensions%20you%20can%20similarly%20fit%20a,and%20that%20involves%20what%20are%20called%20principal%20components.

    Args:
        xyz (numpy.ndarray):
            Your group of points
    """
    x0 = xyz.mean(axis=0)
    X = xyz
    PX = X - x0
    Xt = X.T
    XtPX = np.matmul(Xt, PX)
    vals, vecs = np.linalg.eig(XtPX)
    line_vec = vecs[np.argmax(vals)]
    # How to calculating error?
    return line_vec/np.linalg.norm(line_vec)


def get_norm_vector(a, b):
    """
    obtain normal vector of a plane contain vectors a and b

    Args:
    -----------
        a (_type_):
            _description_
        b (_type_):
            _description_

    Returns:
    -----------
        _type_:
            _description_

    Notes:
    -----------
     _notes_

    Examples:
    -----------
     _examples_
    """
    nv_a = np.cross(a, b)
    norm = np.linalg.norm(nv_a)
    nv_a = nv_a/norm
    return nv_a


def get_plane_distance(a, b):

    # \
    #    \
    #     \
    #      ---------> a
    len_a = np.linalg.norm(a)
    proj_b = np.dot(b, a) * a / len_a**2
    plane_d = np.linalg.norm(b - proj_b)
    return plane_d


def get_plane_eq(a, b, c=np.array([0, 0, 1])):
    """
    obtain plane equation

    Args:
    -----------
        a (_type_):
            _description_
        b (_type_):
            _description_
        c (_type_, optional):
            _description_. Defaults to np.array([0, 0, 1]).

    Returns:
    -----------
        _type_:
            _description_

    Notes:
    -----------
     _notes_

    Examples:
    -----------
     _examples_
    """
    n_vec_a = get_norm_vector(c, a)
    d1_a = np.abs(np.dot(n_vec_a, a))
    d2_a = np.abs(np.dot(n_vec_a, a+b))
    n_vec_b = get_norm_vector(b, c)
    d1_b = np.abs(np.dot(n_vec_b, b))
    d2_b = np.abs(np.dot(n_vec_b, a+b))
    return n_vec_a, d1_a, d2_a, n_vec_b, d1_b, d2_b
