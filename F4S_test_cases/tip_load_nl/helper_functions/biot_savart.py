import numpy as np
import matplotlib.pyplot as plt

"""
Functions to determine the velocity induced at a point in 3D space over time due to the UVLM
"""
def biot_savart_points(case_data, pnts: np.ndarray) -> np.ndarray:
    n_tstep = case_data.ts
    n_pnts = pnts.shape[1]

    w_ind = np.zeros([n_tstep, n_pnts, 3])

    for i_ts in range(n_tstep):
        [zeta, gamma] = extract_zeta_gamma(case_data, i_ts)

        for i_pnt in range(n_pnts):
            pnt = np.atleast_2d(pnts[:, i_pnt]).T
            w_ind[i_ts, i_pnt, :] = biot_savart(zeta, gamma, pnt)
    return w_ind

def biot_savart(zeta, gamma, pnt: np.ndarray) -> np.array:
    w_ind = np.zeros(3)

    for i_M in range(gamma.shape[0]):
        for i_N in range(gamma.shape[1]):

            zeta_v = np.zeros([3, 4])
            zeta_v[:, 0] = zeta[:, i_M, i_N]
            zeta_v[:, 1] = zeta[:, i_M+1, i_N]
            zeta_v[:, 2] = zeta[:, i_M+1, i_N+1]
            zeta_v[:, 3] = zeta[:, i_M, i_N+1]
            zeta_v = zeta_v.reshape([3, 1, 4])

            for v1 in range(4):
                v2 = (v1 + 1) % 4
                r_0l = zeta_v[:, :, v2] - zeta_v[:, :, v1]
                r_1l = pnt - zeta_v[:, :, v1]
                r_2l = pnt - zeta_v[:, :, v2]

                # w_ind = w_ind + biot_savart_elem(r_0l, r_1l, r_2l, gamma[i_M, i_N])
                w_ind = w_ind + biot_savart_elem(r_0l, r_1l, r_2l, 1)
                pass
    
    return w_ind/(4.0*np.pi)

def biot_savart_elem(r_0l: np.ndarray, r_1l: np.ndarray, r_2l: np.ndarray, gamma: float) -> np.ndarray:
    r1l_unit = r_1l/l2norm(r_1l)
    r2l_unit = r_2l/l2norm(r_2l)
    r1l_s = skew(r_1l)
    return np.squeeze(r_0l.T @ (r1l_unit - r2l_unit) * (r1l_s @ r_2l))/np.power(l2norm(r1l_s @ r_2l), 2) * gamma

def skew(a: np.array) -> np.ndarray:
    a = np.squeeze(a)
    return np.array([[0.0, -a[2], a[1]], 
                     [a[2], 0.0, -a[0]],
                     [-a[1], a[0], 0.0]])

def l2norm(a: np.ndarray) -> float:
    return np.sqrt(np.sum(np.power(a, 2)))

def extract_zeta_gamma(case_data, tstep) -> tuple[np.ndarray, np.ndarray]:
    zeta = np.concatenate([case_data.aero.timestep_info[tstep].zeta[0],\
                            case_data.aero.timestep_info[tstep].zeta_star[0]], axis=1)
    gamma = np.concatenate([case_data.aero.timestep_info[tstep].gamma[0],\
                            case_data.aero.timestep_info[tstep].gamma_star[0]], axis=0)
    
    return [zeta, gamma]

def plot_points(case_data, pnts: np.ndarray, tstep: int):
    zeta_b = case_data.aero.timestep_info[tstep].zeta[0]
    zeta_w = case_data.aero.timestep_info[tstep].zeta_star[0]
    
    Mb = zeta_b.shape[1]
    Mw = zeta_w.shape[1]
    N = zeta_b.shape[2]
    n_pnts = pnts.shape[1]

    ax = plt.figure().add_subplot(projection='3d')

    for i_N in range(N):
        ax.plot(zeta_b[0, :, i_N], zeta_b[1, :, i_N], zeta_b[2, :, i_N], 'k-')
        ax.plot(zeta_w[0, :, i_N], zeta_w[1, :, i_N], zeta_w[2, :, i_N], 'r-')

    for i_Mb in range(Mb):
        ax.plot(zeta_b[0, i_Mb, :], zeta_b[1, i_Mb, :], zeta_b[2, i_Mb, :], 'k-')

    for i_Mw in range(Mw):
        ax.plot(zeta_w[0, i_Mw, :], zeta_w[1, i_Mw, :], zeta_w[2, i_Mw, :], 'r-')

    for i_pnt in range(n_pnts):
        ax.plot(pnts[0, i_pnt], pnts[1, i_pnt], pnts[2, i_pnt], 'b.')

    ax.axis('equal')
    ax.view_init(elev=90, azim=0, roll=0)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.show()