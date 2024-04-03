from numba import jit, prange
import numpy as np
import time


@jit(nopython=True, parallel=True, fastmath=True, nogil=True, cache=True)
def ori_to_fz_C2_nb_inplace(qu: np.ndarray) -> np.ndarray:
    for i in prange(qu.shape[0]):
        w, z_abs = qu[i, 0], abs(qu[i, 3])
        if z_abs < w:
            continue
        else:
            qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                -qu[i, 3],
                qu[i, 2],
                -qu[i, 1],
                qu[i, 0],
            )
            if qu[i, 0] < 0:
                qu[i, :] = -qu[i, :]
    return qu


@jit(nopython=True, parallel=True, fastmath=True, nogil=True, cache=True)
def ori_to_fz_D2_nb_inplace(qu: np.ndarray) -> np.ndarray:
    for i in prange(qu.shape[0]):
        x_abs, y_abs, z_abs = abs(qu[i, 1]), abs(qu[i, 2]), abs(qu[i, 3])
        if x_abs < qu[i, 0] and y_abs < qu[i, 0] and z_abs < qu[i, 0]:
            continue
        elif z_abs > y_abs and z_abs > x_abs:
            # 180 about Z
            qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                -qu[i, 3],
                qu[i, 2],
                -qu[i, 1],
                qu[i, 0],
            )
        elif y_abs > x_abs:
            # 180 about Y
            qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                -qu[i, 2],
                -qu[i, 3],
                qu[i, 0],
                qu[i, 1],
            )
        else:
            # 180 about X
            qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                -qu[i, 1],
                qu[i, 0],
                qu[i, 3],
                -qu[i, 2],
            )
        if qu[i, 0] < 0:
            qu[i, :] = -qu[i, :]
    return qu


@jit(nopython=True, parallel=True, fastmath=True, nogil=True, cache=True)
def ori_to_fz_C4_nb_inplace(qu: np.ndarray) -> np.ndarray:
    TAN_22_5 = 2.0**0.5 - 1.0
    TAN_67_5 = 2.0**0.5 + 1.0
    R2 = 1.0 / (2.0**0.5)
    for i in prange(qu.shape[0]):
        w, z_abs = qu[i, 0], abs(qu[i, 3])
        # Not in RFZ
        if z_abs < (TAN_22_5 * w):
            continue
        elif z_abs > (TAN_67_5 * w):
            # Need 180 about Z
            qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                -qu[i, 3],
                qu[i, 2],
                -qu[i, 1],
                qu[i, 0],
            )
        # Need +/-90 about Z
        # +90 about Z
        elif qu[i, 3] < 0:
            qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                R2 * qu[i, 0] - R2 * qu[i, 3],
                R2 * qu[i, 1] + R2 * qu[i, 2],
                R2 * qu[i, 2] - R2 * qu[i, 1],
                R2 * qu[i, 3] + R2 * qu[i, 0],
            )
        # -90 about Z
        else:
            qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                R2 * qu[i, 0] + R2 * qu[i, 3],
                R2 * qu[i, 1] - R2 * qu[i, 2],
                R2 * qu[i, 2] + R2 * qu[i, 1],
                R2 * qu[i, 3] - R2 * qu[i, 0],
            )
        if qu[i, 0] < 0:
            qu[i, :] = -qu[i, :]
    return qu


@jit(nopython=True, parallel=True, fastmath=True, nogil=True, cache=True)
def ori_to_fz_D4_nb_inplace(qu: np.ndarray) -> np.ndarray:
    TAN_22_5 = 2.0**0.5 - 1.0
    TAN_67_5 = 2.0**0.5 + 1.0
    R2 = 1.0 / (2.0**0.5)
    for i in prange(qu.shape[0]):
        w, x, y, z = qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3]
        z_abs = abs(z)
        # SLOW REFERENCE IMPLEMENTATION USING ATAN2
        # angle = (np.arctan2(x, y) + (np.pi / 8.0)) % np.pi
        # if angle <= np.pi / 4.0:
        #     # 0 about Z
        #     y_rot = abs(y)
        # elif (angle <= 3.0 * np.pi / 4.0) and (angle > np.pi / 2.0):
        #     # 90 about Z
        #     y_rot = abs(x)
        # else:
        #     # 135 about Z
        #     y_rot = R2 * abs(y) + R2 * abs(x)
        # LOOKING AT 3D RF-VECTOR PLOTS FROM ORIGIN TOWARDS
        # +Z WITH +Y ON MY RIGHT AND +X UPWARDS

        # LESS SLOW BUT NOT AS FAST AS POSSIBLE IMPLEMENTATION
        # tan_angle = x / y
        # if tan_angle > -TAN_22_5 and tan_angle < TAN_22_5:
        #     y_rot = abs(y)
        #     angle_zone = 0
        # elif tan_angle > TAN_22_5 and tan_angle < TAN_67_5:
        #     y_rot = R2 * abs(y) + R2 * abs(x)
        #     angle_zone = 1
        # elif abs(tan_angle) > TAN_67_5:
        #     y_rot = abs(x)
        #     angle_zone = 2
        # else:
        #     y_rot = R2 * abs(y) + R2 * abs(x)
        #     angle_zone = 3

        # FASTEST IMPLEMENTATION
        if abs(y) > abs(x):
            y_larger = True
            if abs(y) > TAN_67_5 * abs(x):
                y_rot = abs(y)
                y_zone = True
            else:
                y_rot = R2 * abs(y) + R2 * abs(x)
                y_zone = False
        else:
            y_larger = False
            if abs(x) > TAN_67_5 * abs(y):
                y_rot = abs(x)
                y_zone = True
            else:
                y_rot = R2 * abs(x) + R2 * abs(y)
                y_zone = False

        # in RFZ?
        if z_abs < (TAN_22_5 * w) and y_rot < w:
            continue
        # 180 about Z
        elif (z_abs > (TAN_67_5 * w)) and (z_abs > y_rot):
            qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                -qu[i, 3],
                qu[i, 2],
                -qu[i, 1],
                qu[i, 0],
            )
        # +/-90 about Z
        elif (z_abs > (TAN_22_5 * w)) and ((z_abs * R2 + R2 * w) > y_rot):
            if z < 0:
                qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                    R2 * qu[i, 0] - R2 * qu[i, 3],
                    R2 * qu[i, 1] + R2 * qu[i, 2],
                    R2 * qu[i, 2] - R2 * qu[i, 1],
                    R2 * qu[i, 3] + R2 * qu[i, 0],
                )
            else:
                qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                    R2 * qu[i, 0] + R2 * qu[i, 3],
                    R2 * qu[i, 1] - R2 * qu[i, 2],
                    R2 * qu[i, 2] + R2 * qu[i, 1],
                    R2 * qu[i, 3] - R2 * qu[i, 0],
                )
        # not a simple z axis rotation
        else:
            if y_zone:
                if y_larger:
                    # zone 3 apply [0.0, 0.0, 1.0, 0.0]
                    qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                        -qu[i, 2],
                        -qu[i, 3],
                        qu[i, 0],
                        qu[i, 1],
                    )
                else:
                    # zone 2 apply [0.0, 1.0, 0.0, 0.0]
                    qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                        -qu[i, 1],
                        qu[i, 0],
                        qu[i, 3],
                        -qu[i, 2],
                    )
            else:
                if (x * y) >= 0:
                    # zone 6 apply [0.0, R2, R2, 0.0]
                    qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                        -R2 * qu[i, 1] - R2 * qu[i, 2],
                        R2 * qu[i, 0] - R2 * qu[i, 3],
                        R2 * qu[i, 3] + R2 * qu[i, 0],
                        -R2 * qu[i, 2] + R2 * qu[i, 1],
                    )
                else:
                    # zone 7 apply [0.0, -R2, R2, 0.0]
                    qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                        R2 * qu[i, 1] - R2 * qu[i, 2],
                        -R2 * qu[i, 0] - R2 * qu[i, 3],
                        -R2 * qu[i, 3] + R2 * qu[i, 0],
                        R2 * qu[i, 2] + R2 * qu[i, 1],
                    )
            # if angle_zone == 0:
            #     # 180 about Y
            #     qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
            #         -qu[i, 2],
            #         -qu[i, 3],
            #         qu[i, 0],
            #         qu[i, 1],
            #     )
            # elif angle_zone == 1:
            #     # +90 about XY
            #     qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
            #         -R2 * qu[i, 1] - R2 * qu[i, 2],
            #         R2 * qu[i, 0] - R2 * qu[i, 3],
            #         R2 * qu[i, 3] + R2 * qu[i, 0],
            #         -R2 * qu[i, 2] + R2 * qu[i, 1],
            #     )
            # elif angle_zone == 2:
            #     # 180 about X
            #     qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
            #         -qu[i, 1],
            #         qu[i, 0],
            #         qu[i, 3],
            #         -qu[i, 2],
            #     )
            # else:
            #     # -90 about XY
            #     qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
            #         R2 * qu[i, 1] - R2 * qu[i, 2],
            #         -R2 * qu[i, 0] - R2 * qu[i, 3],
            #         -R2 * qu[i, 3] + R2 * qu[i, 0],
            #         R2 * qu[i, 2] + R2 * qu[i, 1],
            #     )
        if qu[i, 0] < 0:
            qu[i, :] = -qu[i, :]
    return qu


@jit(nopython=True, parallel=True, fastmath=True, nogil=True, cache=True)
def ori_to_fz_C3_nb_inplace(qu: np.ndarray) -> np.ndarray:
    TAN_30 = 1.0 / (3.0**0.5)
    R3 = (3.0**0.5) / 2.0
    for i in prange(qu.shape[0]):
        w, z = qu[i, 0], qu[i, 3]
        if np.abs(z) < TAN_30 * w:
            continue
        elif z > 0:
            qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                0.5 * qu[i, 0] + R3 * qu[i, 3],
                0.5 * qu[i, 1] - R3 * qu[i, 2],
                0.5 * qu[i, 2] + R3 * qu[i, 1],
                0.5 * qu[i, 3] - R3 * qu[i, 0],
            )
        else:
            qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                0.5 * qu[i, 0] - R3 * qu[i, 3],
                0.5 * qu[i, 1] + R3 * qu[i, 2],
                0.5 * qu[i, 2] - R3 * qu[i, 1],
                0.5 * qu[i, 3] + R3 * qu[i, 0],
            )
        if qu[i, 0] < 0:
            qu[i, :] = -qu[i, :]
    return qu


@jit(nopython=True, parallel=True, fastmath=True, nogil=True, cache=True)
def ori_to_fz_D3_nb_inplace(qu: np.ndarray) -> np.ndarray:
    R3 = (3.0**0.5) / 2.0
    TAN_30 = 1.0 / (3.0**0.5)
    for i in prange(qu.shape[0]):
        w, x, y, z = qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3]
        z_abs = abs(z)

        # SLOW REFERENCE IMPLEMENTATION USING ATAN2
        # angle = np.arctan2(x, y) % np.pi
        # # three angle mod_60 cases
        # if (angle > (np.pi / 3.0)) and (angle < (2.0 * np.pi / 3.0)):
        #     x_rot = abs(x)
        # else:
        #     x_rot = R3 * abs(y) + 0.5 * abs(x)

        tan_ang = x / y
        if abs(tan_ang) > (3.0**0.5):
            x_rot = abs(x)
            angle_zone = 1
        else:
            x_rot = R3 * abs(y) + 0.5 * abs(x)
            if tan_ang > 0:
                angle_zone = 0
            else:
                angle_zone = 2
        # RFZ condition
        if (z_abs < (TAN_30 * w)) and (x_rot < w):
            continue
        else:
            # polar closer
            if (0.5 * w + R3 * z_abs) > x_rot:
                if z < 0:
                    # [0.5, 0.0, 0.0, R3],
                    qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                        0.5 * qu[i, 0] - R3 * qu[i, 3],
                        0.5 * qu[i, 1] + R3 * qu[i, 2],
                        0.5 * qu[i, 2] - R3 * qu[i, 1],
                        0.5 * qu[i, 3] + R3 * qu[i, 0],
                    )
                else:
                    # [0.5, 0.0, 0.0, -R3],
                    qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                        0.5 * qu[i, 0] + R3 * qu[i, 3],
                        0.5 * qu[i, 1] - R3 * qu[i, 2],
                        0.5 * qu[i, 2] + R3 * qu[i, 1],
                        0.5 * qu[i, 3] - R3 * qu[i, 0],
                    )
            # z-radial zones
            else:
                # if angle < (np.pi / 3.0):
                if angle_zone == 0:
                    # [0.0, 0.5, R3, 0.0] Zone
                    qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                        -0.5 * qu[i, 1] - R3 * qu[i, 2],
                        0.5 * qu[i, 0] - R3 * qu[i, 3],
                        0.5 * qu[i, 3] + R3 * qu[i, 0],
                        -0.5 * qu[i, 2] + R3 * qu[i, 1],
                    )
                # elif angle < (2.0 * np.pi / 3.0):
                elif angle_zone == 1:
                    # [0.0, 1.0, 0.0, 0.0],
                    qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                        -qu[i, 1],
                        qu[i, 0],
                        qu[i, 3],
                        -qu[i, 2],
                    )
                else:
                    # [0.0, -0.5, R3, 0.0]
                    qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                        0.5 * qu[i, 1] - R3 * qu[i, 2],
                        -0.5 * qu[i, 0] - R3 * qu[i, 3],
                        -0.5 * qu[i, 3] + R3 * qu[i, 0],
                        0.5 * qu[i, 2] + R3 * qu[i, 1],
                    )
            if qu[i, 0] < 0:
                qu[i, :] = -qu[i, :]
    return qu


@jit(nopython=True, parallel=True, fastmath=True, nogil=True, cache=True)
def ori_to_fz_C6_nb_inplace(qu: np.ndarray) -> np.ndarray:
    R3 = (3.0**0.5) / 2.0
    TAN75 = 2 + 3**0.5
    TAN15 = 2 - 3.0**0.5
    for i in prange(qu.shape[0]):
        w, z = qu[i, 0], qu[i, 3]
        z_abs = abs(z)
        if z_abs <= TAN15 * w:
            continue
        elif z_abs > TAN75 * w:
            qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                -qu[i, 3],
                qu[i, 2],
                -qu[i, 1],
                qu[i, 0],
            )
        elif z_abs > w:
            if z < 0.0:
                qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                    0.5 * qu[i, 0] - R3 * qu[i, 3],
                    0.5 * qu[i, 1] + R3 * qu[i, 2],
                    0.5 * qu[i, 2] - R3 * qu[i, 1],
                    0.5 * qu[i, 3] + R3 * qu[i, 0],
                )
            else:
                qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                    0.5 * qu[i, 0] + R3 * qu[i, 3],
                    0.5 * qu[i, 1] - R3 * qu[i, 2],
                    0.5 * qu[i, 2] + R3 * qu[i, 1],
                    0.5 * qu[i, 3] - R3 * qu[i, 0],
                )
        else:
            if z < 0.0:
                qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                    R3 * qu[i, 0] - 0.5 * qu[i, 3],
                    R3 * qu[i, 1] + 0.5 * qu[i, 2],
                    R3 * qu[i, 2] - 0.5 * qu[i, 1],
                    R3 * qu[i, 3] + 0.5 * qu[i, 0],
                )
            else:
                qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                    R3 * qu[i, 0] + 0.5 * qu[i, 3],
                    R3 * qu[i, 1] - 0.5 * qu[i, 2],
                    R3 * qu[i, 2] + 0.5 * qu[i, 1],
                    R3 * qu[i, 3] - 0.5 * qu[i, 0],
                )
        if qu[i, 0] < 0.0:
            qu[i, :] = -qu[i, :]
    return qu


@jit(nopython=True, parallel=True, fastmath=True, nogil=True, cache=True)
def ori_to_fz_D6_nb_inplace(qu: np.ndarray) -> np.ndarray:
    TAN75 = 2 + 3**0.5
    TAN15 = 2 - 3.0**0.5
    R3 = (3.0**0.5) / 2.0
    for i in prange(qu.shape[0]):
        w, x, y, z = qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3]
        z_abs = abs(z)
        if abs(y) > abs(x):
            y_new = abs(y)
            x_new = abs(x)
            y_larger = True
        else:
            y_new = abs(x)
            x_new = abs(y)
            y_larger = False
        # check if y/x > tan(75) -> y > tan(75) * x
        if y_new > TAN75 * x_new:
            y_rot = y_new
            y_zone = True
        else:
            y_rot = R3 * y_new + 0.5 * x_new
            y_zone = False

        # SLOW REFERENCE IMPLEMENTATION USING ATAN2
        # angle = (np.arctan2(x, y) + (np.pi / 12.0)) % np.pi
        # if angle <= np.pi / 6.0:
        #     y_rot = abs(y)
        # elif angle <= np.pi / 3.0:
        #     y_rot = 0.5 * abs(x) + R3 * abs(y)
        # elif angle <= np.pi / 2.0:
        #     y_rot = R3 * abs(x) + 0.5 * abs(y)
        # elif angle <= 2.0 * np.pi / 3.0:
        #     y_rot = abs(x)
        # elif angle <= 5.0 * np.pi / 6.0:
        #     y_rot = R3 * abs(x) + 0.5 * abs(y)
        # else:
        #     y_rot = 0.5 * abs(x) + R3 * abs(y)

        # tan_angle = x / y
        # if abs(tan_angle) < TAN15:
        #     y_rot = abs(y)
        #     angle_zone = 0
        # elif abs(tan_angle) > TAN15 and abs(tan_angle) < 1:
        #     y_rot = 0.5 * abs(x) + R3 * abs(y)
        #     if tan_angle > 0:
        #         angle_zone = 1
        #     else:
        #         angle_zone = 5
        # elif abs(tan_angle) > 1 and abs(tan_angle) < TAN75:
        #     y_rot = R3 * abs(x) + 0.5 * abs(y)
        #     if tan_angle > 0:
        #         angle_zone = 2
        #     else:
        #         angle_zone = 4
        # else:
        #     y_rot = abs(x)
        #     angle_zone = 3

        # in RFZ?
        if z_abs < (TAN15 * w) and (y_rot < w):
            continue
        elif (z_abs < w) and (0.5 * z_abs + R3 * w) > y_rot:
            if z < 0:
                # [R3, 0.0, 0.0, 0.5]
                qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                    R3 * qu[i, 0] - 0.5 * qu[i, 3],
                    R3 * qu[i, 1] + 0.5 * qu[i, 2],
                    R3 * qu[i, 2] - 0.5 * qu[i, 1],
                    R3 * qu[i, 3] + 0.5 * qu[i, 0],
                )
            else:
                # [R3, 0.0, 0.0, -0.5]
                qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                    R3 * qu[i, 0] + 0.5 * qu[i, 3],
                    R3 * qu[i, 1] - 0.5 * qu[i, 2],
                    R3 * qu[i, 2] + 0.5 * qu[i, 1],
                    R3 * qu[i, 3] - 0.5 * qu[i, 0],
                )
        elif z_abs < (TAN75 * w) and (R3 * z_abs + 0.5 * w) > y_rot:
            if z < 0:
                # [0.5, 0.0, 0.0, R3]
                qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                    0.5 * qu[i, 0] - R3 * qu[i, 3],
                    0.5 * qu[i, 1] + R3 * qu[i, 2],
                    0.5 * qu[i, 2] - R3 * qu[i, 1],
                    0.5 * qu[i, 3] + R3 * qu[i, 0],
                )
            else:
                # [0.5, 0.0, 0.0, -R3]
                qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                    0.5 * qu[i, 0] + R3 * qu[i, 3],
                    0.5 * qu[i, 1] - R3 * qu[i, 2],
                    0.5 * qu[i, 2] + R3 * qu[i, 1],
                    0.5 * qu[i, 3] - R3 * qu[i, 0],
                )
        elif z_abs > y_rot:
            # [0.0, 0.0, 0.0, 1.0]
            qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                -qu[i, 3],
                qu[i, 2],
                -qu[i, 1],
                qu[i, 0],
            )
        else:
            if y_zone:
                if y_larger:
                    # zone 11 apply [0.0, 0.0, 1.0, 0.0]
                    qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                        -qu[i, 2],
                        -qu[i, 3],
                        qu[i, 0],
                        qu[i, 1],
                    )
                else:
                    # zone 6 apply [0.0, 1.0, 0.0, 0.0]
                    qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                        -qu[i, 1],
                        qu[i, 0],
                        qu[i, 3],
                        -qu[i, 2],
                    )
            else:
                if (x * y) >= 0:
                    if y_larger:
                        # zone 8 apply [0.0, 0.5, R3, 0.0]
                        qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                            -0.5 * qu[i, 1] - R3 * qu[i, 2],
                            0.5 * qu[i, 0] - R3 * qu[i, 3],
                            0.5 * qu[i, 3] + R3 * qu[i, 0],
                            -0.5 * qu[i, 2] + R3 * qu[i, 1],
                        )
                    else:
                        # zone 9 apply [0.0, R3, 0.5, 0.0]
                        qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                            -R3 * qu[i, 1] - 0.5 * qu[i, 2],
                            R3 * qu[i, 0] - 0.5 * qu[i, 3],
                            R3 * qu[i, 3] + 0.5 * qu[i, 0],
                            -R3 * qu[i, 2] + 0.5 * qu[i, 1],
                        )
                else:
                    if y_larger:
                        # zone 7 apply [0.0, -0.5, R3, 0.0]
                        qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                            0.5 * qu[i, 1] - R3 * qu[i, 2],
                            -0.5 * qu[i, 0] - R3 * qu[i, 3],
                            -0.5 * qu[i, 3] + R3 * qu[i, 0],
                            0.5 * qu[i, 2] + R3 * qu[i, 1],
                        )
                    else:
                        # zone 10 apply [0.0, -R3, 0.5, 0.0]
                        qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                            R3 * qu[i, 1] - 0.5 * qu[i, 2],
                            -R3 * qu[i, 0] - 0.5 * qu[i, 3],
                            -R3 * qu[i, 3] + 0.5 * qu[i, 0],
                            R3 * qu[i, 2] + 0.5 * qu[i, 1],
                        )
        # # must be a z-radial zone
        # # elif angle <= np.pi / 6.0:
        # elif angle_zone == 0:
        #     # [0.0, 0.0, 1.0, 0.0]
        #     qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
        #         -qu[i, 2],
        #         -qu[i, 3],
        #         qu[i, 0],
        #         qu[i, 1],
        #     )
        # # elif angle <= np.pi / 3.0:
        # elif angle_zone == 1:
        #     # [0.0, 0.5, R3, 0.0]
        #     qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
        #         -0.5 * qu[i, 1] - R3 * qu[i, 2],
        #         0.5 * qu[i, 0] - R3 * qu[i, 3],
        #         0.5 * qu[i, 3] + R3 * qu[i, 0],
        #         -0.5 * qu[i, 2] + R3 * qu[i, 1],
        #     )
        # # elif angle <= np.pi / 2.0:
        # elif angle_zone == 2:
        #     # [0.0, R3, 0.5, 0.0]
        #     qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
        #         -R3 * qu[i, 1] - 0.5 * qu[i, 2],
        #         R3 * qu[i, 0] - 0.5 * qu[i, 3],
        #         R3 * qu[i, 3] + 0.5 * qu[i, 0],
        #         -R3 * qu[i, 2] + 0.5 * qu[i, 1],
        #     )
        # # elif angle <= 2.0 * np.pi / 3.0:
        # elif angle_zone == 3:
        #     # [0.0, 1.0, 0.0, 0.0]
        #     qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
        #         -qu[i, 1],
        #         qu[i, 0],
        #         qu[i, 3],
        #         -qu[i, 2],
        #     )
        # # elif angle <= 5.0 * np.pi / 6.0:
        # elif angle_zone == 4:
        #     # [0.0, -R3, 0.5, 0.0]
        #     qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
        #         R3 * qu[i, 1] - 0.5 * qu[i, 2],
        #         -R3 * qu[i, 0] - 0.5 * qu[i, 3],
        #         -R3 * qu[i, 3] + 0.5 * qu[i, 0],
        #         R3 * qu[i, 2] + 0.5 * qu[i, 1],
        #     )
        # else:
        #     # [0.0, -0.5, R3, 0.0]
        #     qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
        #         0.5 * qu[i, 1] - R3 * qu[i, 2],
        #         -0.5 * qu[i, 0] - R3 * qu[i, 3],
        #         -0.5 * qu[i, 3] + R3 * qu[i, 0],
        #         0.5 * qu[i, 2] + R3 * qu[i, 1],
        #     )
        if qu[i, 0] < 0:
            qu[i, :] = -qu[i, :]
    return qu


@jit(nopython=True, parallel=True, fastmath=True, nogil=True, cache=True)
def ori_to_fz_T_nb_inplace(qu: np.ndarray) -> np.ndarray:
    for i in prange(qu.shape[0]):
        w, x, y, z = qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3]
        x_abs, y_abs, z_abs = abs(x), abs(y), abs(z)
        if x_abs + y_abs + z_abs < w:
            continue
        elif z_abs > (w + x_abs + y_abs):
            # [0.0, 0.0, 0.0, 1.0]
            qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                -qu[i, 3],
                qu[i, 2],
                -qu[i, 1],
                qu[i, 0],
            )
        elif x_abs > (w + y_abs + z_abs):
            # [0.0, 1.0, 0.0, 0.0]
            qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                -qu[i, 1],
                qu[i, 0],
                qu[i, 3],
                -qu[i, 2],
            )
        elif y_abs > (w + x_abs + z_abs):
            # [0.0, 0.0, 1.0, 0.0]
            qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                -qu[i, 2],
                -qu[i, 3],
                qu[i, 0],
                qu[i, 1],
            )
        # Check each of the 8 octants
        else:
            if x < 0:
                if y < 0:
                    if z < 0:
                        qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                            0.5 * (qu[i, 0] - qu[i, 1] - qu[i, 2] - qu[i, 3]),
                            0.5 * (qu[i, 1] + qu[i, 0] - qu[i, 3] + qu[i, 2]),
                            0.5 * (qu[i, 2] + qu[i, 3] + qu[i, 0] - qu[i, 1]),
                            0.5 * (qu[i, 3] - qu[i, 2] + qu[i, 1] + qu[i, 0]),
                        )
                    else:
                        qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                            0.5 * (qu[i, 0] - qu[i, 1] - qu[i, 2] + qu[i, 3]),
                            0.5 * (qu[i, 1] + qu[i, 0] - qu[i, 3] - qu[i, 2]),
                            0.5 * (qu[i, 2] + qu[i, 3] + qu[i, 0] + qu[i, 1]),
                            0.5 * (qu[i, 3] - qu[i, 2] + qu[i, 1] - qu[i, 0]),
                        )
                else:
                    if z < 0:
                        qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                            0.5 * (qu[i, 0] - qu[i, 1] + qu[i, 2] - qu[i, 3]),
                            0.5 * (qu[i, 1] + qu[i, 0] + qu[i, 3] + qu[i, 2]),
                            0.5 * (qu[i, 2] + qu[i, 3] - qu[i, 0] - qu[i, 1]),
                            0.5 * (qu[i, 3] - qu[i, 2] - qu[i, 1] + qu[i, 0]),
                        )
                    else:
                        qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                            0.5 * (qu[i, 0] - qu[i, 1] + qu[i, 2] + qu[i, 3]),
                            0.5 * (qu[i, 1] + qu[i, 0] + qu[i, 3] - qu[i, 2]),
                            0.5 * (qu[i, 2] + qu[i, 3] - qu[i, 0] + qu[i, 1]),
                            0.5 * (qu[i, 3] - qu[i, 2] - qu[i, 1] - qu[i, 0]),
                        )
            else:
                if y < 0:
                    if z < 0:
                        qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                            0.5 * (qu[i, 0] + qu[i, 1] - qu[i, 2] - qu[i, 3]),
                            0.5 * (qu[i, 1] - qu[i, 0] - qu[i, 3] + qu[i, 2]),
                            0.5 * (qu[i, 2] - qu[i, 3] + qu[i, 0] - qu[i, 1]),
                            0.5 * (qu[i, 3] + qu[i, 2] + qu[i, 1] + qu[i, 0]),
                        )
                    else:
                        qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                            0.5 * (qu[i, 0] + qu[i, 1] - qu[i, 2] + qu[i, 3]),
                            0.5 * (qu[i, 1] - qu[i, 0] - qu[i, 3] - qu[i, 2]),
                            0.5 * (qu[i, 2] - qu[i, 3] + qu[i, 0] + qu[i, 1]),
                            0.5 * (qu[i, 3] + qu[i, 2] + qu[i, 1] - qu[i, 0]),
                        )
                else:
                    if z < 0:
                        qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                            0.5 * (qu[i, 0] + qu[i, 1] + qu[i, 2] - qu[i, 3]),
                            0.5 * (qu[i, 1] - qu[i, 0] + qu[i, 3] + qu[i, 2]),
                            0.5 * (qu[i, 2] - qu[i, 3] - qu[i, 0] - qu[i, 1]),
                            0.5 * (qu[i, 3] + qu[i, 2] - qu[i, 1] + qu[i, 0]),
                        )
                    else:
                        qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                            0.5 * (qu[i, 0] + qu[i, 1] + qu[i, 2] + qu[i, 3]),
                            0.5 * (qu[i, 1] - qu[i, 0] + qu[i, 3] - qu[i, 2]),
                            0.5 * (qu[i, 2] - qu[i, 3] - qu[i, 0] + qu[i, 1]),
                            0.5 * (qu[i, 3] + qu[i, 2] - qu[i, 1] - qu[i, 0]),
                        )
        if qu[i, 0] < 0:
            qu[i, :] = -qu[i, :]
    return qu


@jit(nopython=True, parallel=True, fastmath=True, nogil=True, cache=True)
def ori_to_fz_O_nb_inplace(qu: np.ndarray) -> np.ndarray:
    R2 = 2.0**-0.5
    for i in prange(qu.shape[0]):
        w, x, y, z = qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3]
        x_abs, y_abs, z_abs = abs(x), abs(y), abs(z)
        # reorder x_abs, y_abs, z_abs in increasing order
        if x_abs < y_abs:
            if y_abs < z_abs:
                x_swap, y_swap, z_swap = x_abs, y_abs, z_abs
                largest = 2
                smallest = 0
            else:
                if x_abs < z_abs:
                    x_swap, y_swap, z_swap = x_abs, z_abs, y_abs
                    largest = 1
                    smallest = 0
                else:
                    x_swap, y_swap, z_swap = z_abs, x_abs, y_abs
                    largest = 1
                    smallest = 2
        else:
            if x_abs < z_abs:
                x_swap, y_swap, z_swap = y_abs, x_abs, z_abs
                largest = 2
                smallest = 1
            else:
                if y_abs < z_abs:
                    x_swap, y_swap, z_swap = y_abs, z_abs, x_abs
                    largest = 0
                    smallest = 1
                else:
                    x_swap, y_swap, z_swap = z_abs, y_abs, x_abs
                    largest = 0
                    smallest = 2
        # start with top plane of RFZ
        if z_swap < (2**0.5 - 1) * w:
            # Below RFZ height: in 0 or 11 type
            if w > (x_swap + y_swap + z_swap):
                # in O RFZ
                zone_type = 0
                continue
            else:
                # in a type 11 zone
                zone_type = 11
        # next threshold z between zone 1 and 5
        elif z_swap < (2**0.5 + 1) * w:
            # in 5, 11, or 22 ... closer to 5 than 11?
            # (R2 * w + R2 * z_swap) > 0.5 * (w + x_swap + y_swap + z_swap):
            # simplified:
            if (2**0.5 - 1) * (w + z_swap) > (x_swap + y_swap):
                # closer to 5 than 22?
                # (R2 * w + R2 * z_swap) > (R2 * z_swap + R2 * y_swap):
                # simplified:
                if w > y_swap:
                    zone_type = 5
                else:
                    zone_type = 22
            else:
                # closer to 11 than 22?
                # 0.5 * (w + x_swap + y_swap + z_swap) > (R2 * z_swap + R2 * y_swap):
                # simplified:
                if (w + x_swap) > (2**0.5 - 1) * (y_swap + z_swap):
                    zone_type = 11
                else:
                    zone_type = 22
        # Above the Zone 5 z threshold... in 1, 11, or 22
        else:
            # closer to 1 than 11?
            if z_swap > 0.5 * (w + x_swap + y_swap + z_swap):
                # closer to 1 than 22?
                # z_swap > (R2 * z_swap + R2 * y_swap):
                # simplified:
                if z_swap > (2**0.5 + 1) * y_swap:
                    zone_type = 1
                else:
                    zone_type = 22
            else:
                # closer to 11 than 22?
                # if 0.5 * (w + x_swap + y_swap + z_swap) > (R2 * z_swap + R2 * y_swap):
                # simplified:
                if (w + x_swap) > (2**0.5 - 1) * (y_swap + z_swap):
                    zone_type = 11
                else:
                    zone_type = 22
        # now we know the zone type
        if zone_type == 1:
            # three choices based on which was largest in magnitude
            if largest == 0:
                # Zone 2: apply [0.0, 1.0, 0.0, 0.0]
                qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                    -qu[i, 1],
                    qu[i, 0],
                    qu[i, 3],
                    -qu[i, 2],
                )
            elif largest == 1:
                # Zone 3: apply [0.0, 0.0, 1.0, 0.0]
                qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                    -qu[i, 2],
                    -qu[i, 3],
                    qu[i, 0],
                    qu[i, 1],
                )
            else:
                # Zone 1: apply [0.0, 0.0, 0.0, 1.0]
                qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                    -qu[i, 3],
                    qu[i, 2],
                    -qu[i, 1],
                    qu[i, 0],
                )
        elif zone_type == 5:
            # 6 choices based on largest magnitude of x,y,z & pos vs neg
            if largest == 0:
                if x < 0:
                    # Zone 16: apply [R2, R2, 0.0, 0.0]
                    qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                        R2 * qu[i, 0] - R2 * qu[i, 1],
                        R2 * qu[i, 1] + R2 * qu[i, 0],
                        R2 * qu[i, 2] + R2 * qu[i, 3],
                        R2 * qu[i, 3] - R2 * qu[i, 2],
                    )
                else:
                    # Zone 17: apply [R2, -R2, 0.0, 0.0]
                    qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                        R2 * qu[i, 0] + R2 * qu[i, 1],
                        R2 * qu[i, 1] - R2 * qu[i, 0],
                        R2 * qu[i, 2] - R2 * qu[i, 3],
                        R2 * qu[i, 3] + R2 * qu[i, 2],
                    )
            elif largest == 1:
                if y < 0:
                    # Zone 18: apply [R2, 0.0, R2, 0.0]
                    qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                        R2 * qu[i, 0] - R2 * qu[i, 2],
                        R2 * qu[i, 1] - R2 * qu[i, 3],
                        R2 * qu[i, 2] + R2 * qu[i, 0],
                        R2 * qu[i, 3] + R2 * qu[i, 1],
                    )
                else:
                    # Zone 19: apply [R2, 0.0, -R2, 0.0]
                    qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                        R2 * qu[i, 0] + R2 * qu[i, 2],
                        R2 * qu[i, 1] + R2 * qu[i, 3],
                        R2 * qu[i, 2] - R2 * qu[i, 0],
                        R2 * qu[i, 3] - R2 * qu[i, 1],
                    )
            else:
                # apply [R2, 0.0, 0.0, +/- R2]
                if z < 0:
                    # Zone 4: apply [R2, 0.0, 0.0, R2]
                    qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                        R2 * qu[i, 0] - R2 * qu[i, 3],
                        R2 * qu[i, 1] + R2 * qu[i, 2],
                        R2 * qu[i, 2] - R2 * qu[i, 1],
                        R2 * qu[i, 3] + R2 * qu[i, 0],
                    )
                else:
                    # Zone 5: apply [R2, 0.0, 0.0, -R2]
                    qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                        R2 * qu[i, 0] + R2 * qu[i, 3],
                        R2 * qu[i, 1] - R2 * qu[i, 2],
                        R2 * qu[i, 2] + R2 * qu[i, 1],
                        R2 * qu[i, 3] - R2 * qu[i, 0],
                    )
        elif zone_type == 11:
            # 8 different possible returning quaternions based on signs
            # [0.5, +/- 0.5, +/- 0.5, +/- 0.5]
            if x < 0:
                if y < 0:
                    if z < 0:
                        # Zone 15
                        qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                            0.5 * (qu[i, 0] - qu[i, 1] - qu[i, 2] - qu[i, 3]),
                            0.5 * (qu[i, 1] + qu[i, 0] - qu[i, 3] + qu[i, 2]),
                            0.5 * (qu[i, 2] + qu[i, 3] + qu[i, 0] - qu[i, 1]),
                            0.5 * (qu[i, 3] - qu[i, 2] + qu[i, 1] + qu[i, 0]),
                        )
                    else:
                        # Zone 9
                        qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                            0.5 * (qu[i, 0] - qu[i, 1] - qu[i, 2] + qu[i, 3]),
                            0.5 * (qu[i, 1] + qu[i, 0] - qu[i, 3] - qu[i, 2]),
                            0.5 * (qu[i, 2] + qu[i, 3] + qu[i, 0] + qu[i, 1]),
                            0.5 * (qu[i, 3] - qu[i, 2] + qu[i, 1] - qu[i, 0]),
                        )
                else:
                    if z < 0:
                        # Zone 8
                        qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                            0.5 * (qu[i, 0] - qu[i, 1] + qu[i, 2] - qu[i, 3]),
                            0.5 * (qu[i, 1] + qu[i, 0] + qu[i, 3] + qu[i, 2]),
                            0.5 * (qu[i, 2] + qu[i, 3] - qu[i, 0] - qu[i, 1]),
                            0.5 * (qu[i, 3] - qu[i, 2] - qu[i, 1] + qu[i, 0]),
                        )
                    else:
                        # Zone 10
                        qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                            0.5 * (qu[i, 0] - qu[i, 1] + qu[i, 2] + qu[i, 3]),
                            0.5 * (qu[i, 1] + qu[i, 0] + qu[i, 3] - qu[i, 2]),
                            0.5 * (qu[i, 2] + qu[i, 3] - qu[i, 0] + qu[i, 1]),
                            0.5 * (qu[i, 3] - qu[i, 2] - qu[i, 1] - qu[i, 0]),
                        )
            else:
                if y < 0:
                    if z < 0:
                        # Zone 12
                        qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                            0.5 * (qu[i, 0] + qu[i, 1] - qu[i, 2] - qu[i, 3]),
                            0.5 * (qu[i, 1] - qu[i, 0] - qu[i, 3] + qu[i, 2]),
                            0.5 * (qu[i, 2] - qu[i, 3] + qu[i, 0] - qu[i, 1]),
                            0.5 * (qu[i, 3] + qu[i, 2] + qu[i, 1] + qu[i, 0]),
                        )
                    else:
                        # Zone 13
                        qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                            0.5 * (qu[i, 0] + qu[i, 1] - qu[i, 2] + qu[i, 3]),
                            0.5 * (qu[i, 1] - qu[i, 0] - qu[i, 3] - qu[i, 2]),
                            0.5 * (qu[i, 2] - qu[i, 3] + qu[i, 0] + qu[i, 1]),
                            0.5 * (qu[i, 3] + qu[i, 2] + qu[i, 1] - qu[i, 0]),
                        )
                else:
                    if z < 0:
                        # Zone 14
                        qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                            0.5 * (qu[i, 0] + qu[i, 1] + qu[i, 2] - qu[i, 3]),
                            0.5 * (qu[i, 1] - qu[i, 0] + qu[i, 3] + qu[i, 2]),
                            0.5 * (qu[i, 2] - qu[i, 3] - qu[i, 0] - qu[i, 1]),
                            0.5 * (qu[i, 3] + qu[i, 2] - qu[i, 1] + qu[i, 0]),
                        )
                    else:
                        # Zone 11
                        qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                            0.5 * (qu[i, 0] + qu[i, 1] + qu[i, 2] + qu[i, 3]),
                            0.5 * (qu[i, 1] - qu[i, 0] + qu[i, 3] - qu[i, 2]),
                            0.5 * (qu[i, 2] - qu[i, 3] - qu[i, 0] + qu[i, 1]),
                            0.5 * (qu[i, 3] + qu[i, 2] - qu[i, 1] - qu[i, 0]),
                        )
        # Zone 22 type
        else:
            # 6 choices based on:
            # smallest magnitude among xyz and...
            # sign sameness of remaining two values
            if smallest == 2:
                # if (x * y) > 0:
                if ((x < 0) and (y < 0)) or ((x >= 0) and (y >= 0)):
                    # Zone 6: apply [0.0, R2, R2, 0.0]
                    qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                        -R2 * qu[i, 1] - R2 * qu[i, 2],
                        R2 * qu[i, 0] - R2 * qu[i, 3],
                        R2 * qu[i, 3] + R2 * qu[i, 0],
                        -R2 * qu[i, 2] + R2 * qu[i, 1],
                    )
                else:
                    # Zone 7: apply [0.0, -R2, R2, 0.0]
                    qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                        R2 * qu[i, 1] - R2 * qu[i, 2],
                        -R2 * qu[i, 0] - R2 * qu[i, 3],
                        -R2 * qu[i, 3] + R2 * qu[i, 0],
                        R2 * qu[i, 2] + R2 * qu[i, 1],
                    )
            elif smallest == 1:
                # if (x * z) > 0:
                if ((x < 0) and (z < 0)) or ((x >= 0) and (z >= 0)):
                    # Zone 20: apply [0.0, R2, 0.0, R2]
                    qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                        -R2 * qu[i, 1] - R2 * qu[i, 3],
                        R2 * qu[i, 0] + R2 * qu[i, 2],
                        R2 * qu[i, 3] - R2 * qu[i, 1],
                        -R2 * qu[i, 2] + R2 * qu[i, 0],
                    )
                else:
                    # Zone 21: apply [0.0, -R2, 0.0, R2]
                    qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                        R2 * qu[i, 1] - R2 * qu[i, 3],
                        -R2 * qu[i, 0] + R2 * qu[i, 2],
                        -R2 * qu[i, 3] - R2 * qu[i, 1],
                        R2 * qu[i, 2] + R2 * qu[i, 0],
                    )
            else:
                # if (y * z) > 0:
                if ((y < 0) and (z < 0)) or ((y >= 0) and (z >= 0)):
                    # Zone 22: apply [0.0, 0.0, R2, R2]
                    qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                        -R2 * qu[i, 2] - R2 * qu[i, 3],
                        -R2 * qu[i, 3] + R2 * qu[i, 2],
                        R2 * qu[i, 0] - R2 * qu[i, 1],
                        R2 * qu[i, 1] + R2 * qu[i, 0],
                    )
                else:
                    # Zone 23: apply [0.0, 0.0, -R2, R2]
                    qu[i, 0], qu[i, 1], qu[i, 2], qu[i, 3] = (
                        R2 * qu[i, 2] - R2 * qu[i, 3],
                        R2 * qu[i, 3] + R2 * qu[i, 2],
                        -R2 * qu[i, 0] - R2 * qu[i, 1],
                        -R2 * qu[i, 1] + R2 * qu[i, 0],
                    )
        if qu[i, 0] < 0:
            qu[i, :] = -qu[i, :]
    return qu


@jit(nopython=True, parallel=True, fastmath=True, nogil=True, cache=True)
def test_rand_quaternions(n: int) -> np.ndarray:
    """
    Generate uniformly distributed elements of SO(3) as quaternions. This
    routine includes both 3-sphere hemispheres (will return unit quaternions
    with negative real part)

    Args:
        n (int): the number of orientations to sample

    Returns:
        np.ndarray: uniform unit quaternions (n, 4) with positive real part

    Notes:

    Due to Ken Shoemake:

    Shoemake, Ken. "Uniform random rotations." Graphics Gems III (IBM Version).
    Morgan Kaufmann, 1992. 124-132.

    """
    # h = ( sqrt(1-u) sin(2πv), sqrt(1-u) cos(2πv), sqrt(u) sin(2πw), sqrt(u) cos(2πw))
    # define random quaternions
    u = np.random.rand(n).astype(np.float64)
    v = np.random.rand(n).astype(np.float64)
    w = np.random.rand(n).astype(np.float64)
    q1 = np.sqrt(1 - u) * np.sin(2 * np.pi * v)
    q2 = np.sqrt(1 - u) * np.cos(2 * np.pi * v)
    q3 = np.sqrt(u) * np.sin(2 * np.pi * w)
    q4 = np.sqrt(u) * np.cos(2 * np.pi * w)
    qu_rand = np.stack((q1, q2, q3, q4), axis=-1)
    # # normalize the quaternions and set positive real part
    for i in prange(n):
        qu_rand[i, :] = qu_rand[i, :] / np.sqrt(np.sum(qu_rand[i, :] ** 2))
        if qu_rand[i, 0] < 0:
            qu_rand[i, :] = -qu_rand[i, :]
    return qu_rand


# TIMINGS
quats_test_ref = test_rand_quaternions(50000000).astype(np.float32)
quats_test_np = quats_test_ref.copy()
n_runs = 5

### Inplace Timings ###
quats_test_np = quats_test_ref.copy()
out = ori_to_fz_C2_nb_inplace(quats_test_np)
total_time = 0
for i in range(n_runs):
    quats_test_np = quats_test_ref.copy()
    start = time.time()
    out = ori_to_fz_C2_nb_inplace(quats_test_np)
    total_time += time.time() - start
print("manual numba C2 Inplace Timing:", 1000 * total_time / n_runs)

quats_test_np = quats_test_ref.copy()
out = ori_to_fz_D2_nb_inplace(quats_test_np)
total_time = 0
for i in range(n_runs):
    quats_test_np = quats_test_ref.copy()
    start = time.time()
    out = ori_to_fz_D2_nb_inplace(quats_test_np)
    total_time += time.time() - start
print("manual numba D2 Inplace Timing:", 1000 * total_time / n_runs)

quats_test_np = quats_test_ref.copy()
out = ori_to_fz_C3_nb_inplace(quats_test_np)
total_time = 0
for i in range(n_runs):
    quats_test_np = quats_test_ref.copy()
    start = time.time()
    out = ori_to_fz_C3_nb_inplace(quats_test_np)
    total_time += time.time() - start
print("manual numba C3 Inplace Timing:", 1000 * total_time / n_runs)

quats_test_np = quats_test_ref.copy()
out = ori_to_fz_D3_nb_inplace(quats_test_np)
total_time = 0
for i in range(n_runs):
    quats_test_np = quats_test_ref.copy()
    start = time.time()
    out = ori_to_fz_D3_nb_inplace(quats_test_np)
    total_time += time.time() - start
print("manual numba D3 Inplace Timing:", 1000 * total_time / n_runs)

quats_test_np = quats_test_ref.copy()
out = ori_to_fz_C4_nb_inplace(quats_test_np)
total_time = 0
for i in range(n_runs):
    quats_test_np = quats_test_ref.copy()
    start = time.time()
    out = ori_to_fz_C4_nb_inplace(quats_test_np)
    total_time += time.time() - start
print("manual numba C4 Inplace Timing:", 1000 * total_time / n_runs)

quats_test_np = quats_test_ref.copy()
out = ori_to_fz_D4_nb_inplace(quats_test_np)
total_time = 0
for i in range(n_runs):
    quats_test_np = quats_test_ref.copy()
    start = time.time()
    out = ori_to_fz_D4_nb_inplace(quats_test_np)
    total_time += time.time() - start
print("manual numba D4 Inplace Timing:", 1000 * total_time / n_runs)

quats_test_np = quats_test_ref.copy()
out = ori_to_fz_C6_nb_inplace(quats_test_np)
total_time = 0
for i in range(n_runs):
    quats_test_np = quats_test_ref.copy()
    start = time.time()
    out = ori_to_fz_C6_nb_inplace(quats_test_np)
    total_time += time.time() - start
print("manual numba C6 Inplace Timing:", 1000 * total_time / n_runs)

quats_test_np = quats_test_ref.copy()
out = ori_to_fz_D6_nb_inplace(quats_test_np)
total_time = 0
for i in range(n_runs):
    quats_test_np = quats_test_ref.copy()
    start = time.time()
    out = ori_to_fz_D6_nb_inplace(quats_test_np)
    total_time += time.time() - start
print("manual numba D6 Inplace Timing:", 1000 * total_time / n_runs)

quats_test_np = quats_test_ref.copy()
out = ori_to_fz_T_nb_inplace(quats_test_np)
total_time = 0
for i in range(n_runs):
    quats_test_np = quats_test_ref.copy()
    start = time.time()
    out = ori_to_fz_T_nb_inplace(quats_test_np)
    total_time += time.time() - start
print("manual numba  T Inplace Timing:", 1000 * total_time / n_runs)

quats_test_np = quats_test_ref.copy()
out = ori_to_fz_O_nb_inplace(quats_test_np)
total_time = 0
for i in range(n_runs):
    quats_test_np = quats_test_ref.copy()
    start = time.time()
    out = ori_to_fz_O_nb_inplace(quats_test_np)
    total_time += time.time() - start
print("manual numba  O Inplace Timing:", 1000 * total_time / n_runs)
