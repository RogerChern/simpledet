import math
import numpy as np
import operator

from six.moves import reduce
from PIL import Image, ImageDraw


def get_line_length(point1, point2):
    return math.sqrt( math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

def get_2d_rotation_matrix_from_radian_batch(angle_in_radian):
    """
    Args:
        angle_in_radian: (N, )
    Returns:
        rotation_matrix: (N, 2, 2)
    """
    assert angle_in_radian.ndim == 1, angle_in_radian
    R = np.array([
            [np.cos(angle_in_radian), -np.sin(angle_in_radian)],
            [np.sin(angle_in_radian),  np.cos(angle_in_radian)]], dtype=np.float32)  # (2, 2, N)
    return np.einsum('ijk->kij', R)

def rotate_quadruplet_batch(quadruplet, angle):
    """
    Args:
        quadruplet: (N, 4, 2)
        angle: a single float in radian or (N, )
    Returns:
        rotated_quadruplet: (N, 4, 2)
    """
    assert angle.ndim == 1, angle
    if isinstance(angle, float):
        angle = np.ones(shape=(quadruplet.shape[0], )) * angle

    R = get_2d_rotation_matrix_from_radian_batch(angle)
    return np.einsum('ijk,imk->ijm', quadruplet, R)

def align_quadruplet_corner_to_rectangle(quadruplet):
    """
    Mapping the 4 corners of a quadruplet to those of a rectangle.
    After the mapping, (x1, y1), (x2, y2), (x3, y3), (x4, y4) of the quadruplet shall be aligned with topleft, topright, botomright, bottomleft corners of a rectangle.
    Args:
        quadruplet: (8, )
    Returns:
        aligned_quadruplet: (8, )
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = quadruplet
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    corner_orders = [
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
        [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
        [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
        [[x4, y4], [x1, y1], [x2, y2], [x3, y3]],
    ]
    dst_coordinate = [
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
    ]
    force = 100000000.0
    force_flag = -1
    for i in range(4):
        temp_force = get_line_length(corner_orders[i][0], dst_coordinate[0]) + \
                     get_line_length(corner_orders[i][1], dst_coordinate[1]) + \
                     get_line_length(corner_orders[i][2], dst_coordinate[2]) + \
                     get_line_length(corner_orders[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    return reduce(operator.add, corner_orders[force_flag])

def convert_aligned_quadruplet_to_rotated_bbox(quadruplet, with_modulo=True):
    """
    Args:
        quadruplet: (N, 8), quadruplets in format [x1, y1, x2, y2, x3, y3, x4, y4]
    Returns:
        rotated_bbox: (N, 5), rotated bboxes in format [cx, cy, w, h, clockwise_angle_in_radian]
    """
    bbox = np.array(quadruplet, dtype=np.float32, copy=True)
    bbox = np.reshape(bbox, newshape=(-1, 4, 2))  # (N, 4, 2)
    angle_in_radian = np.arctan((bbox[:, 1, 1] - bbox[:, 0, 1]) / max((bbox[:, 1, 0] - bbox[:, 0, 0]), 1))
    center = np.mean(bbox, axis=1)  # (N, 2)
    unrotated = rotate_quadruplet_batch(bbox - center[:, None, :], -angle_in_radian)

    xmin = np.min(unrotated[:, :, 0], axis=1)
    xmax = np.max(unrotated[:, :, 0], axis=1)
    ymin = np.min(unrotated[:, :, 1], axis=1)
    ymax = np.max(unrotated[:, :, 1], axis=1)

    w = xmax - xmin + 1
    h = ymax - ymin + 1
    w = w[:, None]
    h = h[:, None]

    # TODO: check it
    if with_modulo:
        angle_in_radian = angle_in_radian[:, np.newaxis] % (2 * np.pi)
    else:
        angle_in_radian = angle_in_radian[:, np.newaxis]
    rotated_bbox = np.concatenate((center, w, h, angle_in_radian), axis=1)
    return rotated_bbox

def convert_rotated_bbox_to_quadruplet(rotated_bbox):
    """
    Args:
        rotated_bbox: (N, 5)
    Returns:
        rotated_quadruplet: (N, 8)
    """
    assert rotated_bbox.ndim == 2 and rotated_bbox.shape[-1] == 5
    cx, cy, w, h, r = np.split(rotated_bbox, 5, axis=-1)
    xmin = -(w - 1) / 2
    ymin = -(h - 1) / 2
    xmax = (w - 1) / 2
    ymax = (h - 1) / 2
    xminymin = np.concatenate([xmin, ymin], axis=1)
    xmaxymin = np.concatenate([xmax, ymin], axis=1)
    xmaxymax = np.concatenate([xmax, ymax], axis=1)
    xminymax = np.concatenate([xmin, ymax], axis=1)
    quadruplet = np.stack([xminymin, xmaxymin, xmaxymax, xminymax], axis=1)
    assert quadruplet.shape[-2:] == (4, 2), quadruplet.shape
    rotated_quadruplet = rotate_quadruplet_batch(quadruplet, r.reshape(-1))
    center = np.concatenate((cx, cy), axis=-1)
    return rotated_quadruplet + center[:, None, :]
