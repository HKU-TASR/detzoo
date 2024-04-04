
"""
This module provides functions to convert bounding boxes between different formats.

Formats:
- ltwh: (left, top, width, height)
- ltrb: (left, top, right, bottom)
- ccwh: (center_x, center_y, width, height)

Each function takes a tensor of shape (N, 4) as input and returns a tensor of the same shape.

"""

def ltwh_to_ltrb(ltwh):
    ltrb = ltwh.clone()
    ltrb[:, 2] = ltwh[:, 0] + ltwh[:, 2]
    ltrb[:, 3] = ltwh[:, 1] + ltwh[:, 3]
    return ltrb

def ltrb_to_ltwh(ltrb):
    ltwh = ltrb.clone()
    ltwh[:, 2] = ltrb[:, 2] - ltrb[:, 0]
    ltwh[:, 3] = ltrb[:, 3] - ltrb[:, 1]
    return ltwh

def ltwh_to_ccwh(ltwh):
    ccwh = ltwh.clone()
    ccwh[:, 0] = ltwh[:, 0] + ltwh[:, 2] / 2
    ccwh[:, 1] = ltwh[:, 1] + ltwh[:, 3] / 2
    return ccwh

def ccwh_to_ltwh(ccwh):
    ltwh = ccwh.clone()
    ltwh[:, 0] = ccwh[:, 0] - ccwh[:, 2] / 2
    ltwh[:, 1] = ccwh[:, 1] - ccwh[:, 3] / 2
    return ltwh

def ltrb_to_ccwh(ltrb):
    ccwh = ltrb.clone()
    ccwh[:, 0] = (ltrb[:, 0] + ltrb[:, 2]) / 2
    ccwh[:, 1] = (ltrb[:, 1] + ltrb[:, 3]) / 2
    ccwh[:, 2] = ltrb[:, 2] - ltrb[:, 0]
    ccwh[:, 3] = ltrb[:, 3] - ltrb[:, 1]
    return ccwh

def ccwh_to_ltrb(ccwh):
    ltrb = ccwh.clone()
    ltrb[:, 0] = ccwh[:, 0] - ccwh[:, 2] / 2
    ltrb[:, 1] = ccwh[:, 1] - ccwh[:, 3] / 2
    ltrb[:, 2] = ccwh[:, 0] + ccwh[:, 2] / 2
    ltrb[:, 3] = ccwh[:, 1] + ccwh[:, 3] / 2
    return ltrb