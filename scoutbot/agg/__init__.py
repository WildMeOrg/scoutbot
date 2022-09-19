# -*- coding: utf-8 -*-
"""Aggregation (AGG) returns unified detects for an image given its individual tile detections

This module defines how the tile-base localization detection results are aggregated
at the image level.  This includes the ability to weight the importance of detections
along the border of each tile within an image, and performing non-maximum suppression (NMS)
on the combined results.
"""
import numpy as np
import utool as ut

from scoutbot import log

MARGIN = 32.0
AGG_THRESH = 0.4
NMS_THRESH = 0.2


def iou(box1, box2):
    """
    Computes the IoU (Intersection over Union) ratio for two bounding boxes.
    """
    inter_xtl = max(box1['xtl'], box2['xtl'])
    inter_ytl = max(box1['ytl'], box2['ytl'])
    inter_xbr = min(box1['xbr'], box2['xbr'])
    inter_ybr = min(box1['ybr'], box2['ybr'])

    inter_w = inter_xbr - inter_xtl
    inter_h = inter_ybr - inter_ytl

    if inter_w <= 0 or inter_h <= 0:
        inter = 0.0
    else:
        inter_w = max(0.0, inter_xbr - inter_xtl)
        inter_h = max(0.0, inter_ybr - inter_ytl)
        inter = inter_w * inter_h

    area1 = box1['w'] * box1['h']
    area2 = box2['w'] * box2['h']

    union = area1 + area2 - inter

    return area1, area2, inter, union


def demosaic(img_shape, tile_grids, loc_outputs, margin=MARGIN):
    """
    Demosaics a list of tiles and their respective detections back into the original
    image's coordinate system.
    """
    assert len(tile_grids) == len(loc_outputs)

    img_h, img_w = img_shape[:2]

    detects = []
    for tile_grid, loc_output in zip(tile_grids, loc_outputs):

        tile_xtl = tile_grid['x']
        tile_ytl = tile_grid['y']
        tile_w = tile_grid['w']
        tile_h = tile_grid['h']

        for detect in loc_output:
            detect_xtl = detect['x']
            detect_ytl = detect['y']
            detect_w = detect['w']
            detect_h = detect['h']
            detect_conf = detect['c']
            detect_label = detect['l']

            detect_xbr = detect_xtl + detect_w
            detect_ybr = detect_ytl + detect_h

            detect_box = {
                'xtl': detect_xtl / tile_w,
                'ytl': detect_ytl / tile_h,
                'xbr': detect_xbr / tile_w,
                'ybr': detect_ybr / tile_h,
                'w': detect_w / tile_w,
                'h': detect_h / tile_h,
            }

            margin_percent_w = margin / tile_w
            margin_percent_h = margin / tile_h

            center_box = {
                'xtl': margin_percent_w,
                'ytl': margin_percent_h,
                'xbr': 1.0 - margin_percent_w,
                'ybr': 1.0 - margin_percent_h,
                'w': 1.0 - (2.0 * margin_percent_w),
                'h': 1.0 - (2.0 * margin_percent_h),
            }
            area, _, inter, union = iou(detect_box, center_box)

            overlap = 0.0 if area <= 0 else inter / area
            overlap = round(overlap, 8)
            assert 0.0 <= overlap and overlap <= 1.0
            multiplier = np.sqrt(overlap)

            final_conf = round(detect_conf * multiplier, 4)
            if final_conf <= 0.0:
                continue

            final_xtl = int(np.around(tile_xtl + detect_xtl))
            final_ytl = int(np.around(tile_ytl + detect_ytl))
            final_w = int(np.around(detect_w))
            final_h = int(np.around(detect_h))
            final_xbr = final_xtl + final_w
            final_ybr = final_ytl + final_h

            # Check size with image frame
            final_xtl = min(max(final_xtl, 0), img_w)
            final_ytl = min(max(final_ytl, 0), img_h)
            final_xbr = min(max(final_xbr, 0), img_w)
            final_ybr = min(max(final_ybr, 0), img_h)
            final_w = final_xbr - final_xtl
            final_h = final_ybr - final_ytl

            final_area = final_w * final_h
            if final_area <= 0.0:
                continue

            detects.append(
                {
                    'l': detect_label,
                    'c': final_conf,
                    'x': final_xtl,
                    'y': final_ytl,
                    'w': final_w,
                    'h': final_h,
                }
            )

    return detects


def compute(
    img_shape, tile_grids, loc_outputs, agg_thresh=AGG_THRESH, nms_thresh=NMS_THRESH
):
    """
    Compute the aggregated image-level detection results for a given list of tile-level detections
    """
    from scoutbot.agg.py_cpu_nms import py_cpu_nms

    log.info(f'Aggregating {len(tile_grids)} tiles onto {img_shape} canvas')

    # Demosaic tile detection results and aggregate across the image
    detects = demosaic(img_shape, tile_grids, loc_outputs)

    # Filter low-confidence detections
    detects = [detect for detect in detects if detect['c'] >= agg_thresh]

    # Run NMS on aggregated detections
    coords = np.vstack(
        [
            [
                detect['x'],
                detect['y'],
                detect['x'] + detect['w'],
                detect['y'] + detect['h'],
            ]
            for detect in detects
        ]
    )
    confs = np.array([detect['c'] for detect in detects])

    keeps = py_cpu_nms(coords, confs, nms_thresh)
    final = ut.take(detects, keeps)
    final.sort(key=lambda val: val['c'], reverse=True)

    log.info(f'Found {len(final)} detections')

    return final
