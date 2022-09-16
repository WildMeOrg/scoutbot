# -*- coding: utf-8 -*-
'''
ScoutBot is the machine learning interface for the Wild Me Scout project.

Notes:
    detection_config = {
        'algo': 'tile_aggregation',
        'config_filepath': 'variant3-32',
        'weight_filepath': 'densenet+lightnet;scout-5fbfff26-boost3,0.400,scout_5fbfff26_v0,0.4',
        'nms_thresh': 0.8,
        'sensitivity': 0.5077,
    }

    (
        wic_model_tag,
        wic_thresh,
        weight_filepath,
        nms_thresh,
    ) = 'scout-5fbfff26-boost3,0.400,scout_5fbfff26_v0,0.4'


    wic_confidence_list = ibs.scout_wic_test(
        gid_list, classifier_algo='densenet', model_tag=wic_model_tag
    )
    config = {
        'grid': False,
        'algo': 'lightnet',
        'config_filepath': weight_filepath,
        'weight_filepath': weight_filepath,
        'nms': True,
        'nms_thresh': nms_thresh,
        'sensitivity': 0.0,
    }
    prediction_list = depc.get_property(
        'localizations', gid_list_, None, config=config
    )
'''
from scoutbot import agg, loc, tile, wic

VERSION = '0.1.2'
version = VERSION
__version__ = VERSION


def fetch(pull=False):
    wic.fetch(pull=pull)
    loc.fetch(pull=pull)


def pipeline(
    filepath,
    wic_thresh=wic.WIC_THRESH,
    loc_thresh=loc.LOC_THRESH,
    loc_nms_thresh=loc.NMS_THRESH,
    agg_thresh=agg.AGG_THRESH,
    agg_nms_thresh=agg.NMS_THRESH,
):
    import utool as ut

    # Run tiling
    img_shape, tile_grids, tile_filepaths = tile.compute(filepath)

    # Run WIC
    wic_outputs = wic.post(wic.predict(wic.pre(tile_filepaths)))

    # Threshold for WIC
    flags = [wic_output.get('positive') >= wic_thresh for wic_output in wic_outputs]
    loc_tile_grids = ut.compress(tile_grids, flags)
    loc_tile_filepaths = ut.compress(tile_filepaths, flags)

    # Run localizer
    loc_data, loc_sizes = loc.pre(loc_tile_filepaths)
    loc_preds = loc.predict(loc_data)
    loc_outputs = loc.post(
        loc_preds, loc_sizes, loc_thresh=loc_thresh, nms_thresh=loc_nms_thresh
    )
    assert len(loc_tile_grids) == len(loc_outputs)

    # Run Aggregation
    detects = agg.compute(
        img_shape,
        loc_tile_grids,
        loc_outputs,
        agg_thresh=agg_thresh,
        nms_thresh=agg_nms_thresh,
    )

    return detects
