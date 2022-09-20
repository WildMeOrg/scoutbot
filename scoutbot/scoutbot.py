#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI for ScoutBot
"""
import json
from os.path import exists

import click
import utool as ut

import scoutbot
from scoutbot import agg, loc, log, wic


def pipeline_filepath_validator(ctx, param, value):
    if not exists(value):
        log.error(f'Input filepath does not exist: {value}')
        ctx.exit()
    return value


@click.command()
@click.option(
    '--filepath',
    help='Path to image',
    required=True,
    type=str,
    callback=pipeline_filepath_validator,
)
@click.option(
    '--output',
    help='Path to output JSON (if unspecified, results are printed to screen)',
    default=None,
    type=click.IntRange(0, 100, clamp=True),
)
@click.option(
    '--wic_thresh',
    help='Whole Image Classifier (WIC) confidence threshold',
    default=wic.WIC_THRESH,
    type=click.IntRange(0, 100, clamp=True),
)
@click.option(
    '--loc_thresh',
    help='Localizer (LOC) confidence threshold',
    default=loc.LOC_THRESH,
    type=click.IntRange(0, 100, clamp=True),
)
@click.option(
    '--loc_nms_thresh',
    help='Localizer (LOC) non-maximum suppression (NMS) threshold',
    default=loc.NMS_THRESH,
    type=click.IntRange(0, 100, clamp=True),
)
@click.option(
    '--agg_thresh',
    help='Aggregation (AGG) confidence threshold',
    default=agg.AGG_THRESH,
    type=click.IntRange(0, 100, clamp=True),
)
@click.option(
    '--agg_nms_thresh',
    help='Aggregation (AGG) non-maximum suppression (NMS) threshold',
    default=agg.NMS_THRESH,
    type=click.IntRange(0, 100, clamp=True),
)
def pipeline(
    filepath, output, wic_thresh, loc_thresh, loc_nms_thresh, agg_thresh, agg_nms_thresh
):
    """
    Run the ScoutBot pipeline on an input image filepath
    """
    wic_thresh /= 100.0
    loc_thresh /= 100.0
    loc_nms_thresh /= 100.0
    agg_thresh /= 100.0
    agg_nms_thresh /= 100.0

    detects = scoutbot.pipeline(
        filepath,
        wic_thresh=wic_thresh,
        loc_thresh=loc_thresh,
        loc_nms_thresh=loc_nms_thresh,
        agg_thresh=agg_thresh,
        agg_nms_thresh=agg_nms_thresh,
    )

    if output:
        with open(output, 'w') as outfile:
            json.dump(detects, outfile)
    else:
        log.info(ut.repr3(detects))


@click.command('fetch')
def fetch():
    """
    Fetch the required machine learning ONNX models for the WIC and LOC
    """
    scoutbot.fetch()


@click.command('example')
def example():
    """
    Run a test of the pipeline on an example image
    """
    scoutbot.example()


@click.group()
def cli():
    """
    ScoutBot CLI
    """
    pass


cli.add_command(fetch)
cli.add_command(pipeline)
cli.add_command(example)


if __name__ == '__main__':
    cli()
