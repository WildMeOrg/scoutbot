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


@click.command('fetch')
@click.option(
    '--config',
    help='Which ML models to use for inference',
    default=None,
    type=click.Choice(['phase1', 'mvp', 'old', 'new']),
)
def fetch(config):
    """
    Fetch the required machine learning ONNX models for the WIC and LOC
    """
    scoutbot.fetch(config=config)


@click.command('pipeline')
@click.argument(
    'filepath',
    nargs=1,
    type=str,
    callback=pipeline_filepath_validator,
)
@click.option(
    '--config',
    help='Which ML models to use for inference',
    default=None,
    type=click.Choice(['phase1', 'mvp', 'old', 'new']),
)
@click.option(
    '--output',
    help='Path to output JSON (if unspecified, results are printed to screen)',
    default=None,
    type=str,
)
@click.option(
    '--wic_thresh',
    help='Whole Image Classifier (WIC) confidence threshold',
    default=int(wic.CONFIGS[None]['thresh'] * 100),
    type=click.IntRange(0, 100, clamp=True),
)
@click.option(
    '--loc_thresh',
    help='Localizer (LOC) confidence threshold',
    default=int(loc.CONFIGS[None]['thresh'] * 100),
    type=click.IntRange(0, 100, clamp=True),
)
@click.option(
    '--loc_nms_thresh',
    help='Localizer (LOC) non-maximum suppression (NMS) threshold',
    default=int(loc.CONFIGS[None]['nms'] * 100),
    type=click.IntRange(0, 100, clamp=True),
)
@click.option(
    '--agg_thresh',
    help='Aggregation (AGG) confidence threshold',
    default=int(agg.CONFIGS[None]['thresh'] * 100),
    type=click.IntRange(0, 100, clamp=True),
)
@click.option(
    '--agg_nms_thresh',
    help='Aggregation (AGG) non-maximum suppression (NMS) threshold',
    default=int(agg.CONFIGS[None]['nms'] * 100),
    type=click.IntRange(0, 100, clamp=True),
)
def pipeline(
    filepath,
    config,
    output,
    wic_thresh,
    loc_thresh,
    loc_nms_thresh,
    agg_thresh,
    agg_nms_thresh,
):
    """
    Run the ScoutBot pipeline on an input image filepath.  An example output of the JSON
    can be seen below.

    .. code-block:: javascript

            {
                '/path/to/image.ext': {
                    'wic': 0.5,
                    'loc': [
                        {
                            'l': 'elephant',
                            'c': 0.9,
                            'x': 100,
                            'y': 100,
                            'w': 50,
                            'h': 10
                        },
                        ...
                    ],
                }
            }
    """
    if config is not None:
        config = config.strip().lower()
    wic_thresh /= 100.0
    loc_thresh /= 100.0
    loc_nms_thresh /= 100.0
    agg_thresh /= 100.0
    agg_nms_thresh /= 100.0

    wic_, detects = scoutbot.pipeline(
        filepath,
        config=config,
        wic_thresh=wic_thresh,
        loc_thresh=loc_thresh,
        loc_nms_thresh=loc_nms_thresh,
        agg_thresh=agg_thresh,
        agg_nms_thresh=agg_nms_thresh,
    )

    data = {
        filepath: {
            'wic': wic_,
            'loc': detects,
        }
    }

    if output:
        with open(output, 'w') as outfile:
            json.dump(data, outfile)
    else:
        print(ut.repr3(data))


@click.command('batch')
@click.argument(
    'filepaths',
    nargs=-1,
    type=str,
)
@click.option(
    '--config',
    help='Which ML models to use for inference',
    default=None,
    type=click.Choice(['phase1', 'mvp', 'old', 'new']),
)
@click.option(
    '--output',
    help='Path to output JSON (if unspecified, results are printed to screen)',
    default=None,
    type=str,
)
@click.option(
    '--wic_thresh',
    help='Whole Image Classifier (WIC) confidence threshold',
    default=int(wic.CONFIGS[None]['thresh'] * 100),
    type=click.IntRange(0, 100, clamp=True),
)
@click.option(
    '--loc_thresh',
    help='Localizer (LOC) confidence threshold',
    default=int(loc.CONFIGS[None]['thresh'] * 100),
    type=click.IntRange(0, 100, clamp=True),
)
@click.option(
    '--loc_nms_thresh',
    help='Localizer (LOC) non-maximum suppression (NMS) threshold',
    default=int(loc.CONFIGS[None]['nms'] * 100),
    type=click.IntRange(0, 100, clamp=True),
)
@click.option(
    '--agg_thresh',
    help='Aggregation (AGG) confidence threshold',
    default=int(agg.CONFIGS[None]['thresh'] * 100),
    type=click.IntRange(0, 100, clamp=True),
)
@click.option(
    '--agg_nms_thresh',
    help='Aggregation (AGG) non-maximum suppression (NMS) threshold',
    default=int(agg.CONFIGS[None]['nms'] * 100),
    type=click.IntRange(0, 100, clamp=True),
)
def batch(
    filepaths,
    config,
    output,
    wic_thresh,
    loc_thresh,
    loc_nms_thresh,
    agg_thresh,
    agg_nms_thresh,
):
    """
    Run the ScoutBot pipeline in batch on a list of input image filepaths.
    An example output of the JSON can be seen below.

    .. code-block:: javascript

            {
                '/path/to/image1.ext': {
                    'wic': 0.5,
                    'loc': [
                        {
                            'l': 'elephant',
                            'c': 0.9,
                            'x': 100,
                            'y': 100,
                            'w': 50,
                            'h': 10
                        },
                        ...
                    ],
                },
                '/path/to/image2.ext': {
                    'wic': 0.5,
                    'loc': [
                        {
                            'l': 'elephant',
                            'c': 0.9,
                            'x': 100,
                            'y': 100,
                            'w': 50,
                            'h': 10
                        },
                        ...
                    ],
                },
                ...
            }
    """
    if config is not None:
        config = config.strip().lower()
    wic_thresh /= 100.0
    loc_thresh /= 100.0
    loc_nms_thresh /= 100.0
    agg_thresh /= 100.0
    agg_nms_thresh /= 100.0

    log.debug(f'Running batch on {len(filepaths)} files...')

    wic_list, detects_list = scoutbot.batch(
        filepaths,
        config=config,
        wic_thresh=wic_thresh,
        loc_thresh=loc_thresh,
        loc_nms_thresh=loc_nms_thresh,
        agg_thresh=agg_thresh,
        agg_nms_thresh=agg_nms_thresh,
    )
    results = zip(filepaths, wic_list, detects_list)

    data = {}
    for filepath, wic_, detects in results:
        data[filepath] = {
            'wic': wic,
            'loc': detects,
        }

    if output:
        with open(output, 'w') as outfile:
            json.dump(data, outfile)
    else:
        print(ut.repr3(data))


@click.command('example')
def example():
    """
    Run a test of the pipeline on an example image with the Phase 1 models
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
cli.add_command(batch)
cli.add_command(example)


if __name__ == '__main__':
    cli()
