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
    type=click.Choice(['phase1', 'mvp']),
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
    type=click.Choice(['phase1', 'mvp']),
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
    Run the ScoutBot pipeline on an input image filepath
    """
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

    if output:
        with open(output, 'w') as outfile:
            data = {
                filepath: {
                    'wic': wic_,
                    'loc': detects,
                }
            }
            json.dump(data, outfile)
    else:
        log.info(filepath)
        log.info(f'WIC: {wic_:0.04f}')
        log.info('LOC: {}'.format(ut.repr3(detects)))


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
    type=click.Choice(['phase1', 'mvp']),
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
    Run the ScoutBot pipeline in batch on a list of input image filepaths
    """
    config = config.strip().lower()
    wic_thresh /= 100.0
    loc_thresh /= 100.0
    loc_nms_thresh /= 100.0
    agg_thresh /= 100.0
    agg_nms_thresh /= 100.0

    log.info(f'Running batch on {len(filepaths)} files...')

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

    if output:
        with open(output, 'w') as outfile:
            data = {}
            for filepath, wic_, detects in results:
                data[filepath] = {
                    'wic': wic,
                    'loc': detects,
                }
                json.dump(data, outfile)
    else:
        for filepath, wic_, detects in results:
            log.info(filepath)
            log.info(f'WIC: {wic_:0.04f}')
            log.info('LOC: {}'.format(ut.repr3(detects)))


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
