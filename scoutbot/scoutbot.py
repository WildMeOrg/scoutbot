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


model_option = [
    click.option(
        '--config',
        help='Which ML models to use for inference',
        default=None,
        type=click.Choice(['phase1', 'mvp', 'old', 'new', 'v3', 'v3-cls']),
    ),
]

output_option = [
    click.option(
        '--output',
        help='Path to output JSON (if unspecified, results are printed to screen)',
        default=None,
        type=str,
    ),
]

shared_options = [
    click.option(
        '--backend_device',  # torch backend device
        help='Specifies the device for inference (see YOLO and PyTorch documentation for more information).',
        default='cuda:0',
        type=click.Choice(['cuda:0', 'cuda', 'mps', 'cpu']),
    ),
    click.option(
        '--wic_thresh',
        help='Whole Image Classifier (WIC) confidence threshold',
        default=int(wic.CONFIGS[None]['thresh'] * 100),
        type=click.IntRange(0, 100, clamp=True),
    ),
    click.option(
        '--loc_thresh',
        help='Localizer (LOC) confidence threshold',
        default=int(loc.CONFIGS[None]['thresh'] * 100),
        type=click.IntRange(0, 100, clamp=True),
    ),
    click.option(
        '--loc_nms_thresh',
        help='Localizer (LOC) non-maximum suppression (NMS) threshold',
        default=int(loc.CONFIGS[None]['nms'] * 100),
        type=click.IntRange(0, 100, clamp=True),
    ),
    click.option(
        '--agg_thresh',
        help='Aggregation (AGG) confidence threshold',
        default=int(agg.CONFIGS[None]['thresh'] * 100),
        type=click.IntRange(0, 100, clamp=True),
    ),
    click.option(
        '--agg_nms_thresh',
        help='Aggregation (AGG) non-maximum suppression (NMS) threshold',
        default=int(agg.CONFIGS[None]['nms'] * 100),
        type=click.IntRange(0, 100, clamp=True),
    ),
]


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


@click.command('pipeline')
@click.argument(
    'filepath',
    nargs=1,
    type=str,
    callback=pipeline_filepath_validator,
)
@add_options(model_option)
@add_options(shared_options)
@add_options(output_option)
def pipeline(
    filepath,
    config,
    output,
    backend_device,
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

    if config in ['v3', 'v3-cls']:
        wic_, detects = scoutbot.pipeline_v3(
            filepath,
            config,
            backend_device=backend_device,
            loc_thresh=loc.CONFIGS[config]['thresh'],
            slice_height=loc.CONFIGS[config]['slice_height'],
            slice_width=loc.CONFIGS[config]['slice_width'],
            overlap_height_ratio=loc.CONFIGS[config]['overlap_height_ratio'],
            overlap_width_ratio=loc.CONFIGS[config]['overlap_width_ratio'],
            perform_standard_pred=loc.CONFIGS[config]['perform_standard_pred'],
            postprocess_class_agnostic=loc.CONFIGS[config]['postprocess_class_agnostic'],
        )
    else:
        wic_, detects = scoutbot.pipeline(
            filepath,
            config=config,
            backend_device=backend_device,
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

    log.debug('Outputting results...')
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
@add_options(model_option)
@add_options(shared_options)
@add_options(output_option)
def batch(
    filepaths,
    config,
    output,
    backend_device,
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

    if config in ['v3', 'v3-cls']:
        wic_list, detects_list = scoutbot.batch_v3(
            filepaths,
            config,
            backend_device=backend_device,
            loc_thresh=loc.CONFIGS[config]['thresh'],
            slice_height=loc.CONFIGS[config]['slice_height'],
            slice_width=loc.CONFIGS[config]['slice_width'],
            overlap_height_ratio=loc.CONFIGS[config]['overlap_height_ratio'],
            overlap_width_ratio=loc.CONFIGS[config]['overlap_width_ratio'],
            perform_standard_pred=loc.CONFIGS[config]['perform_standard_pred'],
            postprocess_class_agnostic=loc.CONFIGS[config]['postprocess_class_agnostic'],
        )
    else:
        wic_list, detects_list = scoutbot.batch(
            filepaths,
            config=config,
            backend_device=backend_device,
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
            'wic': wic_,
            'loc': detects,
        }

    log.debug('Outputting results...')
    if output:
        with open(output, 'w') as outfile:
            json.dump(data, outfile)
    else:
        print(ut.repr3(data))


@click.command('fetch')
@add_options(model_option)
def fetch(config):
    """
    Fetch the required machine learning ONNX models for the WIC and LOC
    """
    scoutbot.fetch(config=config)


@click.command('example')
def example():
    """
    Run a test of the pipeline on an example image with the default configuration.
    """
    scoutbot.example()


@click.command('get_classes')
@add_options(output_option)
def get_classes(output):
    """
    Run a test of the pipeline on an example image with the default configuration.
    """
    classes = scoutbot.get_classes()
    log.debug('Outputting classes list...')
    if output:
        with open(output, 'w') as outfile:
            json.dump(classes, outfile)
    else:
        print(ut.repr3(classes))


@click.group()
def cli():
    """
    ScoutBot CLI
    """
    pass


cli.add_command(pipeline)
cli.add_command(batch)
cli.add_command(fetch)
cli.add_command(example)
cli.add_command(get_classes)


if __name__ == '__main__':
    cli()
