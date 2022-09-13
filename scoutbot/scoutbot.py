#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The lecture materials for Lecture 1: Dataset Prototyping and Visualization
"""
import click


@click.command()
@click.option(
    '--config', help='Path to config file', default='configs/mnist_resnet18.yaml'
)
def wic(config):
    """
    
    """
    pass


@click.command()
@click.option(
    '--config', help='Path to config file', default='configs/mnist_resnet18.yaml'
)
def main(config):
    """
    
    """
    pass


if __name__ == '__main__':
    main()
