#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
2022 Wild Me
'''
import setuptools


def load_requirements(filename='requirements.txt'):
    with open(filename, 'r') as file:
        return file.read().splitlines()


if __name__ == '__main__':
    setuptools.setup(
        install_requires=load_requirements()
    )
