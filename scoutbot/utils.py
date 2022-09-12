# -*- coding: utf-8 -*-
'''
2022 Wild Me
'''
import logging
from logging.handlers import TimedRotatingFileHandler

import torch
import yaml

DAYS = 21


def init_logging():
    """
    Setup Python's built in logging functionality with on-disk logging, and prettier logging with Rich
    """
    # Import Rich
    import rich
    from rich.logging import RichHandler
    from rich.style import Style
    from rich.theme import Theme

    name = 'scoutbot'

    # Setup placeholder for logging handlers
    handlers = []

    # Configuration arguments for console, handlers, and logging
    console_kwargs = {
        'theme': Theme(
            {
                'logging.keyword': Style(bold=True, color='yellow'),
                'logging.level.notset': Style(dim=True),
                'logging.level.debug': Style(color='cyan'),
                'logging.level.info': Style(color='green'),
                'logging.level.warning': Style(color='yellow'),
                'logging.level.error': Style(color='red', bold=True),
                'logging.level.critical': Style(color='red', bold=True, reverse=True),
                'log.time': Style(color='white'),
            }
        )
    }
    handler_kwargs = {
        'rich_tracebacks': True,
        'tracebacks_show_locals': True,
    }
    logging_kwargs = {
        'level': logging.INFO,
        'format': '[%(name)s] %(message)s',
        'datefmt': '[%X]',
    }

    # Add file-baesd log handler
    handlers.append(
        TimedRotatingFileHandler(
            filename=f'{name}.log',
            when='midnight',
            backupCount=DAYS,
        ),
    )

    # Add rich (fancy logging) log handler
    rich.reconfigure(**console_kwargs)
    handlers.append(RichHandler(**handler_kwargs))

    # Setup global logger with the handlers and set the default level to INFO
    logging.basicConfig(handlers=handlers, **logging_kwargs)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log = logging.getLogger(name)

    return log


def init_config(config, log):
    # load config
    log.info(f'Using config "{config}"')
    cfg = yaml.safe_load(open(config, 'r'))

    cfg['log'] = log

    # check if GPU is available
    device = cfg.get('device')
    if device not in ['cpu']:
        if torch.cuda.is_available():
            cfg['device'] = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            cfg['device'] = 'mps'
        else:
            log.warning(
                f'WARNING: device set to "{device}" but not available; falling back to CPU...'
            )
            cfg['device'] = 'cpu'

    device = cfg.get('device')
    log.info(f'Using device "{device}"')

    return cfg
