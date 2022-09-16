# -*- coding: utf-8 -*-
'''
Scoutbot utilities file for common and handy functions.
'''
import logging
from logging.handlers import TimedRotatingFileHandler

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
