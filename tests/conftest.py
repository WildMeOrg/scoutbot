# -*- coding: utf-8 -*-
import logging

log = logging.getLogger('pytest.conftest')  # pylint: disable=invalid-name


# @pytest.fixture()
# def cfg(config):
#     from scoutbot import utils

#     log = utils.init_logging()
#     cfg = utils.init_config(config, log)

#     cfg['output'] = 'scoutbot/{}'.format(cfg['output'])

#     return cfg


# @pytest.fixture()
# def device(cfg):
#     device = cfg.get('device')

#     return device


# @pytest.fixture()
# def net(cfg):
#     from scoutbot import model

#     net, _, _ = model.load(cfg)
#     net.eval()

#     return net
