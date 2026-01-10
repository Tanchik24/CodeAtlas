import logging
import logging.config
from functools import lru_cache

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'default_formatter': {
            'format': '%(asctime)s | %(levelname)-5s | %(name)s.%(funcName)s | %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },

    'handlers': {
        'stream_handler': {
            'class': 'logging.StreamHandler',
            'formatter': 'default_formatter',
        },
    },

    'loggers': {
        'main_logger': {
            'handlers': ['stream_handler'],
            'level': 'INFO',
        }
    }
}

@lru_cache()
def get_logger():
    logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger('main_logger')
