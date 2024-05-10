#import atexit, builtins, decimal, functools
import logging, os.path as osp, sys, copy
import matplotlib
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from logging import config


class ColoredFormatter(logging.Formatter):
    ## add ANSI escape sequences to [%(levelname)s] in log message
    colors = {
        'WARNING': "\x1b[33;20m",  # Yellow
        'INFO': "\x1b[32;20m",     # Green
        'DEBUG': "\x1b[34;20m",    # Blue
        'CRITICAL': "\x1b[35;20m", # Magenta
        'ERROR': "\x1b[31;20m",    # Red
    }
    reset = "\x1b[0m"

    def format(self, record):
        new_record = copy.copy(record)
        levelname = new_record.levelname
        if levelname in self.colors:
            new_record.levelname = self.colors[levelname] + levelname + self.reset
        return super().format(new_record)

class LoggerManager:
    _loggers = {}
    _debug_cfg = None
    _debug_root = None 
    _initialized = False
    _dd = {20: "INFO", 10: "DEBUG"}
    
    @classmethod
    def setup(cls, output_dir, cfg, fn='stdout.log'):
        
        if cls._initialized: 
            print("LOG INNITED ALREADY")
            return 

        log_file = osp.join(output_dir, fn)
        cls._debug_cfg = cfg
        cls._debug_root = cfg.ROOT
        LOGGING_CONFIG = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'plain': {
                    'format': '[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)4d]: %(message)s',
                    'datefmt': "%m/%d %H:%M:%S",
                },
                'colored': {
                    '()': ColoredFormatter,
                    'format': '[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)4d]: %(message)s',
                    'datefmt': "%m/%d %H:%M:%S",
                },
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'DEBUG',
                    'formatter': 'colored',
                },
                'file': {
                    'class': 'logging.FileHandler',
                    'filename': log_file,
                    'level': 'DEBUG',
                    'formatter': 'plain',
                }
            },
            'root': {
                'level': 'DEBUG',
                'handlers': ['console', 'file'] if log_file else ['console'],
            },
            'loggers': {
                # Additional loggers can be configured here based on the config if needed
            },
        }

        logging.config.dictConfig(LOGGING_CONFIG)
        cls._initialized = True
        return log_file

    @classmethod
    def get_logger(cls, module):
        if not cls._initialized:
            raise Exception("LoggerManager is not setup yet. Please call LoggerManager.setup() first.")

        ## returns if already created 
        if module in cls._loggers:
            return cls._loggers[module]
        
        logger = logging.getLogger(module)
        
        ## sets the log level based on cfg for the specified module in which the get_logger was called
        ## if the root is set, set all modeule loggers to DEBUG
        if cls._debug_root: level = logging.DEBUG
        else:
            #logger.error(f"{module=}")
            
            ## utils/misc vldt/vldt vldt/metric train test tmp 
            aaa = getattr(cls._debug_cfg, module.split('.')[-1].upper(), 2)
            #logger.error(f"{module.split('.')[-1]} {aaa}")
            if aaa == 1: level = logging.DEBUG
            elif aaa == 0: level  = logging.INFO
            else: 
                ## data/.. nets/.. loss/.. 
                bbb = getattr(cls._debug_cfg, module.split('.')[0].upper(), 2)    
                #logger.error(f"{module.split('.')[0]} {bbb}")
                if bbb == 1: level  = logging.DEBUG
                elif bbb == 0: level  = logging.INFO
                else: level = logging.INFO

        #logger.error(f"{module=} {level=}")
        logger.setLevel(level)
        cls._loggers[module] = logger
        logger.debug(f"{module} logger initialized w/ level {cls._dd[level]}")
        return logger
    