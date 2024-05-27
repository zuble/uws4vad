import logging

def get_log(module, debug=0):
    if not debug:
        level = logging.INFO
    else:
        level = logging.DEBUG
    log = logging.getLogger(module)
    log.setLevel(level)
    return log