import logging

LEVEL = {
    'full':  None,
    'data':  logging.INFO,
    'vldt':  logging.DEBUG,
    'train':  logging.INFO,
    'loss':  logging.INFO,
    'net':  logging.INFO
}


def get_log(module_name, cfg=None):
    log = logging.getLogger(module_name)
    #log.error(LEVEL)
    
    ## enters one by main call
    if cfg is not None and cfg.get("debug"):
        d = cfg.get("debug")
        if d.full:
            LEVEL["full"]: logging.DEBUG
            log.warning("ALL MODULES")
        else:
            ## if > 0 level is debug
            if d.data:
                LEVEL["data"]: logging.DEBUG
                log.warning("DATA")
            if d.vldt:
                LEVEL["vldt"]: logging.DEBUG
                log.warning("VLDT")
            if d.train:
                LEVEL["train"]: logging.DEBUG
                log.warning("TRAIN")
            if d.loss:
                LEVEL["loss"]: logging.DEBUG
                log.warning("LOSS")
            if d.net:
                LEVEL["net"]: logging.DEBUG
                log.warning("NET")
        #
        
    if LEVEL["full"] is not None:
        log.setLevel(LEVEL["full"])
        log.warning(module_name)
    else:
        if '.data.' in module_name:
            log.setLevel( LEVEL["data"] )
        if '.vldt.' in module_name:
            log.setLevel( LEVEL["vldt"] )
        if '.train.' in module_name:
            log.setLevel( LEVEL["train"] )
        if '.loss.' in module_name:
            log.setLevel( LEVEL["loss"] )
        if '.net.' in module_name:
            log.setLevel( LEVEL["net"] )        
    return log