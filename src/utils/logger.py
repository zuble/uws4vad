import logging, os

#LEVEL = {
#    'full':  None,
#    'data':  logging.INFO,
#    'vldt':  logging.DEBUG,
#    'train':  logging.INFO,
#    'loss':  logging.INFO,
#    'net':  logging.INFO
#}


#def get_log(module_name, cfg=None):
#    log = logging.getLogger(module_name)
#    #log.error(LEVEL)
#    
#    ## enters one by main call
#    if cfg is not None and cfg.get("debug"):
#        d = cfg.get("debug")
#        if d.full:
#            LEVEL["full"]: logging.DEBUG
#            log.warning("ALL MODULES")
#        else:
#            ## if > 0 level is debug
#            if d.data:
#                LEVEL["data"]: logging.DEBUG
#                log.warning("DATA")
#            if d.vldt:
#                LEVEL["vldt"]: logging.DEBUG
#                log.warning("VLDT")
#            if d.train:
#                LEVEL["train"]: logging.DEBUG
#                log.warning("TRAIN")
#            if d.loss:
#                LEVEL["loss"]: logging.DEBUG
#                log.warning("LOSS")
#            if d.net:
#                LEVEL["net"]: logging.DEBUG
#                log.warning("NET")
#        #
#        
#    if LEVEL["full"] is not None:
#        log.setLevel(LEVEL["full"])
#        log.warning(module_name)
#    else:
#        if '.data.' in module_name:
#            log.setLevel( LEVEL["data"] )
#        if '.vldt.' in module_name:
#            log.setLevel( LEVEL["vldt"] )
#        if '.train.' in module_name:
#            log.setLevel( LEVEL["train"] )
#        if '.loss.' in module_name:
#            log.setLevel( LEVEL["loss"] )
#        if '.net.' in module_name:
#            log.setLevel( LEVEL["net"] )        
#    return log


def get_log(module_name, cfg=None):
    log = logging.getLogger(module_name)

    if cfg is not None: 
        if cfg.get("debug"):
            if cfg.debug.full:
                os.environ.setdefault('UWS4VAD_DBG_FULL', 'DEBUG')
                lvl = logging.DEBUG
            else:
                if cfg.debug.data:
                    os.environ.setdefault('UWS4VAD_DBG_DATA', 'DEBUG')
                if cfg.debug.vldt:
                    os.environ.setdefault('UWS4VAD_DBG_VLDT', 'DEBUG')
                if cfg.debug.train:
                    os.environ.setdefault('UWS4VAD_DBG_TRAIN', 'DEBUG')
                if cfg.debug.model:
                    os.environ.setdefault('UWS4VAD_DBG_MODEL', 'DEBUG')
                lvl = logging.INFO
            #os.environ.setdefault('HYDRA_FULL_ERROR', '1')
        else: lvl = logging.INFO
    else:
        ## all except main
        tt = ''
        if 'src.data' in module_name: tt = 'DATA'
        if 'src.vldt' in module_name: tt = 'VLDT'
        if 'src.train' in module_name: tt = 'TRAIN'
        if 'src.model' in module_name: tt = 'MODEL'
        
        tmp = os.environ.get('UWS4VAD_DBG_'+tt, 'INFO')
        #log.error(f"{tmp} {module_name}")
        lvl={
            'DEBUG':logging.DEBUG,
            'INFO':logging.INFO,
        }.get(tmp, logging.INFO)
        #log.warning(f"{tmp} {module_name}")
        
    log.setLevel( lvl )
    return log