import logging, os


def get_log(module_name, cfg=None):
    log = logging.getLogger(module_name)

    if cfg is not None: 
        if cfg.get("debug"):
            # Convert list of debug paths into a comma-separated string
            dbgps_var = ','.join(cfg["debug"]["path"])
            # Store the string in an environment variable
            os.environ['UWS4VAD_DBG'] = dbgps_var
            log.warning(f"Debugging enabled for [{dbgps_var}] in module {module_name}")
            lvl = logging.DEBUG
        else: 
            os.environ['UWS4VAD_DBG'] = ""
            lvl = logging.INFO
        #else: lvl
    else:
        lvl = logging.INFO
    
        dbgps_var = os.environ.get('UWS4VAD_DBG', '')
        dbgps = dbgps_var.split(',') if dbgps_var else []
        for dbgp in dbgps:
            if dbgp in module_name:
                log.warning(f"Debugging enabled for [{dbgp}] in module {module_name}")
                lvl = logging.DEBUG
                log.warning(f"dbg[{dbgp}] set {module_name}")
                break
            
    log.setLevel( lvl )
    return log