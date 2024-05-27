import importlib, pkgutil
from utils.logger import get_log

def get_loss(cfg, dvc):

    log = logger.get_log(cfg, __name__)
    
    lfx = {}
    ldata = {}
    ldata["label"] = None
    
    
    c_frmt = getattr(cfg.TRAIN, cfg.TRAIN.FRMT)
    
    
    for loss_idx, loss_id in enumerate(c_frmt.LOSS):
        loss_id = loss_id.upper()
        c = getattr(cfg.LOSS, loss_id)
        #log.info('losscfg:\n{}'.format(c.dump()))
        
        if loss_id == 'BCE':
            loss_mdl = importlib.import_module(".rnkg", package="loss")
            lfx['bce'] = loss_mdl.BCE()


        elif loss_id == 'RNKG':
            loss_mdl = importlib.import_module(".rnkg", package="loss")
            if 'attnomil' in cfg.NET.NAME: raise Exception('RankingLoss not possible with attnomil ')
            lfx['rnkg'] = loss_mdl.RankingLoss(
                bs=cfg.TRAIN.BS, 
                nsegments=c_frmt.LEN, 
                dvc=dvc, 
                lambda12=c.LAMBDA12, 
                version=c.VERSION
                )


        elif loss_id == 'RTFM':
            loss_mdl = importlib.import_module(".rnkg", package="loss")
            lfx['rtfm'] = loss_mdl.RTFML(
                cfg=c,
                dvc=dvc
                )


        elif loss_id == 'MBS':
            loss_mdl = importlib.import_module(".mbs", package="loss")
            lfx['mbs'] = loss_mdl.MultiBranchSupervision(
                dvc, 
                vis
                )
        
        
        elif loss_id == 'CLAS': 
            if c.FX == 'topk':
                #c.TOPK = cfg.DATA.RGB.SEGMENTNFRAMES
                log.debug(f'(ORIG k = 16 = SEGMENTNFRAMES) || {c.TOPK=} {cfg.DATA.RGB.SEGMENTNFRAMES=} ')
            loss_mdl = importlib.import_module(".clas", package="loss")
            lfx['clas'] = loss_mdl.CLAS( cfg.TRAIN.BS, dvc, c)
            ldata["seqlen"] = None 
        
        
        #elif loss_id == 'CMA_MIL':
        #    cma_mil.init(log)
        #    lfx.append(cma_mil.CMA_MIL(cfg.TRAIN.BS, dvc, c))
        #    ldata.append({"label":None, "seqlen":None})
            
        else: raise Exception('Loss not implemented')
    
        loss_mdl.init(log)
        log.info(f"loss[{loss_idx}] {list(lfx.keys())[-1]}\n{list(lfx.values())[-1]}\nldata:\n{ldata}\nlosscfg:\n{c.dump()}")

    
    return lfx, ldata
