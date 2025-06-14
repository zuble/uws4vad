'''        
class XELRankingLoss(Loss):
    def __init__(self, bs, dvc, frmt, mem_len, weight=None, batch_axis=0, **kwargs):
        super(XELRankingLoss, self).__init__(weight, batch_axis, **kwargs)
        
        self.bs = bs
        self.dvc = dvc
        self.gth_abn={
            'seg':self.gth_abn_seg,
            'seq':self.gth_abn_seg
        }.get(frmt.lower())
        self.mem_len = mem_len
    
    def gth_abn_seg(self, x):
        return x[self.bs//2:]

    ## makes sense ? unpredictable balance
    def gth_abn_seq(self, x, ldata):
        return x[self.bs//2:]
    
    def forward(self, slscores, ldata):
        
        abn_slscores = self.gth_abn(slscores) ## (bs//2, t)
        hib_scores = ldata['hib_scores'] ## (1, mem_len)
        margin_value = ldata['hib_dyn_margin']
        log.debug(f"XELRNKG/ {abn_slscores.shape=} {hib_scores.shape=} {margin_value}")

        max_a = np.max(abn_slscores, axis=1, keepdims=True)  ## (bs//2, 1)
        max_a = np.repeat(max_a, repeats=self.mem_len, axis=1)  ## (bs//2, mem_len)
        log.debug(f"XELRNKG/ {max_a.shape=}")
        
        hib_scores = np.repeat(hib_scores, repeats=abn_slscores.shape[0], axis=0)  ## (bs//2, mem_len)
        log.debug(f"XELRNKG/ {hib_scores.shape=}")
        assert max_a.shape == hib_scores.shape

        ## hinge loss over highest scores in abnormal feats and hib_scores
        margin_1 = np.ones_like(max_a) * margin_value
        tmp = npx.relu(margin_1 - max_a + hib_scores)
        hard_hinge_loss = np.mean( tmp )
        log.debug(f"XELRNKG/ loss_hinge {hard_hinge_loss}")
        
        ## loss validation
        ## hib_scores closer to 0 since they are segments with high as
        hard_score_loss=np.mean(hib_scores)
        log.debug(f"XELRNKG/ loss_hib {hard_score_loss}")
        
        return hard_hinge_loss + hard_score_loss

'''
"""  
########
## https://github.com/sdjsngs/XEL-WSAD/tree/main
## hard hinge loss usable in NetPstFwd at lossfx['xel']
_C.TRAIN.XEL = CN()
_C.TRAIN.XEL.ENABLE = True
_C.TRAIN.XEL.MEM_LEN = 0 ## auto set as cfg.TRAIN.EPOCHBATCHS * (cfg.TRAIN.BS//2)
_C.TRAIN.XEL.HIB_UPD = False ## dont change
_C.TRAIN.XEL.WARMUP = 1 ## epo rel
_C.TRAIN.XEL.MARGSTEP = 10 ## epo rel
_C.TRAIN.XEL.MARGLIST = [0.5,0.6,0.7,0.8,0.9,1.0]
## warmup+(step*6) = epochs
## epochbatchs * (bs//2)
## ucf 101

for bat, tdata in enumerate(loader):
    ndata = net(feat)
    #loss_glob =
    
    if cfg.TRAIN.XEL.ENABLE:
        # warm up  for ucf in 10 epoch for shanghaitech 5 epoch
        # margin_step 1500 and warm up end point 1500 for  shanghai tech
        #  margin step 15000 and 15000 for ucf crime
        # binary 750 and 750 for SH
        # 7500 and 7500 for UCF
        #MARGIN_LIST=[0.5,0.6,0.7,0.8,0.9,1.0]
        #margin_step = cfg.TRAIN.XEL.MARGSTEP * cfg.TRAIN.EPOCHBATCHS #750
        #warmup_end = cfg.TRAIN.XEL.WARMUP * cfg.TRAIN.EPOCHBATCHS #750
        ## esgotado when WARMUP + len(margin_list) * MARGSTEP = epos
        #curr_iter = (trn_inf['epo']) * cfg.TRAIN.EPOCHBATCHS + bat

        ###
        ldata['hib_scores'] = net(  np.expand_dims(ldata['hib'],axis=0) )['slscores'] ## (1,mem_len)
        log.debug(f" XELFWD {ldata['hib'].shape} -> {ldata['hib_scores'].shape}")
        ###

        #if curr_iter > warmup_end:
            #margin_idx = (curr_iter-warmup_end) // margin_step
        if trn_inf['epo'] > cfg.TRAIN.XEL.WARMUP:
            
            ###
            ###
            ## assumes that in the netpstfwd, only a rshp_out was done to ndata['slscores'] so its (b,t)
            loss_xel = lossfx['xel'](ndata['slscores'], ldata)
            loss_glob = loss_glob + loss_xel
            cfg.merge_from_list(['TRAIN.XEL.HIB_UPD',True])
            
            lxel = loss_xel.item()
            tmeter.update({'xel': lxel})
            #net_pst_fwd.updt_lbat('xel', lxel)
######
## XEL
if cfg.TRAIN.XEL.ENABLE:
    ## on epo end upd hib_bank
    if cfg.TRAIN.XEL.HIB_UPD:
        ldata['hib'] = hib_upg(loader, frmter, net, trn_inf)
        log.debug(f"XEL HIB UPD {trn_inf['epo']} {ldata['hib'].shape}")
    ## update margin  
    if trn_inf['epo'] > cfg.TRAIN.XEL.WARMUP:  
        margin_idx = (trn_inf['epo']-cfg.TRAIN.XEL.WARMUP) // cfg.TRAIN.XEL.MARGSTEP
        if cfg.TRAIN.XEL.MARGLIST[margin_idx] != ldata['hib_dyn_margin']:
            ldata['hib_dyn_margin'] = cfg.TRAIN.XEL.MARGLIST[margin_idx]
            log.info(f"E[{trn_inf['epo']+1}] XEL w/ margin {ldata['hib_dyn_margin']}")

#def hib_sel(nfeat, nslscores):
#    :param nfeat: [T,4096]
#    :param nslscores: [T]
#    :return: slect_feature shape in [1,4096]

#    assert  nfeat.shape[0] == nslscores.shape[0]
#    normal_score = nslscores
#    
#    #max_n, max_n_index = torch.max(normal_score, dim=0)
#    #temp_feature = nfeat[max_n_index.item()].unsqueeze(dim=0)
#    
#    max_n_index = np.argmax(nslscores, axis=0)
#    temp_feature = nfeat[max_n_index].reshape(1, -1)
#
#    # temp_feature=torch.zeros(size=[nfeat.shape[0],1,4096]).cuda()
#    # for i in range (nfeat.shape[0]):
#    #     slect_feature=nfeat[i][max_n_index[i]].unsqueeze(dim=0)
#    #     temp_feature[i]=slect_feature
#
#    return temp_feature


def hib_sel(nfeat, nslscores):
    '''Select the feature corresponding to the maximum score for each sample in the batch'''
    b, t, f = nfeat.shape
    assert nfeat.shape[:2] == nslscores.shape, "Features and scores shape mismatch"

    max_idxs = np.argmax(nslscores, axis=1)  ## (b)
    bat_idxs = np.arange(b)
    sel_feats = nfeat[bat_idxs, max_idxs].reshape(b, f)
    log.debug(f"HIB_SEL {sel_feats.shape}")
    return sel_feats


def hib_upg(loader, frmter, net, trn_inf):
    '''update the memory bank in epoch'''
    #net.eval()
    hib_new = None

    for bat, tdata in enumerate(loader):
        #nfeat = nfeat.cuda().float().squeeze(dim=0)  # batch size in 1
        (nfeat, nlabel), (afeat, alabel) = tdata
        ## b, nc, t, f
        nfeat = frmter.rshp_in(nfeat).copyto(trn_inf['dvc'])
        log.debug(f"HIB_UPD[{bat}] {nfeat.shape=}")
        
        #with torch.no_grad():
        nslscores = net(nfeat)['slscores']
        log.debug(f"HIB_UPD[{bat}] {nslscores.shape=}")


        self_feats = hib_sel(nfeat, nslscores)
        if hib_new is None: hib_new = self_feats
        else:
            #hib_new = torch.cat([hib_new, self_feats], dim=0)
            hib_new = np.concatenate ( (hib_new, self_feats), axis=0)

    return hib_new.copyto(trn_inf['dvc'])
"""