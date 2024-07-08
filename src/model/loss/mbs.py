import torch
import torch.nn as nn

class MultiBranchSupervision(nn.Module):
    def __init__(self, _cfg):
        super(MultiBranchSupervision, self).__init__()
        self.device = devices
        self.eps = 0.2
        self.alpha = 0.8
        self.gamma = 0.8
        self.µ = 0.01
        self.cur_stp = 1
        self.M = 400
        self.bce = nn.BCEWithLogitsLoss()
        self.vis = vis

    def forward(self, mdata, ldata):
        log.debug(f"mbattsup [{self.cur_stp}]")
        if self.cur_stp == self.M:
            log.warning("LG_1 (L pos guide) calc changing")

        x_cls, x_att = mdata["slscores"], mdata["attw"]

        bs = x_cls.shape[0] // 2
        slen = x_cls.shape[1]
        dbg_stp = 100
        L = None

        for bi in range(bs): ## assumes mil
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f"mbattsup[{bi}/{bs}]")

            # NORMAL
            SO_0 = x_cls[bi]
            Ai_0 = x_att[bi]
            SA_0 = Ai_0 * SO_0
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbattsup[{bi}/{bs}]/Scores: SO_0 {SO_0.shape}, Ai_0 {Ai_0.shape}, SA_0 {SA_0.shape}')

            thetai_0 = ((torch.max(Ai_0) - torch.min(Ai_0)) * self.eps) + torch.min(Ai_0)
            SSO_0 = SO_0 * (Ai_0 < thetai_0)
            SSA_0 = SA_0 * (Ai_0 < thetai_0)
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbattsup[{bi}/{bs}]/SupressedScores: SSO_0 {SSO_0.shape}, SSA_0 {SSA_0.shape}')

            # ABNORMAL
            SO_1 = x_cls[bs + bi]
            Ai_1 = x_att[bs + bi]
            SA_1 = Ai_1 * SO_1
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbattsup[{bi}/{bs}]/Scores: SO_1 {SO_1.shape}, Ai_1 {Ai_1.shape}, SA_1 {SA_1.shape}')

            thetai_1 = ((torch.max(Ai_1) - torch.min(Ai_1)) * self.eps) + torch.min(Ai_1)
            SSO_1 = SO_1 * (Ai_1 < thetai_1)
            SSA_1 = SA_1 * (Ai_1 < thetai_1)
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbattsup[{bi}/{bs}]/SupressedScores: SSO_1 {SSO_1.shape}, SSA_1 {SSA_1.shape}')

            ## Guide , eq(9,10)
            LG_0 = torch.mean((Ai_0 - 0) ** 2) ## MSE((Aneg, {0 · · · 0}))
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbattsup[{bi}/{bs}]/LossGuide: LG_0 {LG_0}')

            if self.cur_stp < self.M: lg1_tmp = SO_1
            else: lg1_tmp = (SO_1 > 0.5).float()
            LG_1 = torch.mean((Ai_1 - lg1_tmp) ** 2)
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbattsup[{bi}/{bs}]/LossGuide: LG_1 {LG_1}')
                
            ## Loss L1-Norm , eq(11)
            LN_1 = torch.sum(torch.abs(Ai_1))
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbattsup[{bi}/{bs}]/LossNorm: LN_1 {LN_1}')

            ## Loss SL scores
            z = torch.zeros((1, slen), device=self.device)
            o = torch.ones((1, slen), device=self.device)
            ## Raw Org/Att , eq(12)
            LC_SO = self.bce(SO_0.unsqueeze(0), z) + self.bce(SO_1.unsqueeze(0), o)
            LC_SA = self.bce(SA_0.unsqueeze(0), z) + self.bce(SA_1.unsqueeze(0), o)
            ## Supressed Org/Att , eq(12)
            LC_SSO = self.bce(SSO_0.unsqueeze(0), z) + self.bce(SSO_1.unsqueeze(0), o)
            LC_SSA = self.bce(SSA_0.unsqueeze(0), z) + self.bce(SSA_1.unsqueeze(0), o)

            LC = self.alpha * (LC_SO + LC_SA) + (1 - self.alpha) * (LC_SSO + LC_SSA) ## eq(13)
            if (self.cur_stp % dbg_stp) == 0: 
                log.debug(f'mbattsup[{bi}/{bs}]/LossCls: LC_SO {LC_SO}, LC_SA {LC_SA}, LC_SSO {LC_SSO}, LC_SSA {LC_SSA}, LC_ALL {LC}')
            
            ## Loss Smoothness , eq(14)
            LSmt_SOrg = torch.sum((SO_0[:-1] - SO_0[1:]) ** 2) + torch.sum((SO_1[:-1] - SO_1[1:]) ** 2)
            LSmt_SAtt = torch.sum((SA_0[:-1] - SA_0[1:]) ** 2) + torch.sum((SA_1[:-1] - SA_1[1:]) ** 2)
            LSmt = LSmt_SOrg + LSmt_SAtt
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbattsup[{bi}/{bs}]/LossSmooth: SO {LSmt_SOrg}, SA {LSmt_SAtt}, SUM {LSmt}')
            
            ## Loss Sparsity , eq(14)
            LSpr_SOrg = torch.sum(SO_0) + torch.sum(SO_1)
            LSpr_SAtt = torch.sum(SA_0) + torch.sum(SA_1)
            LSpr = LSpr_SOrg + LSpr_SAtt
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbattsup[{bi}/{bs}]/LossSparse: SO {LSpr_SOrg}, SA {LSpr_SAtt}, SUM {LSpr}')

            tmp = LC + self.gamma * LN_1 + LG_0 + LG_1 + self.µ * LSmt + LSpr  ## eq(15)

            self.plot_loss(LC, self.gamma * LN_1, LG_0, LG_1, self.µ * LSmt, LSpr, tmp)

            if L is None:
                L = tmp
            else:
                L = torch.cat((L, tmp), dim=0)
                
        self.cur_stp += 1
        return L

    def plot_loss(self, LC , LN_1, LG_0, LG_1, LSmt, LSpr, L):
        self.vis.plot_lines('LCls (LC)', LC.detach().cpu().numpy())
        self.vis.plot_lines('LNorm (LN_1)', LN_1.detach().cpu().numpy())
        self.vis.plot_lines('LGuia_0 (LG_0)', LG_0.detach().cpu().numpy())
        self.vis.plot_lines('LGuia_1 (LG_1)', LG_1.detach().cpu().numpy())
        self.vis.plot_lines('LSmth (LSmt)', LSmt.detach().cpu().numpy())
        self.vis.plot_lines('LSpars (LSpr)', LSpr.detach().cpu().numpy())
        self.vis.plot_lines('L (tmp)', L.detach().cpu().numpy())