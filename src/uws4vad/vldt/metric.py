import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, average_precision_score, roc_auc_score, recall_score, roc_curve, precision_recall_curve, auc

import os.path as osp, logging, time, copy

import matplotlib
logging.getLogger('matplotlib').setLevel(logging.WARNING)
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
logging.getLogger('PIL').setLevel(logging.WARNING)
#from PIL import Image

#from prettytable import PrettyTable
from tabulate import tabulate, SEPARATING_LINE
import plotly.graph_objs as go, plotly.express as px
from plotly.subplots import make_subplots

from collections import defaultdict
from uws4vad.utils import get_log
log = get_log(__name__)


class MetricCalculator:
    def calc_metrics(self, gt, preds, key, mtrc_info, curv_info, xtra_mtrcs=False):
        assert len(gt) == len(preds), f'len GT {len(gt)} != len FL {len(preds)}'

        fpr, tpr, _ = roc_curve(gt, preds)
        auc_roc = auc(fpr, tpr)
        #auc_roc2 = roc_auc_score(gt, preds)
        
        ap = average_precision_score(gt, preds) ## <- tend to give lower values
        
        precision, recall, _ = precision_recall_curve(gt, preds)
        au_prc = auc(recall, precision) ## <- trapezoid rule is optimistic

        mtrc_info[key]['AP'].append(ap)
        mtrc_info[key]['AUC-PR'].append(au_prc)
        mtrc_info[key]['AUC-ROC'].append(auc_roc)

        curv_info[key]['precision'].append(precision)
        curv_info[key]['recall'].append(recall)
        curv_info[key]['fpr'].append(fpr)
        curv_info[key]['tpr'].append(tpr)

        if xtra_mtrcs:
            raise NotImplementedError
            ########
            ## F1
            ## https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
            log.info(f"F1 pre")
            
            #thresholds = np.array(thresholds)
            #bin_preds = (preds[:, np.newaxis] >= thresholds).astype(int)
            #f1_scores = [f1_score(gt, bin_preds[:, i], pos_label=1) for i in range(bin_preds.shape[1])]
            
            #f1_scores = [f1_score(gt, (preds >= thresh).astype(int), pos_label=1) for thresh in thresholds]
            log.info(f"F1 pos")

            optimal_f1 = max(f1_scores)
            optimal_threshold = thresholds[f1_scores.index(optimal_f1)]
            optimal_precision = precision[np.argmax(f1_scores)]
            optimal_recall = recall[np.argmax(f1_scores)]

            ## need to be correted because the dict is rewrited this way
            mtrc_info[key] = {
                #'F1': [f1_scores],
                'Optimal_F1': [optimal_f1],
                'Optimal_Threshold': [optimal_threshold],
                'Optimal_Precision': [optimal_precision],
                'Optimal_Recall': [optimal_recall]
                }
            if curv_vis_plot:curv_info[key] = {'f1':[f1], 'ths':[ths]}

    def calc_far(self, gt, preds, key, mtrc_info, th=0.5):
        """Calculates the False Alarm Rate (FAR). (from PEL4VAD)"""
        preds[preds < th] = 0
        preds[preds >= th] = 1
        tn, fp, _, _ = confusion_matrix(gt, preds, labels=[0, 1]).ravel()
        far = fp / (fp + tn)
        mtrc_info[key]['FAR'].append(far)


class DataHandler:
    """Organizes metrics data for label-wise and video-wise analysis."""
    def __init__(self, xtra_mtrcs):
        self.mtrc_info = {}
        self.curv_info = {}
        self.xtra_mtrcs = xtra_mtrcs
        self.calculator = MetricCalculator()

    def proc_by_video(self, dict_data):
        """Organizes data for video-wise analysis."""
        self.lbls4plot = list(dict_data.keys())
        log.debug(f"{self.lbls4plot=}")
        
        self.mtrc_info = {self.lbls4plot[0]: {'FN': [], 'FAR': []} }
        self.mtrc_info.update( {lbl: {'FN': [], 'AP': [], 'AUC-PR': [], 'AUC-ROC': []} for lbl in self.lbls4plot[1:]} )
            
        self.curv_info = {lbl: {'precision': [], 'recall': [], 'fpr': [], 'tpr': []} for lbl in self.lbls4plot[1:]}
        
        for i, (lbl, data) in enumerate(dict_data.items()):
            for fn, gt, fl in zip( data['FN'] , data['GT'] , data['FL']):
                #log.warning(f"*****")
                #log.warning(f"get_fl_vid/vldt_info.DATA/{lbl} {fn}")
                #log.debug(f"get_fl_vid/vldt_info.DATA/GT: {len(gt)} ") #{type(gt)} {type(gt[0])}
                #log.debug(f"get_fl_vid/vldt_info.DATA/FL: {len(fl)} ") #{type(fl)} {type(fl[0])}
                self.mtrc_info[lbl]['FN'].append(fn) ## same indexs of keys arrays belong to metrics of same video
                ## 000.NORM
                if not i:  self.calculator.calc_far(gt, fl, lbl, self.mtrc_info)
                ## populate same idx with mtrc values
                else: self.calculator.calc_metrics(gt, fl, lbl, self.mtrc_info, self.curv_info, self.xtra_mtrcs) 
        
        for lbl_name, metrics in self.mtrc_info.items():
            log.debug(f"*****")
            for k in metrics.keys():
                log.debug(f"self.mtrc_info/{lbl_name}/{k}: {len(metrics[k])} {type(metrics[k])}")
                
        ## self.mtrc_info/000.NORM/FN: 150 <class 'list'>
        ## self.mtrc_info/000.NORM/FAR: 150 <class 'list'>
        ## self.mtrc_info/8.ROADACC/FN: 23 <class 'list'>
        ## self.mtrc_info/8.ROADACC/AP: 23 <class 'list'>
        ## self.mtrc_info/8.ROADACC/AUC-PR: 23 <class 'list'>
        ## self.mtrc_info/8.ROADACC/AUC-ROC: 23 <class 'list'>        
            
    def proc_by_label(self, dict_data):
        """Organizes data for label-wise analysis."""
        self.lbls4plot = list(dict_data.keys())
        log.debug(f"{self.lbls4plot=}")

        self.mtrc_info = {self.lbls4plot[0]: {'FAR': []}}
        self.mtrc_info.update({lbl: {'AP': [], 'AUC-PR': [], 'AUC-ROC': []} for lbl in self.lbls4plot[1:]})

        self.curv_info = {lbl: {'precision': [], 'recall': [], 'fpr': [], 'tpr': []} for lbl in self.lbls4plot[1:]}
        
        for i, (lbl, data) in enumerate(dict_data.items()):
            #log.debug(f"get_fl_lbl/{lbl}: 'GT': {len(data['GT'])}") # {type(data['GT'])} {type(data['GT'][0])}
            #log.debug(f"get_fl_lbl/{lbl}: 'FL': {len(data['FL'])}") # {type(data['FL'])} {type(data['FL'][0])}
            if not i:  # lbl == '000.NORM'
                self.calculator.calc_far(data['GT'], data['FL'], lbl, self.mtrc_info)
                
            else: self.calculator.calc_metrics(data['GT'], data['FL'], lbl, self.mtrc_info, self.curv_info, self.xtra_mtrcs)
    
    def reset(self): self.mtrc_info,self.curv_info={},{}

class Tabler:
    def __init__(self, cfg_vldt, vis=None):
        self.sort_mtrc = cfg_vldt.record_mtrc
        self.full_lbls = cfg_vldt.get('full_anom_lbls', False)
        self.save = cfg_vldt.get('mtrc_savetable', False)
        self.send2visdom = cfg_vldt.get('mtrc_vistable', False)
        self.vis = vis
        self.stat_measures = {
            'std': lambda x: np.std(x),
            'mean': np.mean,
            'median': np.median,
            'range': lambda x: np.ptp(x)
        }
        
    def log_per_lbl(self, mtrc_info): 
        ## AU ROC/PR(single table lbls/metrics) and sends always 2 log , nd in test can send 2 visdom depending on cfg.TEST.VLDT
        mtrc_names = list(next(iter( list(mtrc_info.values())[1:] )).keys()) ## AP AUC-PR AUC-ROC
        headers = ["FL"] + mtrc_names ; rows=[]
        ## FL AP AUC-PR AUC-ROC
        for i, (lbl_name, metrics) in enumerate(mtrc_info.items()):
            if lbl_name != "000.NORM": 
                fmetrics = {k: f"{v[0]:.4f}" for k, v in metrics.items()}
                row = [f'{lbl_name}'] + list(fmetrics.values()) 
                rows.append(row) 
            else:
                tmp_row = [f'{lbl_name}~FAR', f"{metrics['FAR'][0]:.4f}"]
                tmp_row += ['-'] * (len(mtrc_names) - 1)
                
        rows.append([' '] * (len(mtrc_names) + 1))        
        rows.append(tmp_row)
        
        table = tabulate(rows, headers, tablefmt="pretty")
        #self.table2img(table, 'FL')
        return [table]
    
    def log_per_lbl_per_ckpt(self, rslt_infos, ret=False):
        """Log label-wise metrics across multiple checkpoints with statistical analysis"""
        
        ckpt_ids = [self._shorten_path(res.ckpt_path) for res in rslt_infos]
        metric_data = defaultdict(lambda: defaultdict(list))
        
        for result in rslt_infos:
            for lbl, metrics in result.mtrc_info.items():
                if lbl == "000.NORM": 
                    continue  # Handle norm separately
                for mtrc, values in metrics.items():
                    if mtrc in ['AP', 'AUC-PR', 'AUC-ROC']:
                        metric_data[lbl][mtrc].append(values[0])  # Assuming first value is main metric

        # Create sorted structure with statistics
        sorted_data = []
        for lbl, metrics in metric_data.items():
            stat_metrics = {
                'label': lbl,
                'values': {},
                'stats': {}
            }
            
            for mtrc, values in metrics.items():
                stat_metrics['values'][mtrc] = values
                stat_metrics['stats'][mtrc] = {
                    k: f"{func(values):.3f}" 
                    for k, func in self.stat_measures.items()
                }
            
            # Calculate sorting key (mean + std)
            sort_values = metrics.get(self.sort_mtrc, [])
            stat_metrics['sort_key'] = (
                -np.mean(sort_values),  # Primary sort: descending mean
                np.std(sort_values)    # Secondary sort: ascending std
            )
            sorted_data.append(stat_metrics)

        sorted_data.sort(key=lambda x: x['sort_key'])

        ## Build table
        headers = ["Label", f"Mean({self.sort_mtrc})", f"Std({self.sort_mtrc})"] + ckpt_ids
        rows = []
        for item in sorted_data:
            main_metric = item['values'][self.sort_mtrc]
            row = [
                item['label'],
                f"{np.mean(main_metric):.3f}",
                f"{np.std(main_metric):.3f}"
            ]
            
            # Add individual model values
            for model_val in main_metric:
                row.append(f"{model_val:.3f}")
                
            rows.append(row)

        ## Add Norm row at bottom
        norm_data = next((res.mtrc_info["000.NORM"] for res in rslt_infos), None)
        if norm_data:
            far_values = [res.mtrc_info["000.NORM"]['FAR'][0] for res in rslt_infos]
            rows.append([
                "000.NORM~FAR", 
                f"{np.mean(far_values):.3f}",
                f"{np.std(far_values):.3f}"
            ] + [f"{v:.3f}" for v in far_values])

        table = tabulate(rows, headers, tablefmt="pretty", stralign="right")
        log.info(f'\n=== Aggregated Metrics (Sorted by {self.sort_mtrc} variability) ===\n{table}')
        if ret: return table


    def _shorten_path(self, ckpt_path):
        return ckpt_path.split('/')[-1].split('.')[0]
        #return osp.basename(ckpt_path).split('.')[0]
        
    def log_per_vid(self, mtrc_info): 
        ## table-it, 1 per lbl with all videos metrics ordered by sort_mtrc provided high to low
        tables = []
        for i, (lbl_name, metrics) in enumerate(mtrc_info.items()):
            if lbl_name == "000.NORM":
                # Always process and print 000.NORM
                headers = [lbl_name, 'FAR']; rows = []
                # Sort by FAR in ascending order for "000.NORM"
                tmp = list(zip(metrics['FN'], metrics['FAR']))
                tmp.sort(key=lambda x: x[1])
                metrics['FN'], metrics['FAR'] = zip(*tmp)
                
                for i in range(len(metrics['FN'])):
                    row = [metrics['FN'][i], f"{metrics['FAR'][i]:.4f}"]
                    rows.append(row)
                
                table = tabulate(rows, headers, tablefmt="pretty")
                #log.info(f'\n{table}')
                #self.table2img(table, lbl_name)
                tables.append(table)
            else:
                if not self.full_lbls and lbl_name != "111.ANOM": continue
                
                # Process the label (either 111.ANOM when self.full_lbls is False or any label when self.full_lbls is True)
                tmp = list(zip(metrics['FN'], metrics['AP'], metrics['AUC-PR'], metrics['AUC-ROC']))
                
                sort_index = list(metrics.keys()).index(self.sort_mtrc)
                tmp = sorted(tmp, key=lambda x: x[sort_index], reverse=True)
                # Unzip the sorted tuples back into the metrics dictionary
                metrics['FN'], metrics['AP'], metrics['AUC-PR'], metrics['AUC-ROC'] = zip(*tmp)
                
                headers = [lbl_name] + list(metrics.keys())[1:]; rows = []
                for i in range(len(metrics['FN'])):
                    row_metrics = {k: f"{metrics[k][i]:.4f}" for k in headers[1:]}  
                    row = [metrics['FN'][i]] + list(row_metrics.values())  
                    rows.append(row)
                
                table = tabulate(rows, headers, tablefmt="pretty")
                #log.info(f'\n{table}')
                #self.table2img(table, lbl_name)
                tables.append(table)
        return tables
    
    def log_per_vid_per_ckpt(self, rslt_infos, ret=False):
        """Log video-wise metrics across multiple checkpoints with statistical analysis"""
        ckpt_ids = [self._shorten_path(res.ckpt_path) for res in rslt_infos]
        
        ## First, collect all unique videos and their metrics across checkpoints
        video_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for result in rslt_infos:
            for lbl_name, metrics in result.mtrc_info.items():
                if lbl_name == "000.NORM":
                    # Handle NORM videos separately
                    for vid_idx, (vid_name, far_val) in enumerate(zip(metrics['FN'], metrics['FAR'])):
                        video_metrics[lbl_name][vid_name]['FAR'].append(far_val)
                else:
                    # Skip non-ANOM labels if self.full_lbls is False
                    if not self.full_lbls and lbl_name != "111.ANOM": continue
                    for vid_idx, vid_name in enumerate(metrics['FN']):
                        for metric_name in ['AP', 'AUC-PR', 'AUC-ROC']:
                            video_metrics[lbl_name][vid_name][metric_name].append(metrics[metric_name][vid_idx])
                            
        tables = []
        
        # Always process NORM first
        if "000.NORM" in video_metrics:
            norm_headers = ["000.NORM", "Mean(FAR)", "Std(FAR)"] + ckpt_ids
            norm_rows = []
            
            for vid_name, metrics in video_metrics["000.NORM"].items():
                far_values = metrics['FAR']
                row = [
                    vid_name,
                    f"{np.mean(far_values):.4f}",
                    f"{np.std(far_values):.4f}"
                ] + [f"{v:.4f}" for v in far_values]
                norm_rows.append(row)
            
            norm_rows.sort(key=lambda x: float(x[2])) ## Sort by std FAR
            norm_table = tabulate(norm_rows, norm_headers, tablefmt="pretty", stralign="right")
            tables.append(norm_table)
            
        ## Process other labels
        for lbl_name, lbl_data in video_metrics.items():
            if lbl_name == "000.NORM": continue
                
            headers = [lbl_name, f"Mean({self.sort_mtrc})", f"Std({self.sort_mtrc})"] + ckpt_ids
            rows = []
            
            for vid_name, metrics in lbl_data.items():
                main_metric = metrics[self.sort_mtrc]
                row = [
                    vid_name,
                    f"{np.mean(main_metric):.4f}",
                    f"{np.std(main_metric):.4f}"
                ] + [f"{v:.4f}" for v in main_metric]
                
                #rows.append(((-np.mean(main_metric), np.std(main_metric)), row))
                rows.append((np.std(main_metric), row))  # only store std as sorting key
            
            # Sort rows by the sorting key (same as in log_per_vid: high to low)
            rows.sort(key=lambda x: x[0])
            rows = [row for _, row in rows]
            
            table = tabulate(rows, headers, tablefmt="pretty", stralign="right")
            tables.append(table)
        
        for table in tables: log.info(f'\n{table}')
        if ret: return tables

    def table2img(self, table, name, send2visdom=False):
        ## transforms table into img
        fig, ax = plt.subplots(figsize=(7,3))
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, clip_on=False, color='black', zorder=-1))
        ax.text(0.0,1.0,str(table),horizontalalignment='left',verticalalignment='top',fontsize=12,family='monospace', color='white')
        plt.gca().set_facecolor('black')
        ax.axis('off')
        try:
            #if self.save: plt.savefig( osp.join(self.cfg.path.out_dir,f'{name}.png'), dpi=300, bbox_inches='tight', facecolor='black')
            if self.send2visdom or send2visdom: ## send table_img to vis
                #self.vis.close(name) ## starts fresh
                fig.canvas.draw()
                buffer = fig.canvas.tostring_argb()
                width, height = fig.canvas.get_width_height()
                table_img = np.frombuffer(buffer, dtype=np.uint8)
                table_img = table_img.reshape((height, width, 4)) # Reshape for ARGB
                table_img = table_img[..., 1:]  # Drops alpha (first channel in ARGB)
                table_img = table_img.transpose((2, 0, 1))  # Transpose to (C, H, W) for Visdom
                opts = dict(title=name, caption=name, store_history=False)
                self.vis.disp_image(table_img, name, opts)
                ## working w/ matplotlib 3.7.0
                # fig.canvas.draw()
                # table_img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
                # table_img = table_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                # #plt.imsave(osp.join(self.cfg.path.out_dir,'table_np.png'), table_img)
                # opts=dict(title=name, caption=name, store_history=False)
                # self.vis.disp_image(table_img.transpose((2, 0, 1)), name, opts)
        finally:
            plt.close(fig)    



######################################################################
class Metrics:
    def __init__(self, cfg_vldt, vis=None): 
        self.tblr_full_anom =  cfg_vldt.get('full_anom_lbls', False) 
        self.mtrc_vis_plot = cfg_vldt.get('mtrc_visplot', False) 
        self.mtrc_vis_plot_per_epo = cfg_vldt.get('mtrc_visplot_epo', False) #train
        self.gtfl_vis_plot = cfg_vldt.get('gtfl_visplot', False) ## not yet
        
        self.vis = vis
        self.datler = DataHandler( cfg_vldt.get('extra_metrics', False) )
        self.tabler = Tabler(cfg_vldt)
        self.plotter = Plotter(vis)
        
        #self.net_name = cfg.model.net.id

    def get_fl(self, vldt_info):
        """
        Entry point for calculating metrics. Determines whether to calculate
        label-wise metrics or video-wise metrics and calls the appropriate method.
        """
        #self.lbls4plot = list(vldt_info.DATA.keys())
        #log.debug(f"{self.lbls4plot=}")
        
        if vldt_info.per_what == 'vid': ## only4test
            ## lbl: 000.NORM | B1.FIGHT | B2.SHOOT | B4.RIOT | B5.ABUSE | B6.CARACC | G.EXPLOS | 111.ANOM | ALL
            self.datler.proc_by_video(vldt_info.DATA)
            
            table = self.tabler.log_per_vid(self.datler.mtrc_info)
            
            if self.mtrc_vis_plot: 
                log.error("mtrc_visplot not implemented")
                #raise NotImplementedError

        else: ## glob && lbl
            ## glob: 000.NORM | 111.ANOM | ALL
            ## lbl: 000.NORM | B1.FIGHT | B2.SHOOT | B4.RIOT | B5.ABUSE | B6.CARACC | G.EXPLOS | 111.ANOM | ALL
            self.datler.proc_by_label(vldt_info.DATA)
            
            table = self.tabler.log_per_lbl( self.datler.mtrc_info ) 
            
            if self.mtrc_vis_plot_per_epo: 
                self.plotter.metrics_per_epo( self.datler.mtrc_info )
                self.plotter.metrics_all_labels_per_epo( self.datler.mtrc_info  )
            
            ## better iterate outside of here
            ## nevertheless differetn from testplotter
            ## this works as bulk
            for i, (lbl, data) in enumerate(vldt_info.DATA.items()):
                if lbl != 'ALL' and self.gtfl_vis_plot:
                    self.plotter.allflgt(data['GT'], data['FL'], lbl)
                    
            #log.debug(f"{self.datler.mtrc_info=}")    
            #log.debug(f"{self.datler.curv_info=}")
        
        return copy.deepcopy(self.datler.mtrc_info), copy.deepcopy(self.datler.curv_info), table
        # return {
        #     'mtrc_info': self.datler.mtrc_info,
        #     'curv_info': self.datler.curv_info, 
        #     'table_res': table
        # }
    
    def reset(self): self.datler.reset()
######################################################################


class Plotter:
    def __init__(self, vis):
        self.vis = vis  
        self.full_metric_names = {
            'FAR': 'False Alarm Rate (FAR)',
            'AP': 'Average Precision (AP)',
            'AUC-PR': 'Area Under Precision-Recall Curve (AUC-PR)',
            'AUC-ROC': 'Area Under ROC Curve (AUC-ROC)'
            }
        self.all_labels_per_epo_index = {}
        
    def allflgt(self,gt,fl, lbl):
        #self.vis.close('ALLFLGT')
        log.info(f"plot_allflgt: generating asgt {lbl} vis plot...")
        
        xticks = np.arange(len(fl))  
        
        colors = px.colors.qualitative.Plotly
        color = colors[0 % len(colors)]
        fig = make_subplots(rows=1, cols=1, subplot_titles=('ALLFLGT',))
        
        fl_trace = go.Scatter(
            x=xticks, 
            y=fl,
            mode='lines', name='Anomaly Score (AS)', line=dict(color=color, width=2) )
        fig.add_trace(fl_trace)

        gt_trace = go.Scatter(
            x=xticks,
            y=gt,
            mode='none',
            name='Ground Thruth (GT)',
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)',  # Red with opacity
            showlegend=False
        )
        fig.add_trace(gt_trace)
            
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="AS")
        fig.update_layout(height=500, showlegend=True, title_text=f"ASGT - {lbl}")
        
        self.vis.potly(fig)
        
    def curves(self, curv_info):
        ## create everything with px ????
        ## https://plotly.com/python/plotly-express/
        log.debug("Sending per-label curves 2 visdom")
        colors = px.colors.qualitative.Plotly  # Or any other color scale

        ## starts fresh not needed as this runs for test and end epo
        #self.vis.close(f'ROC Curves'); self.vis.close(f'Precision-Recall Curves') 
        
        roc_fig = make_subplots( rows=1, cols=1,subplot_titles=('ROC Curves per Class',) ) #
        pr_fig = make_subplots(rows=1, cols=1, subplot_titles=('PR Curves per Class',)) #
        #if self.xtra_mtrcs: f1_fig = make_subplots(rows=1, cols=1, subplot_titles=('F1 Curves',))
        
        for idx, (lbl, data) in enumerate(curv_info.items()):
            color = colors[idx % len(colors)]

            roc_trace = go.Scatter(
                x=data['fpr'][0], 
                y=data['tpr'][0],
                mode='lines', name=f'{lbl.split(".")[-1]}', line=dict(color=color, width=2) )
            roc_fig.add_trace(roc_trace)

            pr_trace = go.Scatter(
                x=data['recall'][0],
                y=data['precision'][0],
                mode='lines', name=f'{lbl.split(".")[-1]}', line=dict(color=color, width=2) )
            pr_fig.add_trace(pr_trace)
            
            #if self.xtra_mtrcs: 
            #    pr_trace = go.Scatter(
            #        x=data['ths'][0],
            #        y=data['f1'][0],
            #        mode='lines', name=f'F1 {lbl}', line=dict(color=color, width=2) )
            #    pr_fig.add_trace(pr_trace)

        roc_fig.update_xaxes(title_text="False Positive Rate (FPR)")
        roc_fig.update_yaxes(title_text="True Positive Rate (TPR)")
        roc_fig.update_layout( height=500, showlegend=True,  )#title_text="ROC Curves per Class"
        self.vis.potly(roc_fig)
        
        pr_fig.update_xaxes(title_text="Recall")
        pr_fig.update_yaxes(title_text="Precision")
        pr_fig.update_layout(height=500, showlegend=True,  )#title_text="PR Curves per Class"
        self.vis.potly(pr_fig)
        
        #if self.xtra_mtrcs: 
        #    f1_fig.update_xaxes(title_text="Thresholds")
        #    f1_fig.update_yaxes(title_text="F1")
        #    f1_fig.update_layout(height=500, showlegend=True, title_text="F1 Curves" )
        #    self.vis.potly(f1_fig)
    
    ## run at end o fepo with best state
    ## 1 plot per metric, xaxis is lbls
    def metrics(self, mtrc_info):
        log.debug("Sending per-label metrics 2 visdom")
        
        metrics_data = { 'FAR':[], 'AP':[], 'AUC-PR':[], 'AUC-ROC':[] }

        self.vis.close(f'AP')
        self.vis.close(f'AUC-PR')
        self.vis.close(f'AUC-ROC')
        
        for lbl_name, metrics in mtrc_info.items():
            log.debug(f"Plotter/metrics: {lbl_name} {metrics = }")
        
        lbls4plot=[]
        for lbl_name, metrics in mtrc_info.items():
            lbls4plot.append(lbl_name.split('.')[-1])
            if lbl_name == '000.NORM':
                metrics_data['FAR'].extend(metrics['FAR'])
            else:
                for metric_name in ['AP', 'AUC-PR', 'AUC-ROC']:  
                    metrics_data[metric_name].extend(metrics[metric_name])
        
        #xticklabels = lbls4plot  
        #xtickvals = list(range(1, len(xticklabels) + 1))
        #for i, (metric_name, values) in enumerate(metrics_data.items()):
        #    scatter_data = np.column_stack((xtickvals[:len(values)], values))
        #    opts = dict(
        #        title=f'Best {str(metric_name)} per Label',
        #        xtickvals=xtickvals[:len(values)], 
        #        xticklabels=xticklabels[:len(values)]
        #    )
        #    self.vis.scatter(scatter_data, None, opts)
        
        # abnormal + ALL
        xticklabels = lbls4plot[1:]        
        xtickvals = list(range(1, len(xticklabels) + 1))
        #log.debug(f"{metrics_data} ")
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            full_metric_name = self.full_metric_names[metric_name]
            if metric_name == 'FAR':
                scatter_data = np.column_stack(([1], values))  # Use [1] for x-coordinate
                opts = dict(
                    title=f'{str(metric_name)} - Best Abnormal Epoch',
                    xtickvals=[1],  # Only one tick for FAR
                    xticklabels=lbls4plot[0],
                    ylabel=full_metric_name
                )
            else:
                scatter_data = np.column_stack((xtickvals, values))
                opts = dict(
                    title=f'{str(metric_name)} per Class - Best Abnormal Epoch',
                    #legend=self.lbls4plot,
                    #markersize=10, markercolor=colors,
                    markersymbol='diamond-wide',
                    xtickvals=xtickvals,
                    xticklabels=xticklabels,
                    ylabel=full_metric_name
                )
            self.vis.scatter(scatter_data, None, opts)
            
    ## 1 plot per metric per label, xaxis is epochs
    def metrics_per_epo(self, mtrc_info):
        """Plots metrics per epoch in Visdom."""
        log.debug("Sending per-label metrics 2 visdom")
        
        for lbl_name, metrics in mtrc_info.items():
            if '000' in lbl_name:
                wname = f"epo-{lbl_name}"
                self.vis.plot_lines(
                    wname,
                    metrics['FAR'][0],
                    opts=dict(
                        title=f"FAR per Epoch",  
                        xlabel='Validation Epoch',
                        ylabel=self.full_metric_names['FAR']
                    )
                )
            else:
                for metric_name in ['AP', 'AUC-PR', 'AUC-ROC']:
                    full_metric_name = self.full_metric_names[metric_name]
                    wname =  f"epo-{lbl_name}-{metric_name}"
                    self.vis.plot_lines(
                        wname,
                        metrics[metric_name][0],
                        opts=dict(
                            title=f"{metric_name} per Epoch - {lbl_name.split('.')[-1].capitalize()} Set", 
                            xlabel='Validation Epoch',
                            ylabel=full_metric_name 
                        )
                    )
    ## 1 plot per metric with multiple label lines, xaxis is epochs    
    def metrics_all_labels_per_epo(self, mtrc_info):
        """Plots all labels for each metric on a single plot per epoch."""
        for metric_name in ['AP', 'AUC-PR', 'AUC-ROC']:
            full_metric_name = self.full_metric_names[metric_name]
            win_name = f"epo-all-{metric_name}"

            # Collect data for the current metric from all labels
            y_data = []
            for lbl_name, metrics in mtrc_info.items():
                if metric_name in metrics:
                    y_data.append(metrics[metric_name][-1])
                #else:
                #    y_data.append(None) 

            x = self.all_labels_per_epo_index.get(win_name, 0)
            if x == 0: 
                self.vis.vis.line(
                    X=np.array([x]),
                    Y=np.column_stack(y_data),
                    win=win_name,
                    opts=dict(
                        title=f"{metric_name} per Epoch",
                        xlabel='Validation Epoch',
                        ylabel=full_metric_name,
                        legend=[lbl.split(".")[-1] for lbl in mtrc_info if metric_name in mtrc_info[lbl]]
                    )
                )
            else: 
                self.vis.vis.line(
                    X=np.array([x]),
                    Y=np.column_stack(y_data),
                    win=win_name,
                    update='append'
                )
            self.all_labels_per_epo_index[win_name] = x + 1
