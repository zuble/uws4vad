import torch
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, average_precision_score, roc_auc_score, recall_score, roc_curve, precision_recall_curve, auc

import os.path as osp , logging, matplotlib, time
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
import numpy, cv2, matplotlib.pyplot as plt
matplotlib.use('Agg')
#from prettytable import PrettyTable
import plotly.graph_objs as go, plotly.express as px
from plotly.subplots import make_subplots

from tabulate import tabulate
from PIL import Image

from src.utils import get_log
log = get_log(__name__)


class Metrics(object):
    def __init__(self, cfg_vldt, vis=None):

        self.vis = vis

        # Visualization and table flags
        self.xtra_mtrcs = cfg_vldt.get('extra_metrics', False)
        self.mtrc_vis_plot = cfg_vldt.get('mtrc_visplot', False)
        self.mtrc_vis_table = cfg_vldt.get('mtrc_vistabe', False)
        self.mtrc_save_table = cfg_vldt.get('mtrc_savetable', False)
        self.curv_vis_plot = cfg_vldt.get('curv_visplot', False)
        self.gtfl_vis_plot = cfg_vldt.get('gtfl_visplot', False)
        
        #self.net_name = cfg.model.net.id
        
    def calc_mtrc(self, gt, predictions, key):
        assert len(gt) == len(predictions), f'len GT {len(gt)} != len FL {len(predictions)}'
        
        fpr, tpr, thresholds = roc_curve(gt, predictions)
        auc_roc = auc(fpr, tpr)
        #auc_roc2 = roc_auc_score(gt, predictions)
        
        ap = average_precision_score(gt, predictions)
        
        precision, recall, thresholds = precision_recall_curve(gt, predictions)
        au_prc = auc(recall, precision)
        
        #self.mtrc_info[key] = { 'AP': [ap], 'AUC_PR': [au_prc], 'AUC_ROC': [auc_roc] }
        self.mtrc_info[key]['AP'].append(ap)
        self.mtrc_info[key]['AUC_PR'].append(au_prc)
        self.mtrc_info[key]['AUC_ROC'].append(auc_roc)

        if self.curv_vis_plot: 
            #self.curv_info[key] = {'precision':[precision], 'recall':[recall], 'fpr':[fpr], 'tpr':[tpr]}
            self.curv_info[key]['precision'].append(precision)
            self.curv_info[key]['recall'].append(recall)
            self.curv_info[key]['fpr'].append(fpr)
            self.curv_info[key]['tpr'].append(tpr)
        
        if self.xtra_mtrcs:
            ########
            ## F1
            ## https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
            log.info(f"F1 pre")
            
            #thresholds = numpy.array(thresholds)
            #bin_preds = (predictions[:, numpy.newaxis] >= thresholds).astype(int)
            #f1_scores = [f1_score(gt, bin_preds[:, i], pos_label=1) for i in range(bin_preds.shape[1])]
            
            #f1_scores = [f1_score(gt, (predictions >= thresh).astype(int), pos_label=1) for thresh in thresholds]
            
            log.info(f"F1 pos")
            
            optimal_f1 = max(f1_scores)
            optimal_threshold = thresholds[f1_scores.index(optimal_f1)]
            optimal_precision = precision[numpy.argmax(f1_scores)]
            optimal_recall = recall[numpy.argmax(f1_scores)]

            ## need to be correted because the dict is rewrited this way
            self.mtrc_info[key] = {
                #'F1': [f1_scores],
                'Optimal_F1': [optimal_f1],
                'Optimal_Threshold': [optimal_threshold],
                'Optimal_Precision': [optimal_precision],
                'Optimal_Recall': [optimal_recall]
            }
            
            if self.curv_vis_plot:  self.curv_info[key] = {'f1':[f1], 'ths':[ths]}


    def get_fl(self, vldt_info):
        """
        Entry point for calculating metrics. Determines whether to calculate
        label-wise metrics or video-wise metrics and calls the appropriate method.
        """
        self.lbls4plot = list(vldt_info.DATA.keys())
        
        if vldt_info.per_what == 'vid':
            ## rearranges temporaly vldt dict in order to get metrics per lbl
            #self.mtrc_info = {lbl: {} for lbl in self.lbls4plot}
            #self.curv_info = {lbl: {} for lbl in self.lbls4plot}
            #tmp=vldt_info.DATA
            #for lbl, metrics in tmp.items():
            #    metrics['GT'] = numpy.concatenate((metrics['GT']), axis=0)
            #    metrics['FL'] = np.concatenate((metrics['FL']), axis=0).asnumpy()
            #self.get_fl_lbl(tmp)
            #self.proc_mtrc_lbl()
            
            ## get metrics per vid
            ## reset dicts
            self.mtrc_info = {lbl: {'FN': [], 'AP': [], 'AUC_PR': [], 'AUC_ROC': []} for lbl in self.lbls4plot}
            #self.curv_info = {lbl: {} for lbl in self.lbls4plot}
            self.curv_info = {lbl: {'precision': [], 'recall': [], 'fpr': [], 'tpr': []} for lbl in self.lbls4plot}
            
            ## upgrade vldt_info , assures every element in GT/FL is numpy
            ## if check is for ccases when vldt_info is loaded from .pkl
            for lbl, metrics in vldt_info.DATA.items():
                if type(metrics['GT'][0]) != numpy.ndarray:
                    metrics['GT'] = [ numpy.array(gt) for gt in metrics['GT'] ]
                if type(metrics['FL'][0]) != numpy.ndarray:
                    metrics['FL'] = [ fl.asnumpy() for fl in metrics['FL'] ]
                    
            self.get_fl_vid(vldt_info.DATA)
            self.proc_mtrc_vid()
            
        else: ## glob && lbl
            self.mtrc_info = {lbl: {'AP': [], 'AUC_PR': [], 'AUC_ROC': []} for lbl in self.lbls4plot}
            self.curv_info = {lbl: {'precision': [], 'recall': [], 'fpr': [], 'tpr': []} for lbl in self.lbls4plot}
            #self.mtrc_info = {lbl: {} for lbl in self.lbls4plot}
            #self.curv_info = {lbl: {} for lbl in self.lbls4plot}
            self.get_fl_lbl(vldt_info.DATA)
            self.proc_mtrc_lbl()
        
        return self.mtrc_info
    
    #########################     
    ### FRAME LEVEL PER LABEL
    def get_fl_lbl(self, dict_data):
        ## https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
        for lbl, data in dict_data.items():
            log.debug(f"get_fl_lbl/{lbl}")
            log.debug(f"get_fl_lbl/data['GT']: {len(data['GT'])} {type(data['GT'])} {type(data['GT'][0])}")
            log.debug(f"get_fl_lbl/data['FL']: {len(data['FL'])} {type(data['FL'])} {type(data['FL'][0])}")
            self.calc_mtrc( data['GT'], data['FL'], lbl )
            
            #if lbl != 'ALL' and self.gtfl_vis_plot:
            #    self.plot_allflgt(data['GT'], data['FL'])
            ##########################################################
            ## maybe return metrics, if below a th send gtfl to visdom
            ## by calling plot_allflgt 
            ## can do the same for each fl_vid
            ##########################################################
            
        #log.debug(self.mtrc_info)    
    
    def proc_mtrc_lbl(self): 
        ############
        ## AU ROC/PR
        ## table-it (single table lbls/metrics) and sends always 2 log , nd in test can send 2 visdom depending on cfg.TEST.VLDT
        first_metrics = next(iter(self.mtrc_info.values()))
        headers = ["FL"] + list(first_metrics.keys()) ; rows=[]
        for label_str, metrics in self.mtrc_info.items():
            fmetrics = {k: f"{v[0]:.4f}" for k, v in metrics.items()}
            row = [f'{label_str}'] + list(fmetrics.values()) 
            rows.append(row) 
        table = tabulate(rows, headers, tablefmt="pretty")
        log.info(f'\n{table}')
        #self.tabler(table, 'FL')
        
        if self.mtrc_vis_plot: self.plot_metrics()
        
        ############ 
        ## ROC/PR curves
        if self.curv_vis_plot: self.plot_curves()
    
    
    #########################     
    ### FRAME LEVEL PER VIDEO    
    def get_fl_vid(self, dict_data):
        
        for lbl, data in dict_data.items():
            for fn, gt, fl in zip( data['FN'] , data['GT'] , data['FL']):
                log.debug(f"get_fl_vid/{lbl} {fn}")
                log.debug(f"get_fl_vid/data['GT']: {len(gt)} {type(gt)} {type(gt[0])}")
                log.debug(f"get_fl_vid/data['FL']: {len(fl)} {type(fl)} {type(fl[0])}")
                self.mtrc_info[lbl]['FN'].append(fn) ## same indexs of keys arrays belong to metrics of same video
                self.calc_mtrc(gt, fl, lbl)
    
    def proc_mtrc_vid(self):  
        ## table-it, 1 per lbl with all videos metrics ordered by AP high to low
        for class_name, metrics in self.mtrc_info.items():
            tmp = list(zip(metrics['FN'], metrics['AP'], metrics['AUC_PR'], metrics['AUC_ROC']))
            tmp = sorted(tmp, key=lambda x: x[1], reverse=True) ## 1-ap 2-au_prc
            metrics['FN'], metrics['AP'], metrics['AUC_PR'], metrics['AUC_ROC'] = zip(*tmp) 
            
            headers = [class_name] + list(metrics.keys())[1:] ; rows = []
            for i in range(len(metrics['FN'])):
                row_metrics = {k: f"{metrics[k][i]:.4f}" for k in headers[1:]}  
                row = [metrics['FN'][i]] + list(row_metrics.values())  
                rows.append(row)
            table = tabulate(rows, headers, tablefmt="pretty")
            log.info(f'\n{table}')
            self.tabler(table, class_name)

        #if self.mtrc_vis_plot: self.plot_metrics()
        
        ## return self.mtrc_info to watch the vid with inferior ap
        ## or pass the metrics class variable in vldt as it has the updated self.mtrc_info
    
    ######################
    ## plots
    def plot_allflgt(self,gt,fl):
        self.vis.close('ALLFLGT')
        log.info("plot_allflgt: generating allflgt vis plot...")
        
        xticks = numpy.arange(len(fl))  
        
        colors = px.colors.qualitative.Plotly
        color = colors[0 % len(colors)]
        fig = make_subplots(rows=1, cols=1, subplot_titles=('ALLFLGT',))
        
        fl_trace = go.Scatter(
            x=xticks, 
            y=fl,
            mode='lines', name='ALLFLGT', line=dict(color=color, width=2) )
        fig.add_trace(fl_trace)

        gt_trace = go.Scatter(
            x=xticks,
            y=gt,
            mode='none',
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)',  # Red with opacity
            showlegend=False
        )
        fig.add_trace(gt_trace)
            
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="AS")
        fig.update_layout(height=500, showlegend=True, title_text="ALLFLGT")
        
        self.vis.potly(fig)
        
    def plot_curves(self):
        ## create everything with px ????
        ## https://plotly.com/python/plotly-express/
        colors = px.colors.qualitative.Plotly  # Or any other color scale

        self.vis.close(f'ROC Curves'); self.vis.close(f'Precision-Recall Curves') ## starts fresh
        
        roc_fig = make_subplots( rows=1, cols=1, subplot_titles=('ROC Curves',))
        pr_fig = make_subplots(rows=1, cols=1, subplot_titles=('Precision-Recall Curves',))
        if self.xtra_mtrcs: f1_fig = make_subplots(rows=1, cols=1, subplot_titles=('F1 Curves',))
        
        for idx, (lbl, data) in enumerate(self.curv_info.items()):
            color = colors[idx % len(colors)]

            roc_trace = go.Scatter(
                x=data['fpr'][0], 
                y=data['tpr'][0],
                mode='lines', name=f'ROC {lbl}', line=dict(color=color, width=2) )
            roc_fig.add_trace(roc_trace)

            pr_trace = go.Scatter(
                x=data['recall'][0],
                y=data['precision'][0],
                mode='lines', name=f'PR {lbl}', line=dict(color=color, width=2) )
            pr_fig.add_trace(pr_trace)
            
            if self.xtra_mtrcs: 
                pr_trace = go.Scatter(
                    x=data['ths'][0],
                    y=data['f1'][0],
                    mode='lines', name=f'F1 {lbl}', line=dict(color=color, width=2) )
                pr_fig.add_trace(pr_trace)

        roc_fig.update_xaxes(title_text="False Positive Rate")
        roc_fig.update_yaxes(title_text="True Positive Rate")
        roc_fig.update_layout( height=500, showlegend=True, title_text="ROC Curves" )
        self.vis.potly(roc_fig)
        
        pr_fig.update_xaxes(title_text="Recall")
        pr_fig.update_yaxes(title_text="Precision")
        pr_fig.update_layout(height=500, showlegend=True, title_text="Precision-Recall Curves" )
        self.vis.potly(pr_fig)
        
        if self.xtra_mtrcs: 
            f1_fig.update_xaxes(title_text="Thresholds")
            f1_fig.update_yaxes(title_text="F1")
            f1_fig.update_layout(height=500, showlegend=True, title_text="F1 Curves" )
            self.vis.potly(f1_fig)
    
    def plot_metrics(self):
        ## 1 plot per metric, xaxis is lbls
        metrics_data = { 'AP': [], 'AUC_PR': [], 'AUC_ROC': [] }
        
        self.vis.close(f'AP');self.vis.close(f'AUC_PR');self.vis.close(f'AUC_ROC')
        
        
        for label_str, metrics in self.mtrc_info.items():
            #log.debug(f"{label_str} {metrics = }")
            for metric_name in metrics_data.keys():
                metrics_data[metric_name].extend(metrics[metric_name])
                
        xtickvals = list(range(1, len(self.lbls4plot) + 1))
        xticklabels = self.lbls4plot
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            scatter_data = numpy.column_stack((xtickvals, values))
            opts = dict(title=f'{str(metric_name)}',
                        #legend=self.lbls4plot, #markersize=10, #markercolor=colors,
                        xtickvals=xtickvals, xticklabels=xticklabels)
            self.vis.scatter(scatter_data, None, opts)

    def tabler(self, table, name):
        ## transforms table into img
        fig, ax = plt.subplots(figsize=(7,3))
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, clip_on=False, color='black', zorder=-1))
        ax.text(0.0,1.0,str(table),horizontalalignment='left',verticalalignment='top',fontsize=12,family='monospace', color='white')
        plt.gca().set_facecolor('black')
        ax.axis('off')
        try:
            if self.mtrc_save_table:
                plt.savefig( osp.join(self.cfg.path.out_dir,f'{name}.png'), dpi=300, bbox_inches='tight', facecolor='black')
            if self.mtrc_vis_table: ## send table_img to vis
                self.vis.close(name) ## starts fresh
                fig.canvas.draw()
                table_img = numpy.frombuffer(fig.canvas.tostring_rgb(), dtype=numpy.uint8)
                table_img = table_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                #plt.imsave(osp.join(self.cfg.path.out_dir,'table_np.png'), table_img)
                opts=dict(title=name, caption=name, store_history=False)
                self.vis.disp_image(table_img.transpose((2, 0, 1)), name, opts)
        finally:
            plt.close(fig)    


#########
## OLDIES not in use 
'''
def plot_cm(self, name, labels, predictions, threshold=0.5, save=False):
    """
    https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#download_the_kaggle_credit_card_fraud_data_set
    """
    #predictions = np.array(predictions)
    #cm = confusion_matrix(labels, predictions > threshold)
    #plt.clf()
    #plt.figure(figsize=(5,5))
    #sns.heatmap(cm, annot=True, fmt="d")
    #plt.title('Confusion matrix @{:.2f}'.format(threshold))
    #plt.ylabel('Actual label')
    #plt.xlabel('Predicted label')
    #if save: plt.savefig(name+'.png',facecolor='white', transparent=False)
    #plt.show()
    return

def plot_roc(self, name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, color=colors[0], linestyle='--')
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,80])
    plt.ylim([20,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.savefig(name+'.png',facecolor='white', transparent=False)
    plt.show()

def plot_prc(self, name, labels, predictions, **kwargs):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)
    #plt.clf()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plt.plot(precision, recall, label=name, linewidth=2, color=colors[0])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.xlim([-0.5,100.5])
    plt.ylim([-0.5,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.savefig(name+'.png',facecolor='white', transparent=False)
    plt.show()
'''
#########