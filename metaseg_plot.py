

import os
from metaseg_io import metaseg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, kde
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc
import numpy as np



def add_scatterplot_vs_iou(ious, sizes, dataset, shortname, size_fac, scale, setylim=True):
  
  cmap=plt.get_cmap('tab20')
  rho = pearsonr(ious,dataset)
  plt.title(r"$\rho = {:.05f}$".format(rho[0]))
  plt.scatter(ious, dataset, s = sizes/np.max(sizes)*size_fac, linewidth=.5, c=cmap(0), edgecolors=cmap(1), alpha=0.25 )
  plt.xlabel('$\mathit{IoU}_\mathrm{adj}$', labelpad=-10)
  plt.ylabel(shortname, labelpad=-8)
  plt.ylim(-.05,1.05)
  plt.xticks((0,1),fontsize=10*scale)
  plt.yticks((0,1),fontsize=10*scale)
  

  
def make_scatterplots(save_dir, df_full, df_full_nei, filename='iou_vs_ucm_allcls.png'):
  # nei = only cc with non-empty interior

  print("")
  print("making iou scatterplots ...")
  scale = .75
  size_fac = 50*scale
  
  os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin' # for tex in matplotlib
  plt.rc('font', size=10, family='serif')
  plt.rc('axes', titlesize=10)
  plt.rc('figure', titlesize=10*scale)
  plt.rc('text', usetex=True)
  plt.figure(figsize=(9*scale,13*scale),dpi=300)
  
  plt.subplot(5, 3, 1, aspect='equal')
  add_scatterplot_vs_iou(df_full['iou'], df_full['S'], df_full['E'], "$\\bar E$", size_fac, scale)
  plt.subplot(5, 3, 2, aspect='equal')
  add_scatterplot_vs_iou(df_full['iou'], df_full['S'], df_full['D'], "$\\bar D$", size_fac, scale)
  plt.subplot(5, 3, 3, aspect='equal')
  add_scatterplot_vs_iou(df_full['iou'], 1, df_full['S']/df_full['S'].max(), "$S/S_{max}$", .5, scale)
  plt.subplot(5, 3, 4, aspect='equal')
  add_scatterplot_vs_iou(df_full_nei['iou'], df_full_nei['S'], df_full_nei['E_in'], "$\\bar E_{in}$", size_fac, scale)
  plt.subplot(5, 3, 5, aspect='equal')
  add_scatterplot_vs_iou(df_full_nei['iou'], df_full_nei['S'], df_full_nei['D_in'], "$\\bar D_{in}$", size_fac, scale)
  plt.subplot(5, 3, 6, aspect='equal')
  add_scatterplot_vs_iou(df_full_nei['iou'], 1, df_full_nei['S_in']/df_full_nei['S_in'].max(), "$S_{in}/S_{in,max}$", .5, scale)
  plt.subplot(5, 3, 7, aspect='equal')
  add_scatterplot_vs_iou(df_full['iou'], df_full['S'], df_full['E_bd'], "$\\bar E_{bd}$", size_fac, scale)
  plt.subplot(5, 3, 8, aspect='equal')
  add_scatterplot_vs_iou(df_full['iou'], df_full['S'], df_full['D_bd'], "$\\bar D_{bd}$", size_fac, scale)
  plt.subplot(5, 3, 9, aspect='equal')
  add_scatterplot_vs_iou(df_full['iou'], 1, df_full['S_bd']/df_full['S_bd'].max(), "$S_{bd}/S_{bd,max}$", .5, scale)
  plt.subplot(5, 3, 10, aspect='equal')
  add_scatterplot_vs_iou(df_full['iou'], df_full['S'], df_full['E_rel']/df_full['E_rel'].max(), "$\\tilde{\\bar E}/\\tilde{\\bar E}_{max}$", size_fac, scale)
  plt.subplot(5, 3, 11, aspect='equal')
  add_scatterplot_vs_iou(df_full['iou'], df_full['S'], df_full['D_rel']/df_full['D_rel'].max(), "$\\tilde{\\bar D}/\\tilde{\\bar D}_{max}$", size_fac, scale)
  plt.subplot(5, 3, 12, aspect='equal')
  add_scatterplot_vs_iou(df_full['iou'], 1, df_full['S_rel']/df_full['S_rel'].max(), "$\\tilde{S}/\\tilde{S}_{max}$", .5, scale)
  plt.subplot(5, 3, 13, aspect='equal')
  add_scatterplot_vs_iou(df_full_nei['iou'], df_full_nei['S'], df_full_nei['E_rel_in']/df_full_nei['E_rel_in'].max(), "$\\tilde{\\bar E}_{in}/\\tilde{\\bar E}_{in,max}$", size_fac, scale)
  plt.subplot(5, 3, 14, aspect='equal')
  add_scatterplot_vs_iou(df_full_nei['iou'], df_full_nei['S'], df_full_nei['D_rel_in']/df_full_nei['D_rel_in'].max(), "$\\tilde{\\bar D}_{in}/\\tilde{\\bar D}_{in,max}$", size_fac, scale)
  plt.subplot(5, 3, 15, aspect='equal')
  add_scatterplot_vs_iou(df_full_nei['iou'], 1, df_full_nei['S_rel_in']/df_full_nei['S_rel_in'].max(), "$\\tilde{S}_{in}/\\tilde{S}_{in,max}$", .5, scale)

  plt.tight_layout(pad=1.0*scale, w_pad=0.5*scale, h_pad=1.5*scale)
  save_path = os.path.join(metaseg.get("RESULTS_DIR"), filename)
  plt.savefig(save_path)
  print("scatterplots saved to " + save_path)
  
  

def plot_roc_curve(Y, probs, roc_path):
  # roc curve
  fpr, tpr, _ = roc_curve(Y, probs)
  roc_auc = auc(fpr, tpr)
  print("auc", roc_auc)
  plt.figure()
  lw = 2
  plt.plot(fpr, tpr, color='red',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic curve')
  plt.legend(loc="lower right")
  
  roc_dir  = os.path.dirname( roc_path )
  if not os.path.exists( roc_dir ):
    os.makedirs( roc_dir )
  
  plt.savefig(roc_path)
  print("roc curve saved to " + roc_path)
  
  return roc_auc

    

def name_to_latex( name ):
  
  for i in range(100):
    if name == "cprob"+str(i):
      return "$P_{"+str(i)+"}$"

  mapping = { 'E': '$\\mu \\bar E$',
              'E_bd': '$\\mu{\\bar E}_{bd}$',
              'E_in': '$\\mu{\\bar E}_{in}$',
              'E_rel_in': '$\\mu\\tilde{\\bar E}_{in}$',
              'E_rel': '$\\mu\\tilde{\\bar E}$',
              'EV': '$v\\hat E$',
              'EV_bd': '$v{\\hat E}_{bd}$',
              'EV_in': '$v{\\hat E}_{in}$',
              'EV_rel_in': '$v\\tilde{\\hat E}_{in}$',
              'EV_rel': '$v\\tilde{\\hat E}$',
              'D': '$\\mu\\bar M$',
              'D_bd': '$\\mu{\\bar M}_{bd}$',
              'D_in': '$\\mu{\\bar M}_{in}$',
              'D_rel_in': '$\\mu\\tilde{\\bar M}_{in}$',
              'D_rel': '$\\mu\\tilde{\\bar M}$',
              'DV': '$v\\hat M$',
              'DV_bd': '$v{\\hat M}_{bd}$',
              'DV_in': '$v{\\hat M}_{in}$',
              'DV_rel_in': '$v\\tilde{\\hat M}_{in}$',
              'DV_rel': '$v\\tilde{\\hat M}$',
              'P': '$\\mu\\bar V$',
              'P_bd': '$\\mu{\\bar V}_{bd}$',
              'P_in': '$\\mu{\\bar V}_{in}$',
              'P_rel_in': '$\\mu\\tilde{\\bar V}_{in}$',
              'P_rel': '$\\mu\\tilde{\\bar V}$',
              'PV': '$v\\hat V$',
              'PV_bd': '$v{\\hat V}_{bd}$',
              'PV_in': '$v{\\hat V}_{in}$',
              'PV_rel_in': '$v\\tilde{\\hat V}_{in}$',
              'PV_rel': '$v\\tilde{\\hat V}$',
              'K': '$\\bar K$',
              'K_bd': '${\\bar K}_{bd}$',
              'K_in': '${\\bar K}_{in}$',
              'K_rel_in': '$\\tilde{\\bar K}_{in}$',
              'K_rel': '$\\tilde{\\bar K}$',
              'S': '$S$',
              'S_bd': '${S}_{bd}$',
              'S_in': '${S}_{in}$',
              'S_rel_in': '$\\tilde{S}_{in}$',
              'S_rel': '$\\tilde{S}$',
              'mean_x' : '${\\bar x}$',
              'mean_y' : '${\\bar y}$', }
  if str(name) in mapping:
    return mapping[str(name)]
  else:
    return str(name)



def generate_lasso_plots( stats, mean_stats, X_names, class_names ):

  nc = len(X_names) - len(class_names)  
  coefs = np.squeeze(stats['coefs'][0,:,:])
  classcoefs = np.squeeze(stats['coefs'][0,:,nc:])
  #coefs = np.concatenate( [coefs[:,0:nc], np.max( np.abs(coefs[:,nc:]), axis=1 ).reshape( (coefs.shape[0],1) )], axis=1 )
  coefs = coefs[:,0:nc]
  max_acc = np.argmax( stats['penalized_val_acc'][0], axis=-1 )
  alphas = stats["alphas"]
  
  cmap=plt.get_cmap('tab20')
  figsize=(9.3,5.8)
  
  os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin' # for tex in matplotlib
  plt.rc('font', size=10, family='serif')
  plt.rc('axes', titlesize=10)
  plt.rc('figure', titlesize=10)
  plt.rc('text', usetex=True)
  
  plot_names = X_names[0:nc]+["$C_p$"]
  plt.figure(figsize=figsize)
  plt.clf()
  for i in range(coefs.shape[1]):
    plt.semilogx(alphas, coefs[:,i], label=name_to_latex(plot_names[i]), color=cmap(i/20) )
  ymin, ymax = plt.ylim()  
  plt.vlines(alphas[max_acc], ymin, ymax, linestyle='dashed', linewidth=0.5, color='grey')
  legend = plt.legend(loc='upper right')
  plt.xlabel('$\lambda^{-1}$')
  plt.ylabel('coefficients $c_i$')
  plt.axis('tight')
  plt.savefig(metaseg.get("RESULTS_DIR")+'lasso1.pdf', bbox_inches='tight')
  
  plt.clf()
  for i in range(classcoefs.shape[1]):
    plt.semilogx(alphas, classcoefs[:,i], label="$C_{"+str(i)+"}$", color=cmap(i/20) )
  plt.vlines(alphas[max_acc], ymin, ymax, linestyle='dashed', linewidth=0.5, color='grey')
  legend = plt.legend(loc='upper right')
  plt.xlabel('$\lambda^{-1}$')
  plt.ylabel('coefficients $c_i$')
  plt.axis('tight')
  plt.savefig(metaseg.get("RESULTS_DIR")+'lasso2.pdf', bbox_inches='tight')
    
  plt.clf()
  plt.semilogx(alphas, stats['plain_val_acc'][0]                            , label="unpenalized model", color=cmap(2) )
  plt.semilogx(alphas, stats['penalized_val_acc'][0]                        , label="penalized model", color=cmap(0) )
  plt.semilogx(alphas, mean_stats['entropy_val_acc']*np.ones((len(alphas),)), label="entropy baseline", color='black', linestyle='dashed' )
  ymin, ymax = plt.ylim()
  plt.vlines(alphas[max_acc], ymin, ymax, linestyle='dashed', linewidth=0.5, color='grey')
  legend = plt.legend(loc='lower right')
  plt.xlabel('$\lambda^{-1}$')
  plt.ylabel('classification accuracy')
  plt.axis('tight')
  plt.savefig(metaseg.get("RESULTS_DIR")+'classif_perf.pdf', bbox_inches='tight')
    
  plt.clf()
  plt.semilogx(alphas, stats['plain_val_auroc'][0]                            , label="unpenalized model", color=cmap(2) )
  plt.semilogx(alphas, stats['penalized_val_auroc'][0]                        , label="penalized model", color=cmap(0) )
  plt.semilogx(alphas, mean_stats['entropy_val_auroc']*np.ones((len(alphas),)), label="entropy baseline", color='black', linestyle='dashed' )
  ymin, ymax = plt.ylim()
  plt.vlines(alphas[max_acc], ymin, ymax, linestyle='dashed', linewidth=0.5, color='grey')
  legend = plt.legend(loc='lower right')
  plt.xlabel('$\lambda^{-1}$')
  plt.ylabel('AUROC')
  plt.axis('tight')
  plt.savefig(metaseg.get("RESULTS_DIR")+'classif_auroc.pdf', bbox_inches='tight')



def plot_regression( X2_val, y2_val, y2_pred, ya_val, ypred, X_names ):
    
  os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin' # for tex in matplotlib
  plt.rc('font', size=10, family='serif')
  plt.rc('axes', titlesize=10)
  plt.rc('figure', titlesize=10)
  plt.rc('text', usetex=True)
  
  cmap=plt.get_cmap('tab20')
  
  figsize=(3.0,13.0/5.0)
  plt.figure(figsize=figsize, dpi=300)
  plt.clf()
  S_ind = 0
  for S_ind in range(len(X_names)):
    if X_names[S_ind] == "S":
      break
  
  sizes = np.squeeze(X2_val[:,S_ind]*np.std(X2_val[:,S_ind]))
  sizes = sizes - np.min(sizes)
  sizes = sizes / np.max(sizes) * 50 #+ 1.5      
  x = np.arange(0., 1, .01)
  plt.plot( x, x, color='black' , alpha=0.5, linestyle='dashed')
  plt.scatter( y2_val, np.clip(y2_pred,0,1), s=sizes, linewidth=.5, c=cmap(0), edgecolors=cmap(1), alpha=0.25 )
  plt.xlabel('$\mathit{IoU}_\mathrm{adj}$')
  plt.ylabel('predicted $\mathit{IoU}_\mathrm{adj}$')
  plt.savefig(metaseg.get("RESULTS_DIR")+'regression1.png', bbox_inches='tight')

