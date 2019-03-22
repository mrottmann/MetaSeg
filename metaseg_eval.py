
import random
import os
import time
import sys
from PIL import Image
import numpy as np
import pandas as pd
import scipy
from sklearn import datasets, linear_model, preprocessing, model_selection
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from skimage.transform import resize
from multiprocessing import Pool
import pickle

# compiled functions for metric calculation
from metrics import compute_metrics

# include io functions and initialize "metaseg"
# NOTE: please check "metaseg_io.py", in particular "probs_gt_save"
# for instructions on how to prepare your input data for MetaSeg.
# Furthermore, please adjust the variables and paths in "global_defs.py"
from metaseg_io import probs_gt_save, probs_gt_load, \
                       metrics_dump, metrics_load, \
                       components_dump, components_load, \
                       get_save_path_metrics_i, get_save_path_components_i, \
                       get_iou_seg_vis_path_i, get_save_path_stats, \
                       get_save_path_probs_i, get_img_path_fname, \
                       get_meanvarrat_path_i, get_varvarrat_path_i, \
                       get_meandiff_path_i, get_vardiff_path_i, \
                       get_meanentropy_path_i, get_varentropy_path_i, \
                       get_kldiv_path_i, metaseg
                     
from metaseg_plot import add_scatterplot_vs_iou, make_scatterplots, \
                         plot_roc_curve, name_to_latex, generate_lasso_plots, \
                         plot_regression

# NOTE: 
# "cs_labels" is included for the segmentations color code, this is only required for visualization.
# Replace this if necessary and modify the lines in "visualize_metrics_i()" that contain "cs_labels"
# accordingly.
sys.path.append(metaseg.get("DEEPLAB_PARENT_DIR"))
from deeplab import cs_labels

from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

import keras
import keras.backend as K

np.random.seed( 0 )



def main():
  
  metaseg.set_from_argv( sys.argv )
  
  metaseg.print_attr()
  
  if metaseg.get("COMPUTE_IOU"):
    compute_IoU()
  
  if metaseg.get("COMPUTE_METRICS"):
    compute_metrics_per_image()
  
  if metaseg.get("VISUALIZE_METRICS"):
    visualize_metrics()
  
  if metaseg.get("ANALYZE_METRICS"):
    analyze_metrics()
    

def classif_eval( y, y_pred ):  
  return np.mean( np.argmax(y_pred,axis=-1)==y )



def fit_eval( X, y, J, fit_func, eval_func, run ):
  
  np.random.seed( run )
    
  val_mask = np.random.rand(len(y)) < 3.0/6.0
  
  X_val = X[val_mask]
  y_val = y[val_mask]
      
  X_train = X[np.logical_not(val_mask)]
  y_train = y[np.logical_not(val_mask)]
  
  y_val_pred, y_pred = fit_func( X_train[:,J], y_train, X_val[:,J])
  
  score_train = eval_func( y_train, y_pred )
  score_val = eval_func( y_val, y_val_pred )
  
  return [score_train, score_val]
  


def set_append( X, y, I, fit_func, eval_func, n_av ):
  
  notI = [i for i in range(X.shape[-1]) if i not in I]
  
  max_score_val = 0
  max_score = 0
  max_ind = 0
  
  p = Pool(n_av)
  for i in notI:
    print("testing metric",i)
    
    J = I.copy()
    J.append(i)
    
    p_args = [ ( X, y, J, fit_func, eval_func, run ) for run in range(n_av) ]
    run_scores_i = p.starmap( fit_eval, p_args )
    
    mean_train_score = mean_val_score = 0
    std_train_score = std_val_score = 0
    if n_av > 1:
      mean_train_score, mean_val_score = np.mean( run_scores_i, axis=0 )
      std_train_score , std_val_score  = np.std( run_scores_i, axis=0 )
    else:
      #print( run_scores_i )
      mean_train_score, mean_val_score = run_scores_i[0]

    if mean_train_score > max_score:
      max_score = mean_train_score
      max_score_val = mean_val_score
      std_max_score_val = std_val_score
      max_ind = i
      
  return max_ind, max_score_val, std_max_score_val



def regression_fit_and_predict( X_train, y_train, X_test ):
  
  if metaseg.get("USE_NN"):
    
    reg = regularizers.l2(0.005)
    
    model = Sequential()
    model.add(Dense(units=61, activation='relu', kernel_regularizer=reg, input_dim=X_train.shape[-1]))
    model.add(Dense(units=61, activation='relu', kernel_regularizer=reg ))
    model.add(Dense(units=1, kernel_regularizer=reg, activation='linear'))
    
    def stddev(y_true, y_pred):
      return K.sqrt( keras.losses.mean_squared_error(y_true, y_pred) )
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[stddev])
    
    model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=0)
    
    y_test_pred = np.clip( model.predict(X_test), 0, 1 )
    y_train_pred = np.clip( model.predict(X_train), 0, 1 )
    
  else:
    model = linear_model.LinearRegression()
    model.fit(X_train,y_train)
    y_test_pred = np.clip( model.predict(X_test), 0, 1 )
    y_train_pred = np.clip( model.predict(X_train), 0, 1 )

  return y_test_pred, y_train_pred



def classification_l1_fit_and_predict( X_train, y_train, alphas, X_test ):

  model = linear_model.LogisticRegression(C=alphas, penalty='l1', solver='saga', max_iter=1000, tol=1e-3 )
  model.fit( X_train, y_train )
  y_test_pred = model.predict_proba(X_test)
  y_train_pred = model.predict_proba(X_train)
  
  return y_test_pred, y_train_pred, np.asarray(model.coef_[0])



def classification_fit_and_predict( X_train, y_train, X_test ):

  if metaseg.get("USE_NN"):
    
    Y_train = keras.utils.to_categorical(y_train, num_classes=2)
    
    reg = regularizers.l2(0.005)
    
    model = Sequential()
    model.add(Dense(units=61, activation='relu', kernel_regularizer=reg, input_dim=X_train.shape[-1]))
    model.add(Dense(units=61, activation='relu', kernel_regularizer=reg ))
    model.add(Dense(units=2, kernel_regularizer=reg, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(X_train, Y_train, epochs=50, batch_size=128, verbose=0)

    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
  else:
    
    model = linear_model.LogisticRegression(penalty=None, solver='saga', max_iter=1000, tol=1e-3 )
    model.fit( X_train, y_train )
    y_test_pred = model.predict_proba(X_test)
    y_train_pred = model.predict_proba(X_train)
  
  return y_test_pred, y_train_pred



def label_as_onehot(label, num_classes, shift_range=0):
  
  y = np.zeros((num_classes, label.shape[0], label.shape[1]))
  for c in range(shift_range,num_classes+shift_range):
          y[c-shift_range][label==c] = 1
  y = np.transpose(y,(1,2,0)) # shape is (height, width, num_classes)
  return y.astype('uint8')



def classes_to_categorical( classes, nc = None ):

  classes = np.squeeze( np.asarray(classes) )
  if nc == None:
    nc      = np.max(classes)
  classes = label_as_onehot( classes.reshape( (classes.shape[0],1) ), nc ).reshape( (classes.shape[0], nc) )
  names   = [ "C_"+str(i) for i in range(nc) ]
  
  return classes, names



def visualize_segments( comp, metric ):
  
  R = np.asarray( metric )
  R = 1-0.5*R
  G = np.asarray( metric )
  B = 0.3+0.35*np.asarray( metric )
  
  R = np.concatenate( (R, np.asarray([0,1])) )
  G = np.concatenate( (G, np.asarray([0,1])) )
  B = np.concatenate( (B, np.asarray([0,1])) )
  
  components = np.asarray(comp.copy(), dtype='int16')
  components[components  < 0] = len(R)-1
  components[components == 0] = len(R)
  
  img = np.zeros( components.shape+(3,) )
  
  for x in range(img.shape[0]):
    for y in range(img.shape[1]):
      img[x,y,0] = R[components[x,y]-1]
      img[x,y,1] = G[components[x,y]-1]
      img[x,y,2] = B[components[x,y]-1]
  
  img = np.asarray( 255*img ).astype('uint8')
  
  return img



def visualize_segments_diff( comp, metric ):
  
  R = np.clip( 0.5 +  0.5*np.asarray( metric ), 0, 1 )
  G = np.clip( 0.5 -  0.5*np.asarray( metric ), 0, 1 )
  B = np.clip( 0.5 - 0.25*np.asarray( metric ), 0, 1 )
  
  R = np.concatenate( (R, np.asarray([0,1])) )
  G = np.concatenate( (G, np.asarray([0,1])) )
  B = np.concatenate( (B, np.asarray([0,1])) )
  
  components = np.asarray(comp.copy(), dtype='int16')
  components[components  < 0] = len(R)-1
  components[components == 0] = len(R)
  
  img = np.zeros( components.shape+(3,) )
  
  for x in range(img.shape[0]):
    for y in range(img.shape[1]):
      img[x,y,0] = R[components[x,y]-1]
      img[x,y,1] = G[components[x,y]-1]
      img[x,y,2] = B[components[x,y]-1]
  
  img = np.asarray( 255*img ).astype('uint8')
  
  return img



def metrics_to_nparray( metrics, names, normalize=False, non_empty=False, all_metrics=[] ):
  
  I = range(len(metrics['S_in']))
  if non_empty == True:
    I = np.asarray(metrics['S_in']) > 0
  M = np.asarray( [ np.asarray(metrics[ m ])[I] for m in names ] )
  MM = []
  if all_metrics == []:
    MM = M.copy()
  else:
    MM = np.asarray( [ np.asarray(all_metrics[ m ])[I] for m in names ] )
  
  if normalize == True:
    for i in range(M.shape[0]):
      if names[i] != "class":
        M[i] = ( np.asarray(M[i]) - np.mean(MM[i], axis=-1 ) ) / ( np.std(MM[i], axis=-1 ) + 1e-10 )
  
  M = np.squeeze(M.T)
    
  return M



def compute_metrics_i( i ):
  
  if os.path.isfile( get_save_path_probs_i(i) ):
    
    start = time.time()
    
    probs, gt, _ = probs_gt_load( i )
    metrics, components = compute_metrics( probs, gt )
    
    metrics_dump( metrics, i )
    components_dump( components, i )
    
    print("image", i, "processed in {}s\r".format( round(time.time()-start) ) )



def compute_IoU_i( seg, gt, nclasses ):
  
  allI = np.zeros(nclasses)
  allU = np.zeros(nclasses)
  seg = seg.flatten()
  gt  = gt.flatten()
  
  seg[ gt == 255 ] = 255
  
  for i in range(nclasses):
    TP = np.logical_and( seg==i, gt==i )
    FP = np.logical_and( seg==i, gt!=i )
    FN = np.logical_and( seg!=i, gt==i )
    allI[i] = np.sum(TP)
    allU[i] = np.sum(TP) + np.sum(FP) + np.sum(FN)
    
  return allI, allU



def compute_IoU():
  
  print("computing IoUs and mIoU...")
    
  probs, gt, _ = probs_gt_load( 0 )
  nclasses = probs.shape[-1]
  allI, allU = np.zeros(nclasses), np.zeros(nclasses)
  
  for i in range(0,metaseg.get("NUM_IMAGES")):
    
    if i%50 == 0:
      print(str(i)+"/"+str(metaseg.get("NUM_IMAGES"))) 
    
    probs, gt, _ = probs_gt_load( i )
    seg = np.argmax( probs, axis=-1 )
    #seg = seg[256:768,512:1536]
    #gt = gt[256:768,512:1536]
    allI_i, allU_i = compute_IoU_i( seg, gt, nclasses )
    allI += allI_i
    allU += allU_i
    
  IoUs = allI.astype("float32") / allU.astype("float32")
  
  for i in range(nclasses):
    print( "{:20s}".format([label.name for label in cs_labels.labels if label.trainId == i][0]), \
      ":", "{:.2f}%".format(100*IoUs[i]) )
  print( "{:20s}".format("mIoU"), ":", "{:.2f}%".format(100*np.mean(IoUs)))



def compute_metrics_i( i ):
  
  if os.path.isfile( get_save_path_probs_i(i) ):
    
    start = time.time()
    
    hmaps = dict( { "P" : get_meanvarrat_path_i(i),
                    "PV": get_varvarrat_path_i(i),
                    "D" : get_meandiff_path_i(i),
                    "DV": get_vardiff_path_i(i),
                    "E" : get_meanentropy_path_i(i),
                    "EV": get_varentropy_path_i(i),
                    "K" : get_kldiv_path_i(i),
                  } )
    
    probs, gt, _ = probs_gt_load( i )
    metrics, components = compute_metrics( probs, gt, hmaps )
    
    metrics_dump( metrics, i )
    components_dump( components, i )
    
    print("image", i, "processed in {}s\r".format( round(time.time()-start) ) )



def visualize_metrics_i( iou, iou_pred, i ):
  
  if os.path.isfile( get_save_path_probs_i(i) ):
    
    probs, gt, filename = probs_gt_load( i )
    
    path = get_img_path_fname( filename )
    
    input_image = np.asarray(Image.open( path ))
    components  = components_load( i )
    
    pred = np.asarray( np.argmax( probs, axis=-1 ), dtype='int' )
    
    gt[ gt == 255 ] = 0
    predc = np.asarray([ cs_labels.trainId2label[ pred[p,q] ].color for p in range(pred.shape[0]) for q in range(pred.shape[1]) ])
    gtc   = np.asarray([ cs_labels.trainId2label[ gt[p,q]   ].color for p in range(gt.shape[0]) for q in range(gt.shape[1]) ])
    predc = predc.reshape(input_image.shape)
    gtc   = gtc.reshape(input_image.shape)
    
    img_iou = 0.66 * visualize_segments( components, iou ) + 0.34 * input_image
    
    I4 = predc / 2.0 + input_image / 2.0 
    I3 = gtc / 2.0 + input_image / 2.0
    
    img_iou_pred = 0.66 * visualize_segments( components, iou_pred ) + 0.34 * input_image

    img = np.concatenate( (img_iou,img_iou_pred), axis=1 )
    img2 = np.concatenate( (I3,I4), axis=1 )
    img = np.concatenate( (img,img2), axis=0 )
    image = Image.fromarray(img.astype('uint8'), 'RGB')
    
    seg_dir = metaseg.get("IOU_SEG_VIS_DIR")
    if not os.path.exists( seg_dir ):
      os.makedirs( seg_dir )
    image.save(seg_dir+"img"+str(i)+".png")
    
    print("stored:",seg_dir+"img"+str(i)+".png")



def visualize_metrics( ):
  
  num_cores = metaseg.get("NUM_CORES")
  
  print("visualization running")
  
  metrics = metrics_load( 0 )
  
  start = list([ 0, len(metrics["S"]) ])
  
  for i in range(1,metaseg.get("NUM_IMAGES")):    
    m = metrics_load( i )
    start += [ start[-1]+len(m["S"]) ]
    
    for j in metrics:
      metrics[j] += m[j]
  
  nclasses = np.max(metrics["class"])+1
  
  Xa, classes, ya, _, X_names, class_names = metrics_to_dataset( metrics, nclasses, non_empty=False )
  Xa = np.concatenate( (Xa,classes), axis=-1 )
  X_names += class_names
  
  ya_pred, _ = regression_fit_and_predict( Xa, ya, Xa )
  
  print("model r2 score:", r2_score(ya,ya_pred) )
  print(" ")
  
  p = Pool(num_cores)
  
  p_args = [ (ya[start[i]:start[i+1]], ya_pred[start[i]:start[i+1]], i) for i in range(metaseg.get("NUM_IMAGES")) ]
  
  p.starmap( visualize_metrics_i, p_args )



def concatenate_metrics( save=False ):
  
  metrics = metrics_load( 0 )
  
  for i in range(1,metaseg.get("NUM_IMAGES")):
    sys.stdout.write("\t concatenated file number {} / {}\r".format(i+1,metaseg.get("NUM_IMAGES")))
    
    m = metrics_load( i )
    
    for j in metrics:
      metrics[j] += m[j]
  
  print(" ")
  print("connected components:", len(metrics['iou']) )
  print("non-empty connected components:", np.sum( np.asarray(metrics['S_in']) != 0) )
  
  if ( save == True ):
    metrics_dump( metrics, "_all" )
  
  return metrics



def compute_metrics_per_image( ):
  
  num_cores = metaseg.get("NUM_CORES")
  
  print("calculating statistics")
      
  p = Pool(num_cores)
  
  p_args = [ (k,) for k in range(metaseg.get("NUM_IMAGES")) ]
  
  p.starmap( compute_metrics_i, p_args )
  
  concatenate_metrics( save=True )



def adjusted_r2(r2, num_dof, num_samples):
  
  return 1 - (1-r2) * (num_samples - 1) / (num_samples - num_dof - 1)



def fit_model_run( Xa, ya, y0a, alphas, X_names, stats, run ):
  
  print("run",run)
  if run == 0:
    for i in range(len(X_names)):
      print(str(i)+":",X_names[i])
  
  np.random.seed( run )
    
  val_mask = np.random.rand(len(ya)) < 3.0/6.0
  
  Xa_val = Xa[val_mask]
  ya_val = ya[val_mask]
  y0a_val = y0a[val_mask]
      
  Xa_train = Xa[np.logical_not(val_mask)]
  ya_train = ya[np.logical_not(val_mask)]
  y0a_train = y0a[np.logical_not(val_mask)]
  
  coefs = np.zeros((len(alphas),Xa.shape[1]))
  
  max_acc = 0
  best_l1_results = []
  
  for i in range(len(alphas)):
    
    y0a_val_pred, y0a_train_pred, lm_coefs = classification_l1_fit_and_predict( Xa_train, y0a_train, alphas[i], Xa_val )
    
    stats['penalized_val_acc'][run,i] = np.mean( np.argmax(y0a_val_pred,axis=-1)==y0a_val )
    stats['penalized_train_acc'][run,i] = np.mean( np.argmax(y0a_train_pred,axis=-1)==y0a_train )
    
    if stats['penalized_val_acc'][run,i] > max_acc:
      max_acc = stats['penalized_val_acc'][run,i]
      best_l1_results = [y0a_val_pred.copy(), y0a_train_pred.copy()]
    
    print("step"+str(i)+", alpha={:.2E}".format(alphas[i])+", val. acc.: {:.2f}%".format(100*stats['penalized_val_acc'][run,i]), end=", ")
    print("coefs non-zero:", end=" ")
    
    fpr, tpr, _ = roc_curve(y0a_val, y0a_val_pred[:,1])
    stats['penalized_val_auroc'][run,i] = auc(fpr, tpr)
    fpr, tpr, _ = roc_curve(y0a_train, y0a_train_pred[:,1])
    stats['penalized_train_auroc'][run,i] = auc(fpr, tpr)
    
    coefs[i] = lm_coefs
    print([ j for j in range(len(coefs[i])) if np.abs(coefs[i,j]) > 1e-6 ])
    
    if np.sum( np.abs(coefs[i]) > 1e-6 ) > 0 :
      
      y0a_val_pred, y0a_train_pred = classification_fit_and_predict( Xa_train[:,np.abs(coefs[i]) > 1e-6], y0a_train, Xa_val[:,np.abs(coefs[i]) > 1e-6] )
      
      stats['plain_val_acc'][run,i] = np.mean( np.argmax(y0a_val_pred,axis=-1)==y0a_val )
      stats['plain_train_acc'][run,i] = np.mean( np.argmax(y0a_train_pred,axis=-1)==y0a_train )
      fpr, tpr, _ = roc_curve(y0a_val, y0a_val_pred[:,1])
      stats['plain_val_auroc'][run,i] = auc(fpr, tpr)
      fpr, tpr, _ = roc_curve(y0a_train, y0a_train_pred[:,1])
      stats['plain_train_auroc'][run,i] = auc(fpr, tpr)
    else:
      stats['plain_val_acc'][run,i] = stats['penalized_val_acc'][run,i]
      stats['plain_train_acc'][run,i] = stats['penalized_train_acc'][run,i]
      
      stats['plain_val_auroc'][run,i] = stats['penalized_val_auroc'][run,i]
      stats['plain_train_auroc'][run,i] = stats['penalized_train_auroc'][run,i]
      
  
  max_acc = np.argmax(stats['penalized_val_acc'][run])
  
  ypred = np.argmax(best_l1_results[0],axis=-1)
  ypred_t = np.argmax(best_l1_results[1],axis=-1)
  
  E_ind = 0
  for E_ind in range(len(X_names)):
    if X_names[E_ind] == "E":
      break
  
  y0a_val_pred, y0a_train_pred = classification_fit_and_predict( Xa_train[:,E_ind].reshape((Xa_train.shape[0],1)), 
                                 y0a_train, Xa_val[:,E_ind].reshape((Xa_val.shape[0],1)) )
  
  stats['entropy_val_acc'][run] = np.mean( np.argmax(y0a_val_pred,axis=-1)==y0a_val )
  stats['entropy_train_acc'][run] = np.mean( np.argmax(y0a_val_pred,axis=-1)==y0a_val )
  fpr, tpr, _ = roc_curve(y0a_val, y0a_val_pred[:,1])
  stats['entropy_val_auroc'][run] = auc(fpr, tpr)
  fpr, tpr, _ = roc_curve(y0a_train, y0a_train_pred[:,1])
  stats['entropy_train_auroc'][run] = auc(fpr, tpr)
  
  if run == 0:
    plot_roc_curve(y0a_val, best_l1_results[0][:,1], metaseg.get("RESULTS_DIR")+'roccurve.pdf')
  
  stats['iou0_found'][run] = np.sum( np.logical_and(ypred == 1, y0a_val == 1) ) + np.sum( np.logical_and(ypred_t == 1, y0a_train == 1) )
  stats['iou0_not_found'][run] = np.sum( np.logical_and(ypred == 0, y0a_val == 1) ) + np.sum( np.logical_and(ypred_t == 0, y0a_train == 1) )
  stats['not_iou0_found'][run] = np.sum( np.logical_and(ypred == 0, y0a_val == 0) ) + np.sum( np.logical_and(ypred_t == 0, y0a_train == 0) )
  stats['not_iou0_not_found'][run] = np.sum( np.logical_and(ypred == 1, y0a_val == 0) ) + np.sum( np.logical_and(ypred_t == 1, y0a_train == 0) )
  
  
  X2_train = Xa_val.copy()
  y2_train = ya_val.copy()
  X2_val   = Xa_train.copy()
  y2_val   = ya_train.copy()
  
  
  #I = [6, 7, 15, 17, 19, 38, 39, 41, 46, 47]
  I = np.arange(len(X_names))
  y2_val_pred, y2_train_pred = regression_fit_and_predict( X2_train[:,I], y2_train, X2_val[:,I] )
  
  stats['regr_val_mse'][run] = np.sqrt( mean_squared_error(y2_val, y2_val_pred) )
  stats['regr_val_r2'][run]  = r2_score(y2_val, y2_val_pred)
  stats['regr_train_mse'][run] = np.sqrt( mean_squared_error(y2_train, y2_train_pred) )
  stats['regr_train_r2'][run]  = r2_score(y2_train, y2_train_pred)
  
  lmer = linear_model.LinearRegression()
  
  
  y2e_val_pred, y2e_train_pred = regression_fit_and_predict( X2_train[:,E_ind].reshape((X2_train.shape[0],1)),
                                 y2_train, X2_val[:,E_ind].reshape((X2_val.shape[0],1)) )

  stats['entropy_regr_val_mse'][run] = np.sqrt( mean_squared_error(y2_val, y2e_val_pred) )
  stats['entropy_regr_val_r2'][run]  = r2_score(y2_val, y2e_val_pred)
  stats['entropy_regr_train_mse'][run] = np.sqrt( mean_squared_error(y2_train, y2e_train_pred) )
  stats['entropy_regr_train_r2'][run]  = r2_score(y2_train, y2e_train_pred)
  
  stats['coefs'][run] = np.asarray(coefs)
  
  if run == 0:
    
    plot_regression( X2_val, y2_val, y2_val_pred, ya_val, ypred, X_names )
  
  return stats



def greedy_varselect( Xa, ya, y0a, X_names ):

  n_av      = metaseg.get("NUM_LASSO_AVERAGES")
  
  print("averaging over",n_av,"runs")
  
  I1 = list([])
  scores1 = list([])
  std_scores1 = list([])
  
  for i in range(metaseg.get("NUM_VARSELECT")):
    print(i)
    
    ind, score, std_score = set_append( Xa, ya, I1, regression_fit_and_predict, r2_score, n_av )
    I1.append(ind)
    scores1.append(score)
    std_scores1.append(std_score)
  
  print("regression:")
  for i in I1:
    print(name_to_latex(X_names[i]))
  print(scores1)
  
  I2 = list([])
  scores2 = list([])
  std_scores2 = list([])
  
  for i in range(metaseg.get("NUM_VARSELECT")):
    print(i)
    ind, score, std_score = set_append( Xa, y0a, I2, classification_fit_and_predict, classif_eval, n_av )
    I2.append(ind)
    scores2.append(score)
    std_scores2.append(std_score)
  
  print("classification:")
  for i in I2:
    print(name_to_latex(X_names[i]))
  print(scores2)
  
  cmap=plt.get_cmap('tab20')
  figsize=(4,2.5)

  os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin' # for tex in matplotlib
  plt.rc('font', size=10, family='serif')
  plt.rc('axes', titlesize=10)
  plt.rc('figure', titlesize=10)
  plt.rc('text', usetex=True)
  
  plt.figure(figsize=figsize)
  plt.clf()
  
  plt.plot(np.arange(1,metaseg.get("NUM_VARSELECT")+1), scores1, label="regression $R^2$", color=cmap(0/20) )
  
  plt.fill_between(np.arange(1,metaseg.get("NUM_VARSELECT")+1), np.asarray(scores1)-np.asarray(std_scores1), np.asarray(scores1)+np.asarray(std_scores1), alpha=0.33, facecolor=cmap(0/20) )
  
  plt.plot(np.arange(1,metaseg.get("NUM_VARSELECT")+1), scores2, label="classification ACC", color=cmap(2/20) )
  
  plt.fill_between(np.arange(1,metaseg.get("NUM_VARSELECT")+1), np.asarray(scores2)-np.asarray(std_scores2), np.asarray(scores2)+np.asarray(std_scores2), alpha=0.33, facecolor=cmap(2/20) )
  
  ymin, ymax = plt.ylim()  
  legend = plt.legend(loc='lower right')
  plt.xlabel('number of metrics')
  plt.ylabel('')
  plt.axis('tight')
  plt.savefig(metaseg.get("RESULTS_DIR")+'var_select.pdf', bbox_inches='tight')



def compute_correlations( metrics ):
  
  pd.options.display.float_format = '{:,.5f}'.format

  df_full = pd.DataFrame( data=metrics )
  df_full = df_full.copy().drop(["class","iou0"], axis=1)
  features = df_full.copy().drop(["iou"], axis=1).columns    
  df_all  = df_full.copy()
  df_full = df_full.copy().loc[df_full['S_in'].nonzero()[0]]
  make_scatterplots("../", df_full, df_full )
  iou_corrs   = df_full.corr()["iou"]
  print("\n\ncorrelations with iou (only non_empty in)")
  
  for i in range(iou_corrs.shape[0]):
    print( '{0: <10} : '.format(iou_corrs.index[i]) + '{0:+.05f}'.format(iou_corrs[i]))
  
  y0a = metrics_to_nparray( metrics, ["iou0"] , normalize=False, non_empty=True )
  print(" ")
  print("IoU=0:", np.sum(y0a==1), "of", y0a.shape[0] )
  print("IoU>0:", np.sum(y0a==0), "of", y0a.shape[0] )
  
  return iou_corrs



def get_alphas( n_steps, min_pow, max_pow ):
  
  m = interp1d([0,n_steps-1],[min_pow,max_pow])
  alphas = [10 ** m(i).item() for i in range(n_steps)]
  
  return alphas



def init_stats( n_av, alphas, X_names ):

  n_steps   = len(alphas)
  n_metrics = len(X_names) 
  stats     = dict({})
  
  per_alphas_av_stats = ['penalized_val_acc','penalized_val_auroc','penalized_train_acc','penalized_train_auroc', \
                          'plain_val_acc','plain_val_auroc','plain_train_acc','plain_train_auroc', 'coefs' ]
  per_av_stats        = ['entropy_val_acc','entropy_val_auroc','entropy_train_acc','entropy_train_auroc', \
                          'regr_val_mse', 'regr_val_r2', 'regr_train_mse', 'regr_train_r2', \
                          'entropy_regr_val_mse', 'entropy_regr_val_r2', 'entropy_regr_train_mse', 'entropy_regr_train_r2', \
                          'iou0_found', 'iou0_not_found', 'not_iou0_found', 'not_iou0_not_found' ]
                    
  for s in per_alphas_av_stats:
    stats[s] = 0.5*np.ones((n_av,n_steps))
    
  for s in per_av_stats:
    stats[s] = np.zeros((n_av,))
    
  stats["coefs"]        = np.zeros((n_av,n_steps,n_metrics))
  stats["alphas"]       = alphas
  stats["n_av"]         = n_av
  stats["n_metrics"]    = n_metrics
  stats["metric_names"] = X_names
  
  return stats



def merge_stats( stats, single_run_stats, n_av ):
  
  for run in range(n_av):
    for s in stats:
      if s not in ["alphas", "n_av", "n_metrics", "metric_names"]:
        stats[s][run] = single_run_stats[run][s][run]
  
  return stats



def dump_stats( stats, metrics ):
  
  iou_corrs = compute_correlations( metrics )
  y0a = metrics_to_nparray( metrics, ["iou0"] , normalize=False, non_empty=True )

  mean_stats = dict({})
  std_stats = dict({})
  
  for s in stats:
    if s not in ["alphas", "n_av", "n_metrics", "metric_names"]:
      mean_stats[s] = np.mean(stats[s], axis=0)
      std_stats[s]  = np.std( stats[s], axis=0)
  
  best_pen_ind = np.argmax(mean_stats['penalized_val_acc'])
  best_plain_ind = np.argmax(mean_stats['plain_val_acc'])
  
  # dump stats latex ready
  with open(metaseg.get("RESULTS_DIR")+'av_results.txt', 'wt') as f:
    
    for i in range(iou_corrs.shape[0]):
      print( '{0: <10} : '.format(iou_corrs.index[i]) + '{0:+.05f}'.format(iou_corrs[i]), file=f )
    print(" ", file=f )
    
    print("classification", file=f )
    print( "                             & train                &  val                 &    \\\\ ", file= f)
    M = sorted([ s for s in mean_stats if 'penalized' in s and 'acc' in s ])
    print( "ACC penalized               ", end=" & ", file= f )
    for s in M: print( "${:.2f}\%".format(100*mean_stats[s][best_pen_ind])+"(\pm{:.2f}\%)$".format(100*std_stats[s][best_pen_ind]), end=" & ", file=f )
    print("   \\\\ ", file=f )
    M = sorted([ s for s in mean_stats if 'plain' in s and 'acc' in s ])
    print( "ACC unpenalized             ", end=" & ", file= f )
    for s in M: print( "${:.2f}\%".format(100*mean_stats[s][best_pen_ind])+"(\pm{:.2f}\%)$".format(100*std_stats[s][best_pen_ind]), end=" & ", file=f )
    print("   \\\\ ", file=f )
    M = sorted([ s for s in mean_stats if 'entropy' in s and 'acc' in s ])
    print( "ACC entropy baseline        ", end=" & ", file= f )
    for s in M: print( "${:.2f}\%".format(100*mean_stats[s])+"(\pm{:.2f}\%)$".format(100*std_stats[s]), end=" & ", file=f )
    print("   \\\\ ", file=f )
    
    M = sorted([ s for s in mean_stats if 'penalized' in s and 'auroc' in s ])
    print( "AUROC penalized             ", end=" & ", file= f )
    for s in M: print( "${:.2f}\%".format(100*mean_stats[s][best_pen_ind])+"(\pm{:.2f}\%)$".format(100*std_stats[s][best_pen_ind]), end=" & ", file=f )
    print("   \\\\ ", file=f )
    M = sorted([ s for s in mean_stats if 'plain' in s and 'auroc' in s ])
    print( "AUROC unpenalized           ", end=" & ", file= f )
    for s in M: print( "${:.2f}\%".format(100*mean_stats[s][best_pen_ind])+"(\pm{:.2f}\%)$".format(100*std_stats[s][best_pen_ind]), end=" & ", file=f )
    print("   \\\\ ", file=f )
    M = sorted([ s for s in mean_stats if 'entropy' in s and 'auroc' in s ])
    print( "AUROC entropy baseline      ", end=" & ", file= f )
    for s in M: print( "${:.2f}\%".format(100*mean_stats[s])+"(\pm{:.2f}\%)$".format(100*std_stats[s]), end=" & ", file=f )
    print("   \\\\ ", file=f )
    
    print(" ", file=f)
    print("regression", file=f)
    
    M = sorted([ s for s in mean_stats if 'regr' in s and 'mse' in s and 'entropy' not in s ])
    print( "$\sigma$, all metrics       ", end=" & ", file= f )
    for s in M: print( "${:.3f}".format(mean_stats[s])+"(\pm{:.3f})$".format(std_stats[s]), end="    & ", file=f )
    print("   \\\\ ", file=f )
    M = sorted([ s for s in mean_stats if 'regr' in s and 'mse' in s and 'entropy' in s ])
    print( "$\sigma$, entropy baseline  ", end=" & ", file= f )
    for s in M: print( "${:.3f}".format(mean_stats[s])+"(\pm{:.3f})$".format(std_stats[s]), end="    & ", file=f )
    print("   \\\\ ", file=f )
    
    M = sorted([ s for s in mean_stats if 'regr' in s and 'r2' in s and 'entropy' not in s ])
    print( "$R^2$, all metrics          ", end=" & ", file= f )
    for s in M: print( "${:.2f}\%".format(100*mean_stats[s])+"(\pm{:.2f}\%)$".format(100*std_stats[s]), end=" & ", file=f )
    print("   \\\\ ", file=f )
    M = sorted([ s for s in mean_stats if 'regr' in s and 'r2' in s and 'entropy' in s ])
    print( "$R^2$, entropy baseline     ", end=" & ", file= f )
    for s in M: print( "${:.2f}\%".format(100*mean_stats[s])+"(\pm{:.2f}\%)$".format(100*std_stats[s]), end=" & ", file=f )
    print("   \\\\ ", file=f )
    
    print(" ", file=f )
    M = sorted([ s for s in mean_stats if 'iou' in s ])          
    for s in M: print( s, ": {:.0f}".format(mean_stats[s])+"($\pm${:.0f})".format(std_stats[s]), file=f )
    print("IoU=0:", np.sum(y0a==1), "of", y0a.shape[0], "non-empty components", file=f )
    print("IoU>0:", np.sum(y0a==0), "of", y0a.shape[0], "non-empty components", file=f )
    print("total number of components: ", len(metrics['S']), file=f )
    print(" ", file=f )
    
    dump_path = get_save_path_stats()
    dump_dir  = os.path.dirname( dump_path )

    if not os.path.exists( dump_dir ):
      os.makedirs( dump_dir )

    pickle.dump( stats, open( dump_path, "wb" ) )
    

  return mean_stats, std_stats



def metrics_to_dataset( metrics, nclasses, non_empty=True, all_metrics=[] ):
  
  class_names = []
  
  if ( metaseg.get("USE_KLDIV") == True ):
    X_names = sorted([ m for m in metrics if m not in ["class","iou","iou0"] and "cprob" not in m ])
    #X_names = sorted([ m for m in metrics if m not in ["class","iou","iou0"]])
  else:
    X_names = sorted([ m for m in metrics if m not in ["class","iou","iou0"] and "cprob" not in m and "K" not in m ])
  
  # filters:
  #X_names = [ m for m in X_names if "P" not in m and "K" not in m and "E" not in m and "D" not in m and "S" not in m]
  #X_names = [ m for m in X_names if "P" not in m and "K" not in m and "mean" not in m ]
  #X_names = [ m for m in X_names if "V" not in m and "K" not in m and "mean" not in m and "P" not in m]
  #X_names = [ m for m in X_names if "V" not in m ]
  #X_names = [ m for m in X_names if "M" in m ]
  #X_names = [ m for m in X_names if "E" in m and "V" not in m ]
  #X_names = [ m for m in X_names if "P" in m and "V" not in m ]
  #X_names = [ m for m in X_names if "M" in m and "V" not in m ]
  
  print(X_names)
  
  if metaseg.get("CLASS_DTYPE") == metaseg.get("CLASS_DTYPES")[1]:
    class_names = [ "cprob"+str(i) for i in range(nclasses) if "cprob"+str(i) in metrics ]
  elif metaseg.get("CLASS_DTYPE") == metaseg.get("CLASS_DTYPES")[0]:
    class_names = ["class"]
  else:
    class_names = []
  
  Xa      = metrics_to_nparray( metrics, X_names    , normalize=True , non_empty=non_empty, all_metrics=all_metrics )
  classes = metrics_to_nparray( metrics, class_names, normalize=True , non_empty=non_empty, all_metrics=all_metrics )
  ya      = metrics_to_nparray( metrics, ["iou" ]   , normalize=False, non_empty=non_empty )
  y0a     = metrics_to_nparray( metrics, ["iou0"]   , normalize=False, non_empty=non_empty )
  
  if metaseg.get("CLASS_DTYPE") == metaseg.get("CLASS_DTYPES")[0]:
    classes, class_names = classes_to_categorical( classes, nclasses )
  
  return Xa, classes, ya, y0a, X_names, class_names



def analyze_metrics():

  n_av      = metaseg.get("NUM_LASSO_AVERAGES")
  n_steps   = metaseg.get("NUM_LASSO_LAMBDAS")
  num_cores = min(n_av,metaseg.get("NUM_CORES"))

  metrics   = concatenate_metrics( save=False )
  nclasses  = np.max( metrics["class"] ) + 1
  
  Xa, classes, ya, y0a, X_names, class_names = metrics_to_dataset( metrics, nclasses )
  
  if metaseg.get("CLASS_DTYPE") != metaseg.get("CLASS_DTYPES")[2]:
    Xa = np.concatenate( (Xa,classes), axis=-1 )
    X_names += class_names
  
  alphas           = get_alphas( n_steps, min_pow = -4.2, max_pow = 0.8 )
  stats            = init_stats( n_av, alphas, X_names )
  single_run_stats = init_stats( n_av, alphas, X_names )
  
  p = Pool(num_cores)
  p_args = [ ( Xa, ya, y0a, alphas, X_names, single_run_stats, run ) for run in range(n_av) ]
  single_run_stats = p.starmap( fit_model_run, p_args )
  
  stats = merge_stats( stats, single_run_stats, n_av )
  
  if metaseg.get("GREEDY_VARSELECT"):
    greedy_varselect( Xa, ya, y0a, X_names )
  
  mean_stats, _ = dump_stats( stats, metrics )
  
  generate_lasso_plots( stats, mean_stats, X_names, class_names )



if __name__ == '__main__':
  
  main()
      
      
      
