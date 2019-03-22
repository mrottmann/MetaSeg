
import os
import os.path
import time
import sys
import argparse
from scipy.ndimage import zoom
from PIL import Image
from multiprocessing import Pool
from metaseg_io import metaseg, probs_gt_save, probs_gt_load
import h5py
import numpy as np
from skimage.transform import rescale, resize
import pickle
import time

import multiprocessing as mp
from multiprocessing import Process


def merge_crops_i( probs, gt, fname, heatmap_list, img_list, probs_list, p_list, i ):
  
  p_list.append(i)
  
  probs[0] = probs0 = np.squeeze(probs[0].copy())
  probs_agg = np.squeeze(probs[0].copy())
  weights = np.ones(probs0.shape).astype("float32")
  
  for j in range(1,len(probs)):
  
    offset = int(metaseg.get("CROP_DIST"))
    probs[j] = np.squeeze( probs[j] )
    ps = probs[j].shape
    
    probs_j = resize(probs[j], (ps[0]-j*offset, ps[1]-2.0*j*offset), mode="reflect" )
    
    #if j % 2 == 1:
      #probs_j = np.flip( probs_j, axis=1 )
    
    probs[j] = np.divide( probs_agg, weights )
    
    pfilter_x = np.ones( (probs_j.shape[0],1) )
    for k in range(int(offset/2)):
      pfilter_x[k,0] = pfilter_x[-1-k,0] = float(k) / ( float(offset) / 2.0 ) 
    
    pfilter_y = np.ones( (1,probs_j.shape[1]) )
    for k in range(offset):
      pfilter_y[0,k] = pfilter_y[0,-1-k] = float(k) / ( float(offset) )
    
    pfilter = np.matmul(pfilter_x,pfilter_y).astype("float32")
    pfilter = pfilter.reshape( pfilter.shape + (1,) )
    pfilter = np.tile( pfilter, probs_j.shape[-1] )
    
    weights[ int(j*offset/2):-int(j*offset/2), j*offset:-j*offset, : ] += pfilter
    probs_agg[ int(j*offset/2):-int(j*offset/2), j*offset:-j*offset, : ] +=  np.multiply( probs_j, pfilter ).astype("float32")
    
    probs[j][ int(j*offset/2):-int(j*offset/2), j*offset:-j*offset, : ] = \
      np.multiply( probs_j, pfilter ).astype("float32") + \
      np.multiply( probs[j][ int(j*offset/2):-int(j*offset/2), j*offset:-j*offset, : ], 1.0 - pfilter ).astype("float32")
  
  
  probs_agg      = np.divide( probs_agg, weights ).astype("float32")
  probs_agg      = np.asarray( probs_agg, dtype="float32" )
  kl_div_heatmap = kl_div( probs0, probs_agg, i )
  
  dump_path = metaseg.get("PROBS_DIR") + "probs_" + str(i) + ".hdf5"  
  probs_list.append( dict({ "data": probs_agg, "gt": gt, "fname": fname, "save_path" : dump_path, "type": "h5py" }) )
  
  probs                = np.asarray([np.asarray(probs[j]) for j in range(len(probs))], dtype="float32")
  entr                 = np.sum( probs * np.log(probs+np.finfo(float).eps), axis=-1 )
  var_entropy_heatmap  = np.var( entr, axis=0 )
  mean_entropy_heatmap = np.mean( entr, axis=0 )
  
  probs                = np.sort( probs, axis=-1 )
  variation_ratios     = 1.0 - probs[:,:,:,-1]
  var_varrat_heatmap   = np.var( variation_ratios, axis=0 )
  mean_varrat_heatmap  = np.mean( variation_ratios, axis=0 )
  
  diff                 = variation_ratios + probs[:,:,:,-2]
  var_diff_heatmap     = np.var( diff, axis=0 )
  mean_diff_heatmap    = np.mean( diff, axis=0 )
  
  dump_path = metaseg.get("KLDIV_DIR") + "kl_div" + str(i) + ".p"
  heatmap_list.append( dict({"data": kl_div_heatmap, "save_path": dump_path, "type": "pickle" }) )
  
  dump_path = metaseg.get("VARENTROPY_DIR") + "var_entropy" + str(i) + ".p"  
  heatmap_list.append( dict({"data": var_entropy_heatmap, "save_path": dump_path, "type": "pickle" }) )
  dump_path = metaseg.get("VARENTROPY_DIR") + "mean_entropy" + str(i) + ".p"
  heatmap_list.append( dict({"data": mean_entropy_heatmap, "save_path": dump_path, "type": "pickle" }) )

  dump_path = metaseg.get("VARVARRAT_DIR") + "var_varrat" + str(i) + ".p"  
  heatmap_list.append( dict({"data": var_varrat_heatmap, "save_path": dump_path, "type": "pickle" }) )
  dump_path = metaseg.get("VARVARRAT_DIR") + "mean_varrat" + str(i) + ".p"  
  heatmap_list.append( dict({"data": mean_varrat_heatmap, "save_path": dump_path, "type": "pickle" }) )
  
  dump_path = metaseg.get("VARDIFF_DIR") + "var_diff" + str(i) + ".p"  
  heatmap_list.append( dict({"data": var_diff_heatmap, "save_path": dump_path, "type": "pickle" }) )
  dump_path = metaseg.get("VARDIFF_DIR") + "mean_diff" + str(i) + ".p"
  heatmap_list.append( dict({"data": mean_diff_heatmap, "save_path": dump_path, "type": "pickle" }) )

  img = kl_div_heatmap.copy()
  img /= np.max(img.flatten())
  img = np.asarray( img*255, dtype="uint8" )
  image = Image.fromarray(img, 'L')
  save_path = metaseg.get("KLDIV_DIR") + "img/kl_div_img" + str(i) + ".png"
  img_list.append( dict({"data": image, "save_path": save_path, "type": "image" }) )
  
  img = var_varrat_heatmap.copy()
  img /= np.max(img.flatten())
  img = np.asarray( img*255, dtype="uint8" )
  image = Image.fromarray(img, 'L')
  save_path = metaseg.get("VARVARRAT_DIR") + "img/var_varrat_img" + str(i) + ".png"
  img_list.append( dict({"data": image, "save_path": save_path, "type": "image" }) )
  
  img = mean_varrat_heatmap.copy()
  img /= np.max(img.flatten())
  img = np.asarray( img*255, dtype="uint8" )
  image = Image.fromarray(img, 'L')
  save_path = metaseg.get("VARVARRAT_DIR") + "img/mean_varrat_img" + str(i) + ".png"
  img_list.append( dict({"data": image, "save_path": save_path, "type": "image" }) )
  
  img = var_entropy_heatmap.copy()
  img /= np.max(img.flatten())
  img = np.asarray( img*255, dtype="uint8" )
  image = Image.fromarray(img, 'L')
  save_path = metaseg.get("VARENTROPY_DIR") + "img/var_entropy_img" + str(i) + ".png"
  img_list.append( dict({"data": image, "save_path": save_path, "type": "image" }) )
  
  img = mean_entropy_heatmap.copy()
  img /= np.max(img.flatten())
  img = np.asarray( img*255, dtype="uint8" )
  image = Image.fromarray(img, 'L')
  save_path = metaseg.get("VARENTROPY_DIR") + "img/mean_entropy_img" + str(i) + ".png"
  img_list.append( dict({"data": image, "save_path": save_path, "type": "image" }) )
  
  img = var_diff_heatmap.copy()
  img /= np.max(img.flatten())
  img = np.asarray( img*255, dtype="uint8" )
  image = Image.fromarray(img, 'L')
  save_path = metaseg.get("VARDIFF_DIR") + "img/var_diff_img" + str(i) + ".png"
  img_list.append( dict({"data": image, "save_path": save_path, "type": "image" }) )
  
  img = mean_diff_heatmap.copy()
  img /= np.max(img.flatten())
  img = np.asarray( img*255, dtype="uint8" )
  image = Image.fromarray(img, 'L')
  save_path = metaseg.get("VARDIFF_DIR") + "img/mean_diff_img" + str(i) + ".png"
  img_list.append( dict({"data": image, "save_path": save_path, "type": "image" }) )
  
  
  p_list.remove(i)
  
  
  
def kl_div( P, Q, i ):
    
  kl_div_heatmap = np.sum( 0.5* ( P * np.log(P/Q) + Q * np.log(Q/P) ) , axis=-1 )
  
  return kl_div_heatmap



def list_item_save( item ):
    
  save_dir  = os.path.dirname( item["save_path"] )
  if not os.path.exists( save_dir ):
    os.makedirs( save_dir )
    
  if item["type"] == "pickle":
    pickle.dump( item["data"], open(item["save_path"], "wb") )
  elif item["type"] == "image":
    item["data"].save( item["save_path"] )
  elif item["type"] == "h5py":
    file_names = []
    file_names.append(item["fname"].encode('utf8'))
    f = h5py.File( item["save_path"], "w")
    f.create_dataset("probabilities", data=item["data"].astype("float32") )
    f.create_dataset("ground_truths", data=item["gt"].astype("uint8")   )
    f.create_dataset("file_names"   , data=file_names   )    
    f.close()
    

  print(item["save_path"],"saved")
    
    

def load_and_process_data( ):
  
  num_workers = metaseg.get("NUM_CORES") - 1
  
  mgr = mp.Manager()
  
  p_list       = mgr.list()
  probs_list   = mgr.list()
  heatmap_list = mgr.list()
  img_list     = mgr.list()
  
  max_data_list_len = 10
  
  
  for i in range(metaseg.get("NUM_IMAGES")):
    
    print("loaded data chunk",i)
    
    probs = []
    gt = []
    fname = []
    
    for j in range(0,metaseg.get("NUM_CROPS")):
    
      probs_dir = metaseg.get("METASEG_READ_DATA_PATH") + "probs/crop_"+str(j)+"/" + metaseg.get("MODEL_NAME")
      with h5py.File( probs_dir + "/" + "probs_" +str(i) + ".hdf5" , "r") as f_probs:
        probs.append( np.asarray( f_probs["probabilities"] ) )
        if j == 0:
          gt = np.asarray( f_probs["ground_truths"] )
          fname = f_probs['file_names'][0].decode('utf8')
    
    while len(p_list) > num_workers:
      
      while len(heatmap_list) > 0:
        list_item_save( heatmap_list.pop() )
      while len(img_list) > 0:
        list_item_save( img_list.pop() )
      while len(probs_list) > 0:
        list_item_save( probs_list.pop() )
      
      time.sleep(0.1)
        
    p = Process( target=merge_crops_i, args=( probs.copy(), gt.copy(), fname, \
                                              heatmap_list, img_list, probs_list, p_list, i ) )
    p.start()
    
    if len(heatmap_list) > max_data_list_len:
      while len(heatmap_list) > 0:
        list_item_save( heatmap_list.pop() )
    if len(img_list) > max_data_list_len:
      while len(img_list) > 0:
        list_item_save( img_list.pop() )
    if len(probs_list) > max_data_list_len:
      while len(probs_list) > 0:
        list_item_save( probs_list.pop() )
        
  print("waiting for workers to finish...")
  
  while len(p_list) > 0:
    time.sleep(0.1)
  
  while len(heatmap_list) > 0:
    list_item_save( heatmap_list.pop() )
  while len(img_list) > 0:
    list_item_save( img_list.pop() )
  while len(probs_list) > 0:
    list_item_save( probs_list.pop() )



if __name__ == '__main__':
  
  load_and_process_data()


 
