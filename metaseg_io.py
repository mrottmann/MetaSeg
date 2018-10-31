
import os
import pickle
import h5py
import numpy as np

# parameters from global_defs.py, please adjust variables and paths
from global_defs import MetaSeg
metaseg = MetaSeg()



########################################################
# probs_gt_save( probs, gt, image_name, i ):
# This routine stores softmax probabilities, ground truth, and image file name
# in a hdf5 file named "probs#i.py" in the directory metaseg.get("PROBS_DIR").
# In order to run the computation and analysis of the metrics, please make sure that:
#    * all variables and paths in "global_defs.py" are set appropriately,
#    * all necessary packages are installed (cf. README),
#    * your inputs to "probs_gt_save(...)" meet all conditions stated below.
# The visualization part may require minor adjustments for the label color code.
# Please check the NOTE in the header of "metaseg_eval.py" for further details.
#
#      probs: softmax probabilities for i-th image obtained from a neural network
#             3D numpy array (img_dim1,img_dim2,num_classes)
#         gt: ground truth class IDs per pixel for the i-th image
#             2D numpy array (img_dim1,img_dim2)
# image_name: name of the file that contains the i-th input image
#             string
#             "get_img_path_fname(...)" (see below) will search for
#             the pattern "image_name" in file names in all sub directories
#             of the path obtained from "metaseg.get("IMG_DIR")",
#             please cf. "global_defs.py" and make sure, that all
#             paths are set appropriately.
#          i: index of the current image, this will appear in the file name
#             of the stored data, number your outputs continuously from 0 to NUM_IMAGES-1
########################################################

def probs_gt_save( probs, gt, image_name, i ):
  
  file_names = []
  file_names.append(image_name.encode('utf8'))
  
  dump_dir = os.path.dirname( get_save_path_probs_i(i) )
  if not os.path.exists( dump_dir ):
    os.makedirs( dump_dir )
  
  f = h5py.File( get_save_path_probs_i(i), "w")
  f.create_dataset("probabilities", data=probs      )
  f.create_dataset("ground_truths", data=gt         )
  f.create_dataset("file_names"   , data=file_names )
  
  print("file stored:", get_save_path_probs_i(i) )
  
  f.close()



def probs_gt_load( i ):
  
  f_probs = h5py.File( get_save_path_probs_i(i) , "r")
  probs   = np.asarray( f_probs['probabilities'] )
  gt      = np.asarray( f_probs['ground_truths'] )
  probs   = np.squeeze( probs )
  gt      = np.squeeze( gt[0] )
  
  return probs, gt, f_probs['file_names'][0].decode('utf8')



 
def metrics_dump( metrics, i ):

  dump_path = get_save_path_metrics_i( i )
  dump_dir  = os.path.dirname( dump_path )

  if not os.path.exists( dump_dir ):
    os.makedirs( dump_dir )

  pickle.dump( metrics, open( dump_path, "wb" ) )



def metrics_load( i ):
  
  read_path = get_save_path_metrics_i( i )
  metrics = pickle.load( open( read_path, "rb" ) )
  
  return metrics



def components_dump( components, i ):

  dump_path = get_save_path_components_i( i )
  dump_dir  = os.path.dirname( dump_path )

  if not os.path.exists( dump_dir ):
    os.makedirs( dump_dir )

  pickle.dump( components, open( dump_path, "wb" ) )



def components_load( i ):
  
  read_path = get_save_path_components_i( i )
  components = pickle.load( open( read_path, "rb" ) )
  
  return components



def get_save_path_metrics_i( i ):
  
  return metaseg.get("METRICS_DIR") + "metrics" + str(i) +".p"



def get_save_path_components_i( i ):
  
  return metaseg.get("COMPONENTS_DIR") + "components" + str(i) +".p"



def get_save_path_probs_i( i ):
  
  return metaseg.get("PROBS_DIR") + "probs_" + str(i) +".hdf5"



def get_iou_seg_vis_path_i( i ):
  
  return metaseg.get("IOU_SEG_VIS_DIR") + "img" + str(i) +".hdf5"



def get_save_path_stats():
  
  return metaseg.get("STATS_DIR") + "stats.p"



def get_img_path_fname( filename ):
  
  path = []
  for root, dirnames, filenames in os.walk(metaseg.get("IMG_DIR")):
    for fn in filenames:
      if filename in fn:
        path = os.path.join(root, fn)
        break
      
  if path == []:
    print("file", filename, "not found.")
  
  return path
