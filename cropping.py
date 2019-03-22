
import os
import os.path
import time
import sys
import argparse
from PIL import Image
from multiprocessing import Pool
from metaseg_io import *



def crop_img_i( offset, img_dir, cropped_img_dir, subfolder, filename, mode, crop_number ):

  input_image  = Image.open( img_dir + subfolder + filename )
      
  width, height = input_image.size   # Get dimensions
  
  new_width = width - 2*offset
  new_height = height - offset

  left = (width - new_width)/2
  top = (height - new_height)/2
  right = (width + new_width)/2
  bottom = (height + new_height)/2
  
  cropped_input_image = input_image.crop((left, top, right, bottom))
  
  if mode == "img":
    cropped_input_image = cropped_input_image.resize((width,height),Image.BICUBIC)
  elif mode == "gt":
    cropped_input_image = cropped_input_image.resize((width,height),Image.NEAREST)
  
  if not os.path.exists( cropped_img_dir + subfolder ):
    try:
      os.makedirs( cropped_img_dir + subfolder )
    except:
      print("directory has been created in the meantime")
      
  
  cropped_input_image.save( cropped_img_dir + subfolder + filename )
  
  print(filename,"processed")



if __name__ == '__main__':


  parser = argparse.ArgumentParser(description=" ")
  parser.add_argument('--IMG_DIR', type=str, default=None, metavar='img_dir',
                      help=" ")
  parser.add_argument('--GT_DIR', type=str, default=None, metavar='gt_dir',
                      help=" ")
  parser.add_argument('--CROPPED_IMG_DIR', type=str, default=None, metavar='cropped_img_dir',
                      help=" ")
  parser.add_argument('--CROPPED_GT_DIR', type=str, default=None, metavar='cropped_gt_dir',
                      help=" ")
  parser.add_argument('--CROP_NUMBER', type=int, default=8, metavar='crop_number',
                      help=" ")

  args = parser.parse_args()

  num_cores = metaseg.get("NUM_CORES")
        
  p = Pool(num_cores)

  offset = args.CROP_NUMBER * metaseg.get("CROP_DIST")
  
  for img_dir, cropped_img_dir, mode in [ [args.IMG_DIR, args.CROPPED_IMG_DIR, "img"] , [args.GT_DIR, args.CROPPED_GT_DIR, "gt"] ]:

    for root, dirnames, filenames in os.walk(img_dir):
      
      subfolder = root.replace(img_dir,"")+"/"
          
      p_args = [ ( offset, img_dir, cropped_img_dir, subfolder, filename, mode, args.CROP_NUMBER ) for filename in filenames if ".png" in filename ]
      
      p.starmap( crop_img_i, p_args )
      
      
      
