

class MetaSeg:
  
  METASEG_MODEL_NAMES    = [ "xc.mscl.os8", "mn.sscl.os16" ]
  METASEG_CLASS_DTYPES   = [ "one_hot_classes", "probs", "none" ]

  METASEG_MODEL_NAME     = METASEG_MODEL_NAMES[1]
  
  METASEG_NUM_CROPS      = 16

  METASEG_READ_DATA_PATH = "/home/schubert/metaseg/io/nested/"
  METASEG_MY_IO_PATH     = "/home/rottmann/metaseg/io/nested/"
  
  METASEG = { "MODEL_NAMES"          : METASEG_MODEL_NAMES, \
              "CLASS_DTYPES"         : METASEG_CLASS_DTYPES, \
              "MODEL_NAME"           : METASEG_MODEL_NAME, \
              "PROBS_DIR"            : METASEG_READ_DATA_PATH + "probs/probs_agg" + \
                                       str(METASEG_NUM_CROPS) + "/"+ METASEG_MODEL_NAME + "/", \
              "GT_DIR"               : METASEG_READ_DATA_PATH + "../../deeplab/datasets/cityscapes/gtFine/val", \
              "IMG_DIR"              : METASEG_READ_DATA_PATH + "../../deeplab/datasets/cityscapes/leftImg8bit/val", \
              "METRICS_DIR"          : METASEG_MY_IO_PATH     + "metrics/"     + METASEG_MODEL_NAME + "/", \
              "COMPONENTS_DIR"       : METASEG_MY_IO_PATH     + "components/"  + METASEG_MODEL_NAME + "/", \
              "RESULTS_DIR"          : METASEG_MY_IO_PATH     + "results/"     + METASEG_MODEL_NAME + "/", \
              "IOU_SEG_VIS_DIR"      : METASEG_MY_IO_PATH     + "iou_seg_vis/" + METASEG_MODEL_NAME + "/", \
              "STATS_DIR"            : METASEG_MY_IO_PATH     + "stats/"       + METASEG_MODEL_NAME + "/", \
              "DEEPLAB_PARENT_DIR"   : METASEG_READ_DATA_PATH + "../../", \
              "KLDIV_DIR"            : METASEG_READ_DATA_PATH + "probs/kl_div/"+str(METASEG_NUM_CROPS)+"/" + METASEG_MODEL_NAME + "/", \
              "VARDIFF_DIR"          : METASEG_READ_DATA_PATH + "probs/var_diff"+str(METASEG_NUM_CROPS)+"/" + METASEG_MODEL_NAME + "/", \
              "VARENTROPY_DIR"       : METASEG_READ_DATA_PATH + "probs/var_entropy"+str(METASEG_NUM_CROPS)+"/" + METASEG_MODEL_NAME + "/", \
              "VARVARRAT_DIR"        : METASEG_READ_DATA_PATH + "probs/var_varrat"+str(METASEG_NUM_CROPS)+"/" + METASEG_MODEL_NAME + "/", \
              "NUM_IMAGES"           : 500, \
              "NUM_CORES"            : 40, \
              "NUM_LASSO_AVERAGES"   : 10, \
              "NUM_LASSO_LAMBDAS"    : 50, \
              "COMPUTE_IOU"          : False, \
              "COMPUTE_METRICS"      : True, \
              "VISUALIZE_METRICS"    : False, \
              "ANALYZE_METRICS"      : True, \
              "USE_KLDIV"            : True, \
              "GREEDY_VARSELECT"     : False, \
              "NUM_VARSELECT"        : 12, \
              "USE_NN"               : False, \
              "CLASS_DTYPE"          : METASEG_CLASS_DTYPES[1], \
              "NUM_CROPS"            : METASEG_NUM_CROPS, \
              "CROP_DIST"            : 20  
            }
  

  
  def __init__(self):
    for m in self.METASEG:
      setattr(self, m, self.METASEG[m])
  
  
  
  def get( self, name ):
    try:
      return getattr(self, name)
    except:
      print("MetaSeg:", name, "not found.")
      return 0
  
  
  
  def set( self, name, value ):
    if getattr(self, name):
      setattr( self, name, value )
  
  
  
  def print_attr( self ):
    for m in sorted(self.METASEG):
      print(m, ":", getattr(self, m) )


  
  def set_from_argv( self, argv ):
    cline = str()
    for i in range(len(argv)):
      cline += str(argv[i])+" "
    
    if "--" in cline:
      commands = cline.split("--")
      
      for c in commands[1:]:
        c0, c1 = c.split("=")[0], c.split("=")[1]
        
        if self.get( c0 ):
          dtype = type( self.get( c0 ) )
          self.set( c0, dtype(c1) )
    
    
    
    
