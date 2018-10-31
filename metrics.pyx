 
import numpy as np
cimport numpy as np


def segment_search( int i, int j, unsigned char[:,:] seg, short int seg_ind, np.ndarray marked_array, np.ndarray flag_array,
                    float[:,:,:] probs, unsigned char[:,:] gt, heatmaps, metrics,
                    unsigned short int[:,:] members_k, unsigned short int[:,:] members_l,
                    int nclasses, int x_max, int y_max ):
  
  cdef int k, l, ii, jj, x, y, n_in, n_bd, c, I, U, flag_max_x, flag_min_x, flag_max_y, flag_min_y, ic
  cdef unsigned char[:,:] flag
  cdef short int[:,:] marked
  
  if seg[i,j] < 255:
    
    n_in, n_bd = 0, 0
    c = seg[i,j]
    members_k[0,0], members_k[0,1] = i, j
    marked = marked_array
    
    flag_min_x = flag_max_x = i
    flag_min_y = flag_max_y = j
    
    flag = flag_array
    flag[i,j] = 1
    I, U = 0, 0
    marked[i,j] = seg_ind
    
    for m in metrics:
      metrics[m].append( 0 )
    
    # go through union of current segment and corresponding ground truth
    # and identify all inner pixels, boundary pixels and
    # pixels where ground_truth and prediction match
    k = 1
    l = 0
    while k > 0 or l > 0:
      
      flag_k = 0
      
      if k > 0:
        k -= 1
        x, y = members_k[k]
        flag_k = 1
      elif l > 0:
        l -= 1
        x, y = members_l[l]
      
      if flag_k:
        for ii in range(max(x-1,0),min(x+2,x_max)):
          for jj in range(max(y-1,0),min(y+2,y_max)):
            if seg[ii,jj] == c and marked[ii,jj] == 0:
                marked[ii,jj] = seg_ind
                flag[ii,jj] = 1
                if ii > flag_max_x:
                  flag_max_x = ii
                elif ii < flag_min_x:
                  flag_min_x = ii
                if jj > flag_max_y:
                  flag_max_y = jj
                elif jj < flag_min_y:
                  flag_min_y = jj
                members_k[k,0], members_k[k,1] = ii, jj
                k += 1
            elif seg[ii,jj] != c:
                marked[x,y] = -seg_ind
                if gt != []:
                  if gt[ii,jj] == c and flag[ii,jj]==0:
                    flag[ii,jj] = 1
                    if ii > flag_max_x:
                      flag_max_x = ii
                    elif ii < flag_min_x:
                      flag_min_x = ii
                    if jj > flag_max_y:
                      flag_max_y = jj
                    elif jj < flag_min_y:
                      flag_min_y = jj
                    members_l[l,0], members_l[l,1] = ii, jj
                    l += 1

      if not flag_k and gt != []:
        if I == 0:
          break
        for ii in range(max(x-1,0),min(x+2,x_max)):
          for jj in range(max(y-1,0),min(y+2,y_max)):
            #if gt[ii,jj] == c and flag[ii,jj]==0: # ordinary IoU
            if gt[ii,jj] == c and flag[ii,jj]==0 and seg[ii,jj] != c: # IoU_adj
              flag[ii,jj] = 1
              if ii > flag_max_x:
                flag_max_x = ii
              elif ii < flag_min_x:
                flag_min_x = ii
              if jj > flag_max_y:
                flag_max_y = jj
              elif jj < flag_min_y:
                flag_min_y = jj
              members_l[l,0], members_l[l,1] = ii, jj
              l += 1

      if flag_k:
        if marked[x,y] in [seg_ind,-seg_ind]:
          # update heap maps
          if marked[x,y] == seg_ind:
            for h in heatmaps:
              metrics[h+"_in"][-1] += heatmaps[h][x,y]
            n_in += 1
          elif marked[x,y] == -seg_ind:
            for h in heatmaps:
              metrics[h+"_bd"][-1] += heatmaps[h][x,y]
            n_bd += 1
          for ic in range(nclasses):
            metrics["cprob"+str(ic)][-1] += probs[x,y,ic]
          #metrics["mean_x"][-1] += x
          #metrics["mean_y"][-1] += y
          if gt != []:
            if gt[x,y] == c:
              I += 1
              
      U += 1
      
    for ii in range(flag_min_x,flag_max_x+1):
      for jj in range(flag_min_y,flag_max_y+1):
        flag[ii,jj] = 0
    
    # compute all metrics
    metrics["class"   ][-1] = c
    if gt != []:
      metrics["iou"     ][-1] = float(I) / float(U)
      metrics["iou0"    ][-1] = int(I == 0)
    else:
      metrics["iou"     ][-1] = -1
      metrics["iou0"    ][-1] = -1
    metrics["S"       ][-1] = n_in + n_bd
    metrics["S_in"    ][-1] = n_in
    metrics["S_bd"    ][-1] = n_bd
    metrics["S_rel"   ][-1] = float( n_in + n_bd ) / float(n_bd)
    metrics["S_rel_in"][-1] = float( n_in ) / float(n_bd)
    #metrics["mean_x"][-1] /= ( n_in + n_bd )
    #metrics["mean_y"][-1] /= ( n_in + n_bd )
    
    for nc in range(nclasses):
      metrics["cprob"+str(nc)][-1] /= ( n_in + n_bd )
    
    for h in heatmaps:
      metrics[h          ][-1] = (metrics[h+"_in"][-1] + metrics[h+"_bd"][-1]) / float( n_in + n_bd )
      if ( n_in > 0 ):
        metrics[      h+"_in"][-1] /= float(n_in)
      metrics[h+"_bd"    ][-1] /= float(n_bd)
        
      metrics[h+"_rel"   ][-1] = metrics[h      ][-1] * metrics["S_rel"   ][-1]
      metrics[h+"_rel_in"][-1] = metrics[h+"_in"][-1] * metrics["S_rel_in"][-1]


    seg_ind +=1
      
  return marked_array, metrics, seg_ind



def entropy( probs ):

  E = np.sum( np.multiply( probs, np.log(probs+np.finfo(np.float32).eps) ) , axis=-1) / np.log(1.0/probs.shape[-1])
  return np.asarray( E, dtype="float32" )



def probdist( probs ):

  cdef int i, j
  
  arrayA = np.asarray(np.argsort(probs,axis=-1), dtype="uint8")
  arrayD = np.ones( probs.shape[:-1], dtype="float32" )
  
  cdef float[:,:,:] P = probs
  cdef float[:,:]   D = arrayD
  cdef char[:,:,:]  A = arrayA
  
  for i in range( arrayD.shape[0] ):
    for j in range( arrayD.shape[1] ):
      D[i,j] = ( 1 - P[ i, j, A[i,j,-1] ] + P[ i, j, A[i,j,-2] ] )
  
  return arrayD



def prediction(probs, gt, ignore=True ):
    
  pred = np.asarray( np.argmax( probs, axis=-1 ), dtype="uint8" )
  
  if ignore == True:
    pred[ gt==255 ] = 255
  
  return pred



def compute_metrics( probs, gt ):
  
  cdef int i, j
  cdef short int seg_ind
  cdef np.ndarray marked
  cdef np.ndarray members_k
  cdef np.ndarray members_l
  cdef short int[:,:] M
  
  nclasses  = probs.shape[-1]
  dims      = np.asarray( probs.shape[:-1], dtype="uint16" )
  gt        = np.asarray( gt, dtype="uint8" )
  probs     = np.asarray( probs, dtype="float32" )
  seg       = np.asarray( prediction(probs, gt, ignore=True ), dtype="uint8" )
  marked    = np.zeros( dims, dtype="int16" )
  members_k = np.zeros( (np.prod(dims), 2 ), dtype="uint16" )
  members_l = np.zeros( (np.prod(dims), 2 ), dtype="uint16" )
  flag      = np.zeros( dims, dtype="uint8" )
  M         = marked
  
  heatmaps = { "E": entropy( probs ), "D": probdist( probs ) }  
  
  metrics = { "iou": list([]), "iou0": list([]), "class": list([]) } #, "mean_x": list([]), "mean_y": list([]) }
  
  for m in list(heatmaps)+["S"]:
    metrics[m          ] = list([])
    metrics[m+"_in"    ] = list([])
    metrics[m+"_bd"    ] = list([])
    metrics[m+"_rel"   ] = list([])
    metrics[m+"_rel_in"] = list([])
    
  for i in range(nclasses):
    metrics['cprob'+str(i)] = list([])
  
  seg_ind = 1
  
  for i in range(dims[0]):
    for j in range(dims[1]):
      if M[i,j] == 0:
        
        marked, metrics, seg_ind = segment_search( i, j, seg, seg_ind, marked, flag, probs, gt, heatmaps, metrics, members_k, members_l, nclasses, dims[0], dims[1] )
        
  
  return metrics, marked
  
  
  
# testing
def test_metrics():
  
  probs = np.ones((10,10,2))
  for i in range(probs.shape[0]):
    probs[i,:,0] = (i+1)/probs.shape[0]
    probs[i,:,1] = 1 - probs[i,:,0]
    
  gt = np.ones((10,10))
  
  metrics, marked = compute_metrics( probs, gt )
  
  print( "probs[:,:,0]:" )
  print( np.asarray(probs[:,:,0]) )
  print( "marked:" )
  print( marked )
  M = sorted([m for m in metrics])
  for m in M:
    print( m, ":", metrics[m] )
  print(" ")




