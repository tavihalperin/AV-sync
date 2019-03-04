#!/usr/bin/python

import time, os, pdb, argparse, subprocess
import numpy as np
import tensorflow as tf
from skimage.transform import resize
import imageio

from scipy.interpolate import interp1d
from scipy import signal

# ========== ========== ========== ==========
# # PARSE ARGS
# ========== ========== ========== ==========

parser = argparse.ArgumentParser(description = "FaceTracker");
parser.add_argument('--data_dir', type=str, default='data/', help='Output direcotry');
parser.add_argument('videofile', type=str, default='', help='Input video file');
parser.add_argument('--crop_scale', type=float, default=0.5, help='Scale bounding box');
parser.add_argument('--min_track', type=int, default=30, help='Minimum facetrack duration');
opt = parser.parse_args();

base_name = os.path.splitext(os.path.basename(opt.videofile))[0]
setattr(opt,'out_dir', os.path.join(opt.data_dir, 'out', base_name))

# ========== ========== ========== ==========
# # IOU FUNCTION
# ========== ========== ========== ==========

def bb_intersection_over_union(boxA, boxB):
  
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])
 
  interArea = max(0, xB - xA) * max(0, yB - yA)
 
  boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
  boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
 
  iou = interArea / float(boxAArea + boxBArea - interArea)
 
  return iou

# ========== ========== ========== ==========
# # FACE TRACKING
# ========== ========== ========== ==========

def track_shot(opt, scenefaces):

  iouThres  = 0.5     # Minimum IOU between consecutive face detections
  numFail   = 3       # Number of missed detections allowed
  minSize   = 0.05    # Minimum size of faces
  tracks    = []

  while True:
    track     = []
    for faces in scenefaces:
      for face in faces:
        if track == []:
          track.append(face)
          faces.remove(face)
        elif face[0] - track[-1][0] <= numFail:
          iou = bb_intersection_over_union(face[1], track[-1][1])
          if iou > iouThres:
            track.append(face)
            faces.remove(face)
            continue
        else:
          break

    if track == []:
      break
    elif len(track) > opt.min_track:
      
      framenum    = np.array([ f[0] for f in track ])
      bboxes    = np.array([np.array(f[1]) for f in track])

      frame_i   = np.arange(framenum[0],framenum[-1]+1)

      bboxes_i    = []
      for ij in range(0,4):
        interpfn  = interp1d(framenum, bboxes[:,ij])
        bboxes_i.append(interpfn(frame_i))
      bboxes_i  = np.stack(bboxes_i, axis=1)

      if np.mean(bboxes_i[:,3]-bboxes_i[:,1]) > minSize:
        tracks.append([frame_i,bboxes_i])
  return tracks

# ========== ========== ========== ==========
# # VIDEO CROP AND SAVE
# ========== ========== ========== ==========
        
def crop_video(opt,track):

  reader = imageio.get_reader(os.path.join(opt.out_dir, 'video.avi'))

  fps = reader.get_meta_data()['fps']
  cropped_file = os.path.join(opt.out_dir, 'cropped')
  vOut = imageio.get_writer(cropped_file + 't.avi', fps=fps)

  fw, fh = reader.get_meta_data()['size']

  dets = [[], [], []]

  for det in track[1]:

    dets[0].append(((det[3]-det[1])*fw+(det[2]-det[0])*fh)/4) # H+W / 4
    dets[1].append((det[1]+det[3])*fw/2) # crop center x 
    dets[2].append((det[0]+det[2])*fh/2) # crop center y

  # Smooth detections
  dets[0] = signal.medfilt(dets[0],kernel_size=5)   
  dets[1] = signal.medfilt(dets[1],kernel_size=5)
  dets[2] = signal.medfilt(dets[2],kernel_size=7)

  for det in zip(*dets):

    cs  = opt.crop_scale

    bs  = det[0]            # Detection box size
    bsi = int(bs*(1+2*cs))  # Pad videos by this amount 

    try:
        frame = reader.get_next_data()
    except:
        break

    frame = np.pad(frame,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(0,0))
    my  = det[2]+bsi  # BBox center Y
    mx  = det[1]+bsi  # BBox center X

    face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]

    resized = resize(face,(224,224), mode='reflect', anti_aliasing=True)
    vOut.append_data((255*resized).round().clip(0,255).astype(np.uint8))

  audiotmp  = os.path.join(opt.out_dir, 'audio.wav')
  audiostart  = track[0][0]/fps
  audioend  = (track[0][-1]+1)/fps

  vOut.close()

  # ========== CROP AUDIO FILE ==========

  command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 -ss %.3f -to %.3f %s" %
             (os.path.join(opt.out_dir, 'video.avi'),audiostart,audioend,audiotmp)) #-async 1
  output = subprocess.call(command, shell=True, stdout=None)

  if output != 0:
    pdb.set_trace()

  # sample_rate, audio = wavfile.read(audiotmp)

  # ========== COMBINE AUDIO AND VIDEO FILES ==========

  command = ("ffmpeg -y -i %st.avi -i %s -c:v copy -c:a copy %s.avi" % (cropped_file ,audiotmp, cropped_file)) #-async 1
  output = subprocess.call(command, shell=True, stdout=None)

  if output != 0:
    pdb.set_trace()

  print('Written %s'%cropped_file)

  os.remove(cropped_file+'t.avi')


# ========== ========== ========== ==========
# # FACE DETECTION
# ========== ========== ========== ==========

def inference_video(opt):

  # Path to frozen detection graph. This is the actual model that is used for the object detection.
  PATH_TO_CKPT = './protos/frozen_inference_graph_face.pb'

  MIN_CONF = 0.3

  reader = imageio.get_reader(os.path.join(opt.out_dir, 'video.avi'))
  print(reader.get_meta_data())

  detection_graph = tf.Graph()
  with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
          serialized_graph = fid.read()
          od_graph_def.ParseFromString(serialized_graph)
          tf.import_graph_def(od_graph_def, name='')

  dets = []

  with detection_graph.as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=detection_graph, config=config) as sess:
      frame_num = 0;
      while True:
        try:
            image_np = reader.get_next_data()
        except:
            break

        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time
        
        score = scores[0]

        dets.append([]);
        for index in range(0,len(score)):
          if score[index] > MIN_CONF:
            dets[-1].append([frame_num, boxes[0][index].tolist(),score[index]])

        print('%s-%05d; %d dets; %.2f Hz' %
              (os.path.join(opt.out_dir,'video.avi'),
               frame_num,len(dets[-1]),(1/elapsed_time)))
        frame_num += 1

  return dets

# ========== ========== ========== ==========
# # EXECUTE DEMO
# ========== ========== ========== ==========

if not os.path.exists(opt.out_dir):
  os.makedirs(opt.out_dir)

command = ("ffmpeg -y -i %s -qscale:v 4 -async 1 -r 25 -deinterlace %s" %
           (opt.videofile, os.path.join(opt.out_dir, 'video.avi')))
output = subprocess.call(command, shell=True, stdout=None)
faces = inference_video(opt)

tracks = track_shot(opt, faces)
crop_video(opt, tracks[0])
