import tensorflow as tf
import tensorflow.keras.regularizers as regularizers
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.backend as K

import numpy as np
import json
import os
import zipfile
import copy
from skimage import io, transform
from random import randint
from scipy.stats import multivariate_normal

def genanchors(isize, ssize, asizes):
  '''
  Arguments
    isize: image size (h, w)
    ssize: feature map size (h, w)
    asizes: list of anchor sizes, (h, w)
  Return 
    abox4d: anchors in images, each anchor has shape (h, w, k, 4). 4 is lenngth of [y1, x1, y2, x2]
  '''

  # scale of feature map for origin image
  xsof = float(ssize[0])/float(isize[0])
  ysof = float(ssize[1])/float(isize[1])

  # all anchors
  # 4 is length of [y1, x1, y2, x2]
  abox4d = np.zeros(shape=(ssize[0], ssize[1], len(asizes), 4), dtype='float32') # channels last

  # iterate over feature map
  for i in range(0, ssize[0]):
    for j in range(0, ssize[1]):
      apoint = [(i + 0.5)/xsof, (j + 0.5)/ysof]

      # iterate over anchor at a point on feature map
      for k in range(0, len(asizes)):
        abox4d[i, j, k] = [
          apoint[0] - asizes[k][0]/2, 
          apoint[1] - asizes[k][1]/2, 
          apoint[0] + asizes[k][0]/2, 
          apoint[1] + asizes[k][1]/2
        ]
        
  abox2d = abox4d.reshape((-1, 4))
  return abox2d

def comiou2d(abox_2dtensor, bbox_2dtensor):
  '''
  Compute IoU
  '''

  b1y1 = bbox_2dtensor[:, 0:1]
  b1x1 = bbox_2dtensor[:, 1:2]
  b1y2 = bbox_2dtensor[:, 2:3]
  b1x2 = bbox_2dtensor[:, 3:4]

  b2y1 = abox_2dtensor[:, 0:1]
  b2x1 = abox_2dtensor[:, 1:2]
  b2y2 = abox_2dtensor[:, 2:3]
  b2x2 = abox_2dtensor[:, 3:4]

  b3y1 = tf.math.maximum(x=b1y1, y=b2y1)
  b3y2 = tf.math.minimum(x=b1y2, y=b2y2)
  b3x1 = tf.math.maximum(x=b1x1, y=b2x1)
  b3x2 = tf.math.minimum(x=b1x2, y=b2x2)

  # area of box 1
  s1 = (b1y2 - b1y1) * (b1x2 - b1x1)

  # area of box 2
  s2 = (b2y2 - b2y1) * (b2x2 - b2x1)

  # area of box 3
  s3 = tf.where(
    condition=tf.math.logical_or(
      x=tf.math.less(x=b3y2 - b3y1, y=0.0),
      y=tf.math.less(x=b3x2 - b3x1, y=0.0)),
    x=0.0,
    y=(b3y2 - b3y1) * (b3x2 - b3x1))

  return s3/(s1+s2-s3)

def comiou(bbox, pred_bbox):
  '''
  Compute IoU
  '''

  b1y1 = bbox[0]
  b1x1 = bbox[1]
  b1y2 = bbox[2]
  b1x2 = bbox[3]

  b2y1 = pred_bbox[0]
  b2x1 = pred_bbox[1]
  b2y2 = pred_bbox[2]
  b2x2 = pred_bbox[3]

  b3y1 = max(b1y1, b2y1)
  b3y2 = min(b1y2, b2y2)
  b3x1 = max(b1x1, b2x1)
  b3x2 = min(b1x2, b2x2)

  # area of box 1
  s1 = (b1y2 - b1y1) * (b1x2 - b1x1)

  # area of box 2
  s2 = (b2y2 - b2y1) * (b2x2 - b2x1)

  if b3y2 < b3y1:
    return 0

  if b3x2 < b3x1:
    return 0

  s3 = (b3y2 - b3y1) * (b3x2 - b3x1)

  return s3/(s1+s2-s3)

def comloc2d(bbox_2dtensor, abox_2dtensor):
  '''
  Compute bounding box error
  '''

  b1y1 = bbox_2dtensor[:, 0:1]
  b1x1 = bbox_2dtensor[:, 1:2]
  b1y2 = bbox_2dtensor[:, 2:3]
  b1x2 = bbox_2dtensor[:, 3:4]

  b2y1 = abox_2dtensor[:, 0:1]
  b2x1 = abox_2dtensor[:, 1:2]
  b2y2 = abox_2dtensor[:, 2:3]
  b2x2 = abox_2dtensor[:, 3:4]

  h = b1y2 - b1y1
  w = b1x2 - b1x1
  y = b1y1 + 0.5*h
  x = b1x1 + 0.5*w

  ha = b2y2 - b2y1
  wa = b2x2 - b2x1
  ya = b2y1 + 0.5*ha
  xa = b2x1 + 0.5*wa

  ty = (y - ya)/ha
  tx = (x - xa)/wa
  th = tf.math.log(h/ha)/tf.math.log(tf.constant(value=2.0, dtype='float32'))
  tw = tf.math.log(w/wa)/tf.math.log(tf.constant(value=2.0, dtype='float32'))

  t = tf.concat(values=[ty, tx, th, tw], axis=1)

  return t

def loc2box2d(box_2dtensor, bbe_2dtensor):
  '''
  Arguments
    box_4dtensor: (num_of_boxes, 4)
    bbe_2dtensor: (num_of_boxes, 4)
  Return
    tensor
  '''

  ya1 =  box_2dtensor[:, 0:1]
  xa1 =  box_2dtensor[:, 1:2]
  ya2 =  box_2dtensor[:, 2:3]
  xa2 =  box_2dtensor[:, 3:4]

  ha = ya2 - ya1
  wa = xa2 - xa1
  ya = ya1 + 0.5*ha
  xa = xa1 + 0.5*wa

  ty = bbe_2dtensor[:, 0:1]
  tx = bbe_2dtensor[:, 1:2]
  th = bbe_2dtensor[:, 2:3]
  tw = bbe_2dtensor[:, 3:4]

  y = ty*ha + ya
  x = tx*wa + xa
  h = tf.math.pow(2.0, th)*ha
  w = tf.math.pow(2.0, tw)*wa

  y1 = y - 0.5*h
  x1 = x - 0.5*w
  y2 = y + 0.5*h
  x2 = x + 0.5*w

  t = tf.concat(values=[y1, x1, y2, x2], axis=1)

  return t

def match_ic(prediction, batchy_2dtensor):
  '''
  Arguments
    prediction: (batch_size, total_classes)
    batchy_2dtensor: (batch_size, total_classes)
  Return
    tensor
  '''

  pred_y = tf.argmax(input=prediction, axis=-1)
  true_y = tf.argmax(input=batchy_2dtensor, axis=-1)
  tp = tf.where(
    condition=tf.math.logical_and(
      x=tf.math.equal(x=pred_y, y=1),
      y=tf.math.equal(x=pred_y, y=true_y)), 
    x=1, 
    y=0)
  fp = tf.where(
    condition=tf.math.logical_and(
      x=tf.math.equal(x=pred_y, y=1),
      y=tf.math.not_equal(x=pred_y, y=true_y)), 
    x=1, 
    y=0)
  fn = tf.where(
    condition=tf.math.logical_and(
      x=tf.math.equal(x=pred_y, y=0),
      y=tf.math.not_equal(x=pred_y, y=true_y)), 
    x=1, 
    y=0)
  tp = tf.math.reduce_sum(input_tensor=tp, axis=-1)
  fp = tf.math.reduce_sum(input_tensor=fp, axis=-1)
  fn = tf.math.reduce_sum(input_tensor=fn, axis=-1)
  
  return tp, fp, fn

def match_od(boxes, pboxes, iou_threshold=0.5):
  '''
  '''

  tp1 = 0
  tp2 = 0
  fn = 0
  fp = 0
  
  for i in range(len(boxes)):
    intersected = 0
    for j in range(len(pboxes)):
      iou = comiou(bbox=boxes[i], pred_bbox=pboxes[j])
      if iou >= iou_threshold:
        intersected = 1
        break

    tp1 += intersected
    fn += int(not intersected)

  for i in range(len(pboxes)):
    intersected = 0
    for j in range(len(boxes)):
      iou = comiou(bbox=boxes[j], pred_bbox=pboxes[i])
      if iou >= iou_threshold:
        intersected = 1
        break

    tp2 += intersected
    fp += int(not intersected)

  return min(tp1, tp2), fp, fn

def nms(abox_2dtensor, prediction, nsm_iou_threshold, nsm_score_threshold, nsm_max_output_size, total_classes):
  '''
  '''

  loc_2dtensor = prediction[:, total_classes+1:] # (h*w*k, 4)
  pbox_2dtensor = loc2box2d(box_2dtensor=abox_2dtensor, bbe_2dtensor=loc_2dtensor) # (h*w*k, 4)

  clz_2dtensor = prediction[:, :total_classes+1] # (h*w*k, total_classes+1)
  clz_1dtensor = tf.math.argmax(input=clz_2dtensor, axis=-1) # (h*w*k,)

  cancel = tf.where(
    condition=tf.math.less(x=clz_1dtensor, y=total_classes*tf.ones(shape=abox_2dtensor.shape[0], dtype='int64')),
    x=tf.ones(shape=abox_2dtensor.shape[0]),
    y=tf.zeros(shape=abox_2dtensor.shape[0])) # (h*w*k,)

  score_1dtensor = tf.math.reduce_max(input_tensor=clz_2dtensor, axis=-1) # (h*w*k,)
  score_1dtensor *= cancel # (h*w*k,)

  selected_indices, valid_outputs = tf.image.non_max_suppression_padded(
    boxes=pbox_2dtensor,
    scores=score_1dtensor,
    max_output_size=nsm_max_output_size,
    iou_threshold=nsm_iou_threshold,
    score_threshold=nsm_score_threshold,
    pad_to_max_output_size=True)

  box_2dtensor = tf.gather(params=pbox_2dtensor, indices=selected_indices) # (nsm_max_output_size, 4)
  clz_1dtensor = tf.gather(params=clz_1dtensor, indices=selected_indices) # (nsm_max_output_size,)
  clz_2dtensor = tf.expand_dims(input=clz_1dtensor, axis=1) # (nsm_max_output_size, 1)
  box_2dtensor = tf.cast(x=box_2dtensor, dtype='int32')
  clz_2dtensor = tf.cast(x=clz_2dtensor, dtype='int32')

  boxclz_2dtensor = tf.concat(values=[box_2dtensor, clz_2dtensor], axis=-1)

  return boxclz_2dtensor, valid_outputs

def match_od4(boxes, pboxes, areas=[45**2, 91**2, 181**2, 362**2, 724**2], iou_threshold=0.5):
  '''
  '''

  x_boxes = []
  s_boxes = []
  m_boxes = []
  l_boxes = []
  for y1, x1, y2, x2, _ in boxes:
    a = (y2 - y1)*(x2 - x1)
    if a >= areas[0] and a < areas[1]:
      x_boxes.append([y1, x1, y2, x2])
    elif a >= areas[1] and a < areas[2]:
      s_boxes.append([y1, x1, y2, x2])
    elif a >= areas[2] and a < areas[3]:
      m_boxes.append([y1, x1, y2, x2])
    elif a >= areas[3] and a < areas[4]:
      l_boxes.append([y1, x1, y2, x2])
    else:
      print('Box out of range')

  total_boxes = len(boxes)

  total_x_boxes = len(x_boxes)
  total_s_boxes = len(s_boxes)
  total_m_boxes = len(m_boxes)
  total_l_boxes = len(l_boxes)

  total_pboxes = len(pboxes)

  x_tp = 0
  s_tp = 0
  m_tp = 0
  l_tp = 0

  x_fn = 0
  s_fn = 0
  m_fn = 0
  l_fn = 0

  for i in range(total_boxes):
    intersected = 0
    for j in range(total_pboxes):
      iou = comiou(bbox=boxes[i], pred_bbox=pboxes[j])
      if iou >= iou_threshold:
        intersected = 1
        break

    y1, x1, y2, x2, _ = boxes[i]
    a = (y2 - y1)*(x2 - x1)
    if a < areas[1]:
      x_tp += intersected
      x_fn += int(not intersected)
    elif a >= areas[1] and a < areas[2]:
      s_tp += intersected
      s_fn += int(not intersected)
    elif a >= areas[2] and a < areas[3]:
      m_tp += intersected
      m_fn += int(not intersected)
    elif a >= areas[3]:
      l_tp += intersected
      l_fn += int(not intersected)

  x_fp = 0
  s_fp = 0
  m_fp = 0
  l_fp = 0

  for i in range(total_pboxes):
    not_intersected = 1
    for j in range(total_boxes):
      iou = comiou(bbox=boxes[j], pred_bbox=pboxes[i])
      if iou >= iou_threshold:
        not_intersected = 0
        break

    y1, x1, y2, x2, _ = pboxes[i]
    a = (y2 - y1)*(x2 - x1)
    if a < areas[1]:
      x_fp += not_intersected
    elif a >= areas[1] and a < areas[2]:
      s_fp += not_intersected
    elif a >= areas[2] and a < areas[3]:
      m_fp += not_intersected
    elif a >= areas[3]:
      l_fp += not_intersected

  return total_x_boxes, total_s_boxes, total_m_boxes, total_l_boxes, x_tp, s_tp, m_tp, l_tp, x_fn, s_fn, m_fn, l_fn, x_fp, s_fp, m_fp, l_fp


def contrast(image):
  '''
  '''

  image = image - 127
  image = image*(randint(100, 150)/100)
  image = image + 127
  image = np.clip(image, 0, 255)
  return image

def saturate(image):
  '''
  '''

  image = np.clip(image, randint(0, 32), randint(224, 255))
  image = image/np.max(image)
  image = image*255
  return image

def imbalance(image):
  '''
  '''

  if len(image.shape) != 3 or image.shape != 3:
    return image

  image[:, :, 0] += randint(0, 128)
  image[:, :, 1] += randint(0, 128)
  image[:, :, 2] += randint(0, 128)
  image = image/np.max(image)
  image = image*255
  return image

def shine(image, ishape):
  '''
  '''

  if len(image.shape) != 3 or image.shape != 3:
    return image

  pos = np.dstack(np.mgrid[0:ishape[0]:1, 0:ishape[1]:1])
  center_y = randint(0, ishape[0])
  center_x = randint(0, ishape[1])
  rv = multivariate_normal(mean=[center_y, center_x], cov=randint(0, 20000), allow_singular=True)
  heatmap2d = rv.pdf(pos)
  heatmap2d /= np.max(heatmap2d)
  heatmap2d = heatmap2d*randint(0, 128)
  indices = [0, 1, 2]
  np.random.shuffle(indices)
  indices = indices[:2]
  image[:, :, indices[0]] += heatmap2d
  image[:, :, indices[1]] += heatmap2d
  image = image/np.max(image)
  image = image*255
  return image

def blur(image, ishape):
  '''
  '''

  scale = randint(50, 100)/100
  image = transform.resize(image=image, output_shape=(int(scale*ishape[0]), int(scale*ishape[1])))
  image = transform.resize(image=image, output_shape=(int(ishape[0]), int(ishape[1])))
  image = image/np.max(image)
  image = image*255
  return image

def augcolor(image, ishape):
  '''
  '''

  # Change color unifiedly
  image = imbalance(image=image)

  # Contrast
  image = contrast(image=image)

  # Saturate
  image = saturate(image=image)

  # Shine
  image = shine(image=image, ishape=ishape)

  # Lose feature
  image = blur(image=image, ishape=ishape)

  # Grey down
  image = np.clip(image - randint(0, 64), 0, 255)

  # Bright up
  image = np.clip(image + randint(0, 64), 0, 255)

  return image

def flip_image_with_boxes(image, bboxes, ishape, mode):
  '''
  '''

  if mode == 1:
    image = np.fliplr(image)
    for i in range(len(bboxes)):
      x1 = bboxes[i][1]
      x2 = bboxes[i][3]
      bboxes[i][1] = ishape[1] - x2
      bboxes[i][3] = ishape[1] - x1

  elif mode == 2:
    image = np.flipud(image)
    for i in range(len(bboxes)):
      y1 = bboxes[i][0]
      y2 = bboxes[i][2]
      bboxes[i][0] = ishape[0] - y2
      bboxes[i][2] = ishape[0] - y1

  return image, bboxes

def fliplr_with_points(image, points, ishape, mode):
  '''
  '''

  if mode == 0:
    image = np.fliplr(image)
    for i in range(len(points)):
      points[i][1] = ishape[1] - points[i][1]

  return image, points

def fliplr_landmark(image, points, ishape, mode):
  '''
  '''

  if mode == 0:
    image = np.fliplr(image)
    for i in range(len(points)):
      points[i][1] = ishape[1] - points[i][1]

    t = points[0]
    points[0] = points[1]
    points[1] = t

    t = points[3]
    points[3] = points[4]
    points[4] = t

  return image, points

def rotate90_image_with_boxes(image, bboxes):
  '''
  '''

  image = np.transpose(image, axes=[1, 0, 2])
  for i in range(len(bboxes)):
    y1 = bboxes[i][0]
    x1 = bboxes[i][1]
    y2 = bboxes[i][2]
    x2 = bboxes[i][3]

    bboxes[i][0] = x1
    bboxes[i][1] = y1
    bboxes[i][2] = x2
    bboxes[i][3] = y2

  return image, bboxes

def zoom_image_with_boxes(image, bboxes, scale):
  '''
  '''

  bbox_len = len(bboxes)
  image = transform.resize(image=image, output_shape=(int(scale*image.shape[0]), int(scale*image.shape[1])))
  image = image/np.max(image)
  image = image*255
  for i in range(bbox_len):
    bboxes[i][0] *= scale
    bboxes[i][1] *= scale
    bboxes[i][2] *= scale
    bboxes[i][3] *= scale

  return image, bboxes

def randcrop(image, ishape):
  '''
  '''

  max_top_padding = image.shape[0] - ishape[0]
  max_left_padding = image.shape[1] - ishape[1]
  origin_y = randint(0, max_top_padding)
  origin_x = randint(0, max_left_padding)

  image = image[origin_y:origin_y+ishape[0], origin_x:origin_x+ishape[1], :]

  return image

def randcrop_image_with_boxes(image, bboxes, ishape):
  '''
  '''

  bbox_len = len(bboxes)
  bbox_idx = randint(0, bbox_len-1)
  y1 = int(bboxes[bbox_idx][0])
  x1 = int(bboxes[bbox_idx][1])
  y2 = int(bboxes[bbox_idx][2])
  x2 = int(bboxes[bbox_idx][3])
  h = y2 - y1
  w = x2 - x1

  crop_y1 = y1 - randint(0, max(int(ishape[0]-h), 0))
  crop_y2 = crop_y1 + ishape[0]
  if crop_y1 < 0:
    crop_y1 = 0
    crop_y2 = ishape[0]
  if crop_y2 > image.shape[0]:
    crop_y1 = image.shape[0] - ishape[0]
    crop_y2 = image.shape[0]

  crop_x1 = x1 - randint(0, max(int(ishape[1]-w), 0))
  crop_x2 = crop_x1 + ishape[1]
  if crop_x1 < 0:
    crop_x1 = 0
    crop_x2 = ishape[1]
  if crop_x2 > image.shape[1]:
    crop_x1 = image.shape[1] - ishape[1]
    crop_x2 = image.shape[1]

  crop_y1, crop_x1, crop_y2, crop_x2 = int(crop_y1), int(crop_x1), int(crop_y2), int(crop_x2)
  cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2, :]
  cropped_image = cropped_image/np.max(cropped_image)
  cropped_image *= 255

  remain_bboxes = []
  for i in range(bbox_len):
    y1 = bboxes[i][0]
    x1 = bboxes[i][1]
    y2 = bboxes[i][2]
    x2 = bboxes[i][3]
    h = y2 - y1
    w = x2 - x1
    if crop_y1 - y1 > 0.5*h:
      continue
    if crop_x1 - x1 > 0.5*w:
      continue
    if y2 - crop_y2 > 0.5*h:
      continue
    if x2 - crop_x2 > 0.5*w:
      continue

    bboxes[i][0] = y1-crop_y1
    bboxes[i][1] = x1-crop_x1
    bboxes[i][2] = y2-crop_y1
    bboxes[i][3] = x2-crop_x1

    remain_bboxes.append(bboxes[i])

  return [cropped_image, remain_bboxes]

def randcrop_with_points(image, points, ishape):
  '''
  '''

  max_top_padding = image.shape[0] - ishape[0]
  max_left_padding = image.shape[1] - ishape[1]
  origin_y = randint(0, max_top_padding)
  origin_x = randint(0, max_left_padding)

  image = image[origin_y:origin_y+ishape[0], origin_x:origin_x+ishape[1], :]

  for i in range(len(points)):
    points[i][0] -= origin_y
    points[i][1] -= origin_x

  return image, points

def place_image_with_boxes(image, position, placed_image, bboxes):
  '''
  '''

  ishape = image.shape
  pshape = placed_image.shape
  y, x = position
  top, left, bottom, right = y, x, ishape[0]-y, ishape[1]-x
  py, px = pshape[0]//2, pshape[1]//2
  ptop, pleft, pbottom, pright = pshape[0]//2, pshape[1]//2, pshape[0]-pshape[0]//2, pshape[1]-pshape[1]//2

  pcrop_y1 = max(0, py-top)
  pcrop_x1 = max(0, px-left)
  pcrop_y2 = min(pshape[0], py+bottom)
  pcrop_x2 = min(pshape[1], px+right)

  crop_y1 = max(0, y-ptop)
  crop_x1 = max(0, x-pleft)
  crop_y2 = min(ishape[0], y+pbottom)
  crop_x2 = min(ishape[1], x+pright)

  placed_image = placed_image[pcrop_y1:pcrop_y2, pcrop_x1:pcrop_x2, :]

  bbox_len = len(bboxes)
  remain_bboxes = []
  for i in range(bbox_len):
    y1 = bboxes[i][0]
    x1 = bboxes[i][1]
    y2 = bboxes[i][2]
    x2 = bboxes[i][3]
    h = y2 - y1
    w = x2 - x1
    if pcrop_y1 - y1 > 0.5*h:
      continue
    if pcrop_x1 - x1 > 0.5*w:
      continue
    if y2 - pcrop_y2 > 0.5*h:
      continue
    if x2 - pcrop_x2 > 0.5*w:
      continue

    bboxes[i][0] = y1-pcrop_y1+crop_y1
    bboxes[i][1] = x1-pcrop_x1+crop_x1
    bboxes[i][2] = y2-pcrop_y1+crop_y1
    bboxes[i][3] = x2-pcrop_x1+crop_x1

    remain_bboxes.append(bboxes[i])

  image[crop_y1:crop_y2, crop_x1:crop_x2, :] += placed_image

  return image, remain_bboxes

def nest_image_with_boxes(images, anno, ishape):
  '''
  '''

  scales = np.array([2.0, 1.0, 0.5, 0.25])
  positions = np.array([
    [int(ishape[0]/2), int(ishape[1]/2)],
    [int(ishape[0]/2), int(ishape[1]/2)],
    [randint(ishape[0]/4, ishape[0]-ishape[0]/4), randint(ishape[1]/4, ishape[1]-ishape[1]/4)],
    [randint(ishape[0]/8, ishape[0]-ishape[0]/8), randint(ishape[1]/8, ishape[1]-ishape[1]/8)],
  ])
  rindices = [0, 1, 2, 3]
  np.random.shuffle(rindices)
  rindices = rindices[:2]

  scales = scales[rindices]
  positions = positions[rindices]

  image = np.zeros((ishape[0], ishape[1], 3), dtype='float32')
  bboxes = []

  for i in range(len(positions)):
    zoom_image, zoom_bboxes = zoom_image_with_boxes(image=images[i], bboxes=anno[i], scale=scales[i])
    if scales[i] > 1:
      crop_image, crop_bboxes = randcrop_image_with_boxes(image=zoom_image, bboxes=zoom_bboxes, ishape=ishape)
    else:
      crop_image, crop_bboxes = randcrop_image_with_boxes(image=zoom_image, bboxes=zoom_bboxes, ishape=[scales[i]*ishape[0], scales[i]*ishape[1]])
    flip_image, flip_bboxes = flip_image_with_boxes(image=crop_image, bboxes=crop_bboxes, ishape=crop_image.shape[:2], mode=randint(0, 2))
    image, _bboxes = place_image_with_boxes(image=image, position=positions[i], placed_image=flip_image, bboxes=flip_bboxes)
    bboxes += _bboxes

  image = image/np.max(image)
  image = image*255
  
  return image, bboxes

def patch_image_with_boxes(images, anno, ishape):
  '''
  '''

  scales = [1.7, 1.0, 0.5, 0.25]
  np.random.shuffle(scales)
  positions = [
    [int(ishape[0]/4), int(ishape[1]/4)], 
    [int(ishape[0]/4), int(ishape[1]-ishape[1]/4)],
    [int(ishape[0]-ishape[0]/4), int(ishape[1]/4)], 
    [int(ishape[0]-ishape[0]/4), int(ishape[1]-ishape[1]/4)],
  ]
  image = np.zeros((ishape[0], ishape[1], 3), dtype='float32')
  bboxes = []

  for i in range(len(positions)):
    zoom_image, zoom_bboxes = zoom_image_with_boxes(image=images[i], bboxes=anno[i], scale=scales[i])
    crop_image, crop_bboxes = randcrop_image_with_boxes(image=zoom_image, bboxes=zoom_bboxes, ishape=[ishape[0]//2, ishape[1]//2])
    flip_image, flip_bboxes = flip_image_with_boxes(image=crop_image, bboxes=crop_bboxes, ishape=crop_image.shape[:2], mode=randint(0, 2))
    image, _bboxes = place_image_with_boxes(image=image, position=positions[i], placed_image=flip_image, bboxes=flip_bboxes)
    bboxes += _bboxes

  image = image/np.max(image)
  image = image*255
  
  return image, bboxes

def create_image_with_boxes(images, anno, ishape, mode):
  '''
  '''

  if mode == 0:
    image, bboxes = patch_image_with_boxes(images, anno, ishape)
  elif mode == 1:
    image, bboxes = nest_image_with_boxes(images, anno, ishape)
  elif mode == 2:
    image, bboxes = patch_image_with_boxes(images, anno, ishape)
    image = augcolor(image=image, ishape=ishape)
  elif mode == 3:
    image, bboxes = nest_image_with_boxes(images, anno, ishape)
    image = augcolor(image=image, ishape=ishape)

  # if randint(0, 1) == 0: # This doesn't work when height & width are not same
  #   image, bboxes = rotate90_image_with_boxes(image=image, bboxes=bboxes, ishape=ishape)

  if randint(0, 1) == 1: # BW
    image = np.mean(image, axis=-1, keepdims=True)
    image = np.concatenate([image, image, image], axis=-1)

  return image, bboxes

def gentiery(bboxes, abox_2dtensor, iou_thresholds, total_classes, anchor_sampling):
  '''
  '''

  bbox_2dtensor = tf.constant(value=bboxes, dtype='float32') # (total_bboxes, 5)
  bbox_2dtensor = bbox_2dtensor[:, :4] # (total_bboxes, 4)
  bbox_3dtensor = tf.repeat(input=[bbox_2dtensor], repeats=[abox_2dtensor.shape[0]], axis=0) # (h*w*k, total_bboxes, 4)
  bbox_3dtensor = tf.transpose(a=bbox_3dtensor, perm=[1, 0, 2]) # (total_bboxes, h*w*k, 4)

  iou3d = np.zeros((bbox_2dtensor.shape[0], abox_2dtensor.shape[0], 1), dtype='float32') # (total_bboxes, h*w*k, 1)
  for i in range(bbox_3dtensor.shape[0]):
    iou_2dtensor = comiou2d(abox_2dtensor=abox_2dtensor, bbox_2dtensor=bbox_3dtensor[i]) # (h*w*k, 1)
    iou3d[i] = iou_2dtensor

  iou_3dtensor = tf.constant(value=iou3d, dtype='float32') # (total_bboxes, h*w*k, 1)
  iou_2dtensor = tf.squeeze(input=iou_3dtensor, axis=-1) # (total_bboxes, h*w*k)
  iou_2dtensor = tf.transpose(a=iou_2dtensor, perm=[1, 0]) # (h*w*k, total_bboxes)
  anchor_iou_1dtensor = tf.math.reduce_max(input_tensor=iou_2dtensor, axis=-1) # (h*w*k,)
  anchor_type_1dtensor = tf.math.argmax(input=iou_2dtensor, axis=-1) # (h*w*k,)

  # Assign positives, neutral, negatives, zero negatives
  anchor_type_1dtensor = tf.where(
    condition=tf.math.greater_equal(x=anchor_iou_1dtensor, y=iou_thresholds[1]),
    x=anchor_type_1dtensor,
    y=len(bboxes)+1) # (h*w*k,)
  anchor_type_1dtensor = tf.where(
    condition=tf.math.logical_and(
      x=tf.math.less(x=anchor_iou_1dtensor, y=iou_thresholds[1]),
      y=tf.math.greater(x=anchor_iou_1dtensor, y=iou_thresholds[0])),
    x=len(bboxes)+2,
    y=anchor_type_1dtensor) # (h*w*k,)
  anchor_type_1dtensor = tf.where(
    condition=tf.math.logical_and(
      x=tf.math.less_equal(x=anchor_iou_1dtensor, y=iou_thresholds[0]),
      y=tf.math.greater(x=anchor_iou_1dtensor, y=0)),
    x=len(bboxes),
    y=anchor_type_1dtensor) # (h*w*k,)

  # Sample anchors, pad or remove
  anchor_type1d = np.array(anchor_type_1dtensor)
  pos_indices, = np.where(anchor_type1d < len(bboxes))
  neg_indices, = np.where(anchor_type1d == len(bboxes))
  zero_neg_indices, = np.where(anchor_type1d == len(bboxes)+1)

  total_fg = anchor_sampling//2
  total_bg = anchor_sampling - total_fg
  total_pos = len(pos_indices)
  total_neg = len(neg_indices)
  total_zero_neg = len(zero_neg_indices)
  
  no_match_anchors = False
  not_enough_neg_anchors = False

  if total_pos == 0:
    # print('!', end='') # No match anchors
    no_match_anchors = True

  if total_pos > total_fg:
    np.random.shuffle(pos_indices)
    remove_pos_indices = pos_indices[total_fg:]
    anchor_type1d[remove_pos_indices] = len(bboxes)+2

  else:
    total_fg = total_pos
    total_bg = anchor_sampling - total_pos

  total_selected_neg = total_neg
  total_selected_zero_neg = total_zero_neg

  if total_neg > total_bg//2:
    total_selected_zero_neg = min(total_bg//2, total_zero_neg)
    total_selected_neg = min(total_bg - total_selected_zero_neg, total_neg)

  if total_zero_neg > total_bg//2:
    total_selected_neg = min(total_bg//2, total_neg)
    total_selected_zero_neg = min(total_bg - total_selected_neg, total_zero_neg)

  if total_selected_neg + total_selected_zero_neg < total_bg:
    # print('<', end='') # Not enough negative anchors
    not_enough_neg_anchors = True

  # if no_match_anchors is False:
  #   print(total_fg, total_bg)
    
  np.random.shuffle(neg_indices)
  anchor_type1d[neg_indices[total_selected_neg:]] = len(bboxes)+2

  np.random.shuffle(zero_neg_indices)
  anchor_type1d[zero_neg_indices[:total_selected_zero_neg]] = len(bboxes)

  # Compute loc error
  bboxes.append([0, 0, 0, 0, total_classes])
  bboxes.append([0, 0, 0, 0, total_classes+1])
  bboxes.append([0, 0, 0, 0, total_classes+1])
  matching_2dtensor = tf.gather(params=bboxes, indices=anchor_type1d) # (h*w*k, 5)
  matching_2dtensor = tf.cast(x=matching_2dtensor, dtype='float32')
  matching_bbox_2dtensor = matching_2dtensor[:, :4] # (h*w*k, 4)
  matching_cat_1dtensor = tf.cast(x=matching_2dtensor[:, 4], dtype='int64') # (h*w*k,)
  clz_2dtensor = tf.one_hot(indices=matching_cat_1dtensor, depth=total_classes+1, axis=-1) # (h*w*k, total_classes+1)
  loc_2dtensor = comloc2d(bbox_2dtensor=matching_bbox_2dtensor, abox_2dtensor=abox_2dtensor) # (h*w*k, 4)

  del bboxes[-1]
  del bboxes[-1]
  del bboxes[-1]

  return clz_2dtensor, loc_2dtensor, no_match_anchors, not_enough_neg_anchors

def genxy_od(dataset, image_dir, ishape, abox_2dtensor, iou_thresholds, total_examples, total_classes, anchor_sampling, scale_range, aug=[1, 2, 3, 4, 5]):
  '''
  '''

  total_examples += 100 # Guess 100 no_match_anchors
  for i in range(total_examples):
    image_id, bboxes = dataset[np.random.randint(0, len(dataset)-1)]
    bboxes = copy.deepcopy(bboxes)
    image = io.imread(image_dir + '/' + image_id + '.jpg')
    assert len(image.shape) == 3, 'Image shape must be 3 axes'
    assert image.shape[2] == 3, 'Require RBG image'

    aug_idx = randint(0, len(aug)-1)

    if aug[aug_idx] == 1:
      image, bboxes = rotate90_image_with_boxes(image=image, bboxes=bboxes)

    image, bboxes = zoom_image_with_boxes(image=image, bboxes=bboxes, scale=randint(scale_range[0], scale_range[1])/1000)
    image, bboxes = randcrop_image_with_boxes(image=image, bboxes=bboxes, ishape=ishape)

    if aug[aug_idx] == 2:
      image, bboxes = flip_image_with_boxes(image=image, bboxes=bboxes, ishape=ishape, mode=1)

    if aug[aug_idx] == 3:
      image, bboxes = flip_image_with_boxes(image=image, bboxes=bboxes, ishape=ishape, mode=2)

    if aug[aug_idx] == 4:
      image = augcolor(image=image, ishape=ishape)

    if aug[aug_idx] == 5:
      image = np.mean(image, axis=-1, keepdims=True)
      image = np.concatenate([image, image, image], axis=-1)

    clz_2dtensor, loc_2dtensor, no_match_anchors, _ = gentiery(
      bboxes=bboxes, 
      abox_2dtensor=abox_2dtensor, 
      iou_thresholds=iou_thresholds, 
      total_classes=total_classes, 
      anchor_sampling=anchor_sampling)

    if no_match_anchors:
      print('!', end='')
      continue

    batchy_2dtensor = tf.concat(values=[clz_2dtensor, loc_2dtensor], axis=-1) # (h*w*k, total_classes+1+4)
    batchy_3dtensor = tf.expand_dims(input=batchy_2dtensor, axis=0)
    batchx_4dtensor = tf.constant(value=[image], dtype='float32')

    yield batchx_4dtensor, batchy_3dtensor, bboxes

def genxy_mod(dataset, image_dir, ishape, abox_2dtensors, iou_thresholds, total_examples, total_classes, anchor_sampling, scale_range, aug=[1, 2, 3, 4, 5]):
  '''
  '''

  total_examples += 100 # Guess 100 no_match_anchors
  for i in range(total_examples):
    image_id, bboxes = dataset[np.random.randint(0, len(dataset)-1)]
    bboxes = copy.deepcopy(bboxes)
    image = io.imread(image_dir + '/' + image_id + '.jpg')
    assert len(image.shape) == 3, 'Image shape must be 3 axes'
    assert image.shape[2] == 3, 'Require RBG image'

    aug_idx = randint(0, len(aug)-1)

    if aug[aug_idx] == 1:
      image, bboxes = rotate90_image_with_boxes(image=image, bboxes=bboxes)

    image, bboxes = zoom_image_with_boxes(image=image, bboxes=bboxes, scale=randint(scale_range[0], scale_range[1])/1000)
    image, bboxes = randcrop_image_with_boxes(image=image, bboxes=bboxes, ishape=ishape)

    if aug[aug_idx] == 2:
      image, bboxes = flip_image_with_boxes(image=image, bboxes=bboxes, ishape=ishape, mode=1)

    if aug[aug_idx] == 3:
      image, bboxes = flip_image_with_boxes(image=image, bboxes=bboxes, ishape=ishape, mode=2)

    if aug[aug_idx] == 4:
      image = augcolor(image=image, ishape=ishape)

    if aug[aug_idx] == 5:
      image = np.mean(image, axis=-1, keepdims=True)
      image = np.concatenate([image, image, image], axis=-1)

    y_tiers = []
    has_match_anchors = False
    for i in range(len(abox_2dtensors)):
      tier_clz_2dtensor, tier_loc_2dtensor, no_match_anchors, _ = gentiery(
        bboxes=bboxes, 
        abox_2dtensor=abox_2dtensors[i], 
        iou_thresholds=iou_thresholds[i], 
        total_classes=total_classes, 
        anchor_sampling=anchor_sampling[i])

      if no_match_anchors is True:
        tier_clz_2dtensor *= 0

      y_tier = tf.concat(values=[tier_clz_2dtensor, tier_loc_2dtensor], axis=-1) # (h*w*k, total_classes+1+4)
      y_tiers.append(y_tier)
      has_match_anchors |= not no_match_anchors

    if has_match_anchors is not True:
      print('!', end='')
      continue

    batchy_2dtensor = tf.concat(values=y_tiers, axis=0) # (h1*w1*k1 + h2*w2*k2 + ... + hn*wn*kn, total_classes+1+4)
    batchy_3dtensor = tf.expand_dims(input=batchy_2dtensor, axis=0)
    batchx_4dtensor = tf.constant(value=[image], dtype='float32')

    yield batchx_4dtensor, batchy_3dtensor, bboxes

def genxy_od4(dataset, image_dir, ishape, abox_2dtensors, iou_thresholds, total_examples, total_classes, anchor_sampling):
  '''
  '''

  total_examples += 100 # Guess 100 no_match_anchors
  total_patches                 = total_examples//2
  total_patches_w_augcolor          = total_examples//4
  total_nests                 = total_examples//8
  total_nests_w_augcolor            = total_examples - (total_patches+total_patches_w_augcolor+total_nests)
  modes = np.concatenate([np.zeros(total_patches), np.ones(total_nests), 2*np.ones(total_patches_w_augcolor), 3*np.ones(total_nests_w_augcolor)], axis=-1)
  np.random.shuffle(modes)
  
  for _ in range(total_examples):
    four_images = []
    four_bboxes = []
    
    for _ in range(4):
      image_id, bboxes = dataset[randint(0, len(dataset)-1)]
      bboxes = copy.deepcopy(bboxes)
      image = io.imread(image_dir + '/' + image_id + '.jpg')
      assert len(image.shape) == 3, 'Image shape must be 3 axes'
      assert image.shape[2] == 3, 'Require RBG image'
      four_images.append(image)
      four_bboxes.append(bboxes)

    image, bboxes = create_image_with_boxes(images=four_images, anno=four_bboxes, ishape=ishape, mode=modes[i])

    tire1_clz_2dtensor, tire1_loc_2dtensor, no_match_anchors1, _ = gentiery(
      bboxes=bboxes, 
      abox_2dtensor=abox_2dtensors[0], 
      iou_thresholds=iou_thresholds[0], 
      total_classes=total_classes, 
      anchor_sampling=anchor_sampling[0])
    tire2_clz_2dtensor, tire2_loc_2dtensor, no_match_anchors2, _ = gentiery(
      bboxes=bboxes, 
      abox_2dtensor=abox_2dtensors[1], 
      iou_thresholds=iou_thresholds[1], 
      total_classes=total_classes, 
      anchor_sampling=anchor_sampling[1])
    tire3_clz_2dtensor, tire3_loc_2dtensor, no_match_anchors3, _ = gentiery(
      bboxes=bboxes, 
      abox_2dtensor=abox_2dtensors[2], 
      iou_thresholds=iou_thresholds[2], 
      total_classes=total_classes, 
      anchor_sampling=anchor_sampling[2])
    tire4_clz_2dtensor, tire4_loc_2dtensor, no_match_anchors4, _ = gentiery(
      bboxes=bboxes, 
      abox_2dtensor=abox_2dtensors[3], 
      iou_thresholds=iou_thresholds[3], 
      total_classes=total_classes, 
      anchor_sampling=anchor_sampling[3])

    if no_match_anchors1 is True and no_match_anchors2 is True and no_match_anchors3 is True and no_match_anchors4 is True:
      print('!', end='')
      continue

    if no_match_anchors1 is True:
      tire1_clz_2dtensor *= 0
    if no_match_anchors2 is True:
      tire2_clz_2dtensor *= 0
    if no_match_anchors3 is True:
      tire3_clz_2dtensor *= 0
    if no_match_anchors4 is True:
      tire4_clz_2dtensor *= 0

    tier1_y = tf.concat(values=[tire1_clz_2dtensor, tire1_loc_2dtensor], axis=-1) # (h1*w1*k1, total_classes+1+4)
    tier2_y = tf.concat(values=[tire2_clz_2dtensor, tire2_loc_2dtensor], axis=-1) # (h2*w2*k2, total_classes+1+4)
    tier3_y = tf.concat(values=[tire3_clz_2dtensor, tire3_loc_2dtensor], axis=-1) # (h3*w3*k3, total_classes+1+4)
    tier4_y = tf.concat(values=[tire4_clz_2dtensor, tire4_loc_2dtensor], axis=-1) # (h4*w4*k4, total_classes+1+4)

    batchy_2dtensor = tf.concat(values=[tier1_y, tier2_y, tier3_y, tier4_y], axis=0) # (h1*w1*k1 + h2*w2*k2 + h3*w3*k3 + h4*w4*k4, total_classes+1+4)
    batchy_3dtensor = tf.expand_dims(input=batchy_2dtensor, axis=0)
    batchx_4dtensor = tf.constant(value=[image], dtype='float32')

    yield batchx_4dtensor, batchy_3dtensor, bboxes

def get_dataset_info(dataset_name):
  '''
  '''

  dataset_info_list = {
    'mnist-digits': {
      'total_classes': 10,
      'train_anno_file_path': 'mnist-digits/train.txt',
      'train_image_dir_path': 'mnist-digits/train',
      'test_anno_file_path': 'mnist-digits/test.txt',
      'test_image_dir_path': 'mnist-digits/test',
    },
    'fingers': {
      'total_classes': 6,
      'train_anno_file_path': 'fingers/train.txt',
      'train_image_dir_path': 'fingers/train',
      'test_anno_file_path': 'fingers/test.txt',
      'test_image_dir_path': 'fingers/test',
    },
    'face1024': {
      'total_classes': 1,
      'total_train_examples': 500, # 976
      'total_test_examples': 100, # 226
      'train_anno_file_path': 'face1024/train1024.txt',
      'train_image_dir_path': 'face1024/train1024',
      'test_anno_file_path': 'face1024/test1024.txt',
      'test_image_dir_path': 'face1024/test1024',
    },
    'faceali128x128': {
      'total_heatpoints': 5,
      'train_anno_file_path': 'faceali128x128/train.txt',
      'train_image_dir_path': 'faceali128x128/train',
      'test_anno_file_path': 'faceali128x128/test.txt',
      'test_image_dir_path': 'faceali128x128/test',
    },
    'faceid128x128': {
      'total_classes': 3000,
      'train_anno_file_path': 'faceid128x128/train.txt',
      'train_image_dir_path': 'faceid128x128/train',
      'test_anno_file_path': 'faceid128x128/test.txt',
      'test_image_dir_path': 'faceid128x128/test',
    },
    'quizanswer': {
      'total_classes': 2,
      'total_train_examples': 800,
      'total_test_examples': 170,
      'train_anno_file_path': 'quizanswer/train.txt',
      'train_image_dir_path': 'quizanswer/train',
      'test_anno_file_path': 'quizanswer/test.txt',
      'test_image_dir_path': 'quizanswer/test',
    },
    'chess416': {
      'total_classes': 14,
      'total_train_examples': 606,
      'total_test_examples': 29,
      'train_anno_file_path': 'chess416/train.txt',
      'train_image_dir_path': 'chess416/train',
      'test_anno_file_path': 'chess416/test.txt',
      'test_image_dir_path': 'chess416/test',
    },
  }

  dataset_info = dataset_info_list[dataset_name]

  if os.path.isdir(dataset_name) is not True:
    # Download
    os.system('pip install kaggle')
    os.environ['KAGGLE_USERNAME'] = 'baohoa'
    os.environ['KAGGLE_KEY'] = '6feaa6963fe2cfbf1b727543fab8f726'
    os.system('kaggle datasets download -d baohoa/'+dataset_name)

    # Unzip
    with zipfile.ZipFile(dataset_name+'.zip', 'r') as zip_ref:
      os.makedirs(dataset_name)
      zip_ref.extractall('./'+dataset_name+'/')
      zip_ref.close()
      os.system('rm -rf '+dataset_name+'.zip')

  return dataset_info

def load_object_detection_dataset(anno_file_path, total_classes):
  '''
  '''

  def parse_line(line):
    anno = line[:-1].split(' ')
    image_id = anno[0]
    bboxes = anno[1:]
    bboxes = [list(map(float, bboxes[i:i+5])) for i in range(0, len(bboxes), 5)]
    return image_id, bboxes

  anno_file = open(anno_file_path, 'r')
  lines = anno_file.readlines()
  total_lines = len(lines)
  # print('\nTotal lines: {}'.format(total_lines))

  dataset = []
  for i in range(total_lines):
    image_id, bboxes = parse_line(line=lines[i])
    dataset.append([image_id, bboxes])

  return dataset

def load_image_classification_dataset(anno_file_path):
  '''
  '''

  anno_file = open(anno_file_path, 'r')
  lines = anno_file.readlines()
  total_lines = len(lines)
  # print('\nTotal lines: {}'.format(total_lines))

  dataset = []
  for i in range(total_lines):
    line = lines[i]
    anno = line[:-1].split(' ')
    image_file_name, label = anno
    dataset.append([image_file_name, label])

  return dataset  

def genxy_ic(dataset, image_dir, ishape, total_classes, total_examples, batch_size):
  '''
  '''

  total_batches = total_examples//batch_size

  for i in range(total_batches):
    batchx_4dtensor = np.zeros((batch_size, ishape[0], ishape[1], ishape[2]), dtype='float32')
    batchy_2dtensor = np.zeros((batch_size, total_classes), dtype='float32')
    for j in range(batch_size):
      image_file_name, label = dataset[i*batch_size+j]
      image = io.imread(image_dir+'/'+image_file_name)
      assert len(image.shape) >= 2, 'Image shape must be 2 or 3 axes'

      if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)

      image = randcrop(image=image, ishape=ishape)

      aug = randint(0, 1)
      if aug == 1:
        image = augcolor(image=image, ishape=ishape)

      batchx_4dtensor[j] = image
      batchy_2dtensor[j][int(label)] = 1

    yield batchx_4dtensor, batchy_2dtensor

def genheatmaps(image, points, ishape):
  '''
  '''

  image = np.mean(image, axis=-1, keepdims=True)
  heatmap3d = np.zeros((ishape[0], ishape[1], len(points)), dtype='float32')
  for p in range(5):
    py, px = points[p]
    pos = np.dstack(np.mgrid[0:ishape[0]:1, 0:ishape[1]:1])
    rv = multivariate_normal(mean=[py, px], cov=32)
    heatmap2d = rv.pdf(pos)
    heatmap2d /= np.max(heatmap2d)
    heatmap2d *= 255
    heatmap2d += image[:, :, 0]
    heatmap2d /= np.max(heatmap2d)
    heatmap3d[:, :, p] = heatmap2d

  return heatmap3d

def load_heatmap_regression_dataset(anno_file_path, total_heatpoints):
  '''
  '''

  def parse_line(line):
    anno = line[:-1].split(' ')
    image_id = anno[0]
    points = anno[1:]
    points = [list(map(int, points[i:i+2])) for i in range(0, len(points), 2)]
    return image_id, points

  anno_file = open(anno_file_path, 'r')
  lines = anno_file.readlines()
  total_lines = len(lines)
  # print('\nTotal lines: {}'.format(total_lines))

  dataset = []
  for i in range(total_lines):
    image_id, points = parse_line(line=lines[i])
    dataset.append([image_id, points])

  return dataset

def genxy_hmr(dataset, image_dir_path, ishape, total_examples, batch_size):
  '''
  '''

  total_batches = total_examples//batch_size
  for i in range(total_batches):
    batchx4d = np.zeros((batch_size, ishape[0], ishape[1], ishape[2]), dtype='float32')
    batchy4d = np.zeros((batch_size, ishape[0], ishape[1], 5), dtype='float32')

    for j in range(batch_size):
      image_file_name, points = dataset[i*batch_size+j]
      points = copy.deepcopy(points)
      image = io.imread(image_dir_path+'/'+image_file_name)
      assert len(image.shape) == 3, 'Image shape must be 3 axes'

      image = np.array(image, dtype='float32')
      image, points = randcrop_with_points(image=image, points=points, ishape=ishape)

      aug = randint(0, 2)

      if aug == 1:
        image = augcolor(image=image, ishape=ishape)

      if aug == 2:
        image, points = fliplr_landmark(image=image, points=points, ishape=ishape, mode=randint(0, 1))

      if ishape[2] == 1:
        image = np.mean(image, axis=-1, keepdims=True)

      heatmap3d = genheatmaps(image=image, points=points, ishape=ishape)

      batchx4d[j, :, :, :] = image
      batchy4d[j] = heatmap3d

    yield batchx4d, batchy4d
