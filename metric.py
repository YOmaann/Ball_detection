import tensorflow as tf

def iou_loss(y_true, y_pred):
    epsilon = 1e-9

    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)

    # Intersection rectangle
    xA = tf.maximum(y_true[:, 0], y_pred[:, 0])
    yA = tf.maximum(y_true[:, 1], y_pred[:, 1])
    xB = tf.minimum(y_true[:, 2], y_pred[:, 2])
    yB = tf.minimum(y_true[:, 3], y_pred[:, 3])
    
    # Intersection area
    inter_area = tf.maximum(0.0, xB - xA) * tf.maximum(0.0, yB - yA)
    
    # Areas of boxes
    box_true_area = (y_true[:, 2] - y_true[:, 0]) * (y_true[:, 3] - y_true[:, 1])
    box_pred_area = (y_pred[:, 2] - y_pred[:, 0]) * (y_pred[:, 3] - y_pred[:, 1])
    
    # Union area
    union_area = box_true_area + box_pred_area - inter_area + epsilon
    
    # IoU
    iou = inter_area / union_area


    # tf.print("IoU_mean:", tf.reduce_mean(iou), 
    #          "Center_dist_mean:", tf.reduce_mean(center_dist),
    #          "Diou_mean:", tf.reduce_mean(diou),
    #          "Loss_mean:", tf.reduce_mean(loss))

    return iou

def l1_loss(y_true, y_pred):
     return  tf.reduce_mean(tf.abs(y_true - y_pred))
        

def diou_loss(y_true, y_pred):
    # Calculate centers
    pred_cx = (y_pred[:, 2] + y_pred[:, 0]) / 2
    pred_cy = (y_pred[:, 3] + y_pred[:, 1]) / 2
    true_cx = (y_true[:, 2] + y_true[:, 0]) / 2
    true_cy = (y_true[:, 3] + y_true[:, 1]) / 2

    # Return squared distance (modified)
    # check = tf.cast(inter_area <= 1e9, tf.float32)
    # return check * (tf.sqrt(tf.square(true_cy - pred_cy) + tf.square(true_cx - pred_cx)) + 2) + (1 - check) * (1 - iou)

    # return 1 - iou * 1000 + 1000 * tf.square(true_cy - pred_cy) + 1000 * tf.square(true_cx - pred_cx)

    # Diou approach - This loss function is the smoother version of whatever i was trying.

    center_dist = tf.square(true_cx - pred_cx) + tf.square(true_cy - pred_cy)

    # Smallest enclosing box
    enclose_x1 = tf.minimum(y_true[:, 0], y_pred[:, 0])
    enclose_y1 = tf.minimum(y_true[:, 1], y_pred[:, 1])
    enclose_x2 = tf.maximum(y_true[:, 2], y_pred[:, 2])
    enclose_y2 = tf.maximum(y_true[:, 3], y_pred[:, 3])

    enclose_diag = tf.square(enclose_x2 - enclose_x1) + tf.square(enclose_y2 - enclose_y1) + epsilon
    
    iou = iou_loss(y_true, y_pred)
    # DIoU loss
    diou = iou - (center_dist / enclose_diag)
    loss = 1 - diou

    return loss
