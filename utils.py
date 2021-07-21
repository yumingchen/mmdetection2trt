import cv2
import os
import numpy as np
import time
import datetime


def preprocess_image(image_path, input_size=(1333, 800)):
    """
    description: Read an image from image_raw, convert it to RGB,
                 resize and pad it to target size, normalize to [0,1],
                 transform to NCHW format.
    param:
        image_path: str, the image path.
    return:
        image:  the processed image, ndarray, shape is (batch_size, channel, height, width), batch_size = 1.
        image_raw: the original image, ndarray, shape is (height, width, channel)
        h: original height
        w: original width
    """
    image_raw = cv2.imread(image_path)
    origin_h, origin_w, c = image_raw.shape
    if origin_h > origin_w:
        input_w, input_h = 800, 1333
    else:
        input_w, input_h = 1333, 800
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    # Calculate widht and height and paddings
    ratio_w = input_w / origin_w
    ratio_h = input_h / origin_h
    if ratio_h > ratio_w:
        tw = input_w
        th = int(ratio_w * origin_h)
        tx1 = tx2 = 0
        ty1 = int((input_h - th) / 2)
        ty2 = input_h - th - ty1
    else:
        tw = int(ratio_h * origin_w)
        th = input_h
        tx1 = int((input_w - tw) / 2)
        tx2 = input_w - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128,128,128)
    )
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.astype(np.float32)
    # Normalize to [0,1]
    image /= 255.0
    # Normalize to [-1,1]
    def normalization(image, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
        mean = np.array(mean) / 255.0
        std = np.array(std) / 255.0
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        return image
    image = normalization(image)

    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])
    # CHW to NCHW format
    image = np.expand_dims(image, axis=0)
    # Convert the image to row-major order, also known as "C order":
    image = np.ascontiguousarray(image)
    return image, image_raw, origin_h, origin_w, input_h, input_w, image_bgr


def postprocess_image(result, image_raw, origin_h, origin_w, input_h, input_w, image_bgr):

    cls_names = ['badge', 'right', 'wrong', 'no']
    ratio_w, ratio_h = float(input_w/origin_w), float(input_h/origin_h)
    # bbox_origin = result[1].copy()
    # 减掉黑边
    if ratio_h > ratio_w:
        border_size = (input_h-(origin_h*ratio_w))/2
        result[1][0][:, 1] -= border_size
        result[1][0][:, 3] -= border_size
    else:
        border_size = (input_w - (origin_w * ratio_h)) / 2
        result[1][0][:, 0] -= border_size
        result[1][0][:, 2] -= border_size

    scale_factor = min(ratio_w, ratio_h)
    result[1] = result[1] / scale_factor

    num_detections = result[0].item()
    trt_bbox = result[1][0]
    trt_score = result[2][0]
    trt_cls = result[3][0]

    print('the number of bbox: [{}]'.format(num_detections))

    input_image_shape = image_raw.shape
    for i in range(num_detections):
        scores = trt_score[i].item()
        classes = int(trt_cls[i].item())
        if scores < 0.3:
            continue
        bbox = tuple(trt_bbox[i])
        bbox = tuple(int(v) for v in bbox)

        color = ((classes >> 2 & 1) * 128 + (classes >> 5 & 1) * 128,
                 (classes >> 1 & 1) * 128 + (classes >> 4 & 1) * 128,
                 (classes >> 0 & 1) * 128 + (classes >> 3 & 1) * 128)
        cv2.rectangle(image_raw, bbox[:2], bbox[2:], color, thickness=5)
        label = "{}:{:.3f}".format(cls_names[classes], scores)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(image_raw, label,(bbox[0], bbox[1] - 5), font, 1.0, color=color, thickness=2, lineType=cv2.LINE_AA)
    # timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # cv2.imwrite('detection/'+timestamp+'.png', image_raw)
    if input_image_shape[0] > 1280 or input_image_shape[1] > 720:
        scales = min(720 / image_raw.shape[0], 1280 / image_raw.shape[1])
        image_raw = cv2.resize(image_raw, (0, 0), fx=scales, fy=scales)

    cv2.imshow('image', image_raw)
    cv2.waitKey()


def xywh2xyxy(origin_h, origin_w, bbox, input_h, input_w):
    """
    description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    param:
        origin_h:   height of original image
        origin_w:   width of original image
        bbox:          A boxes tensor, each row is a box [center_x, center_y, w, h]
    return:
        y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
    """
    # y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    bbox_xyxy = np.zeros_like(bbox)
    ratio_w = input_w / origin_w
    ratio_h = input_h / origin_h
    if ratio_h > ratio_w:
        bbox_xyxy[:, 0] = bbox[:, 0] - bbox[:, 2] / 2
        bbox_xyxy[:, 2] = bbox[:, 0] + bbox[:, 2] / 2
        bbox_xyxy[:, 1] = bbox[:, 1] - bbox[:, 3] / 2 - (input_h - ratio_w * origin_h) / 2
        bbox_xyxy[:, 3] = bbox[:, 1] + bbox[:, 3] / 2 - (input_h - ratio_w * origin_h) / 2
        bbox_xyxy /= ratio_w
    else:
        bbox_xyxy[:, 0] = bbox[:, 0] - bbox[:, 2] / 2 - (input_w - ratio_h * origin_w) / 2
        bbox_xyxy[:, 2] = bbox[:, 0] + bbox[:, 2] / 2 - (input_w - ratio_h * origin_w) / 2
        bbox_xyxy[:, 1] = bbox[:, 1] - bbox[:, 3] / 2
        bbox_xyxy[:, 3] = bbox[:, 1] + bbox[:, 3] / 2
        bbox_xyxy /= ratio_h
    return bbox_xyxy


def nms_np(pred, thresh):
    '''
    param pred: ndarray [N,6], eg:[xmin,ymin,xmax,ymax,score, classid]
    param thresh: float
    return keep: list[index]
    '''
    x1 = pred[:, 0]
    y1 = pred[:, 1]
    x2 = pred[:, 2]
    y2 = pred[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = pred[:, 4].argsort()[::-1]

    keep = []
    while order.shape[0] > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        over = (w * h) / (area[i] + area[order[1:]] - w * h)
        index = np.where(over <= thresh)[0]
        order = order[index + 1]
    return keep