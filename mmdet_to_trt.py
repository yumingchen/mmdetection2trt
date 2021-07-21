from argparse import ArgumentParser
import cv2
import torch
from mmdet2trt import mmdet2trt
from mmdet2trt.apis import inference_detector, init_detector


def main():
    # parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    # parser.add_argument('config', help='mmdet Config file')
    # parser.add_argument('checkpoint', help='mmdet Checkpoint file')
    # parser.add_argument('save_path', help='tensorrt model save path')
    # parser.add_argument(
    #     '--device', default='cuda:0', help='Device used for inference')
    # parser.add_argument(
    #     '--score-thr', type=float, default=0.3, help='bbox score threshold')
    # parser.add_argument(
    #     '--fp16', action='store_true', help='enable fp16 inference')
    # args = parser.parse_args()
    #
    # cfg_path = args.config
    # save_path保存用于mmdetection-to-tensorrt推理的模型，对于本项目，可以不用。
    save_path = '/home/cym/CYM/output/cascade_rcnn/epoch_37_trt.pth'
    checkpoint = '/home/cym/CYM/output/cascade_rcnn/epoch_37.pth'
    cfg_path = '/home/cym/CYM/project/mmdetection/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco_comp2.py'
    image_path = '/home/cym/CYM/dataset/comp2_coco/images/0a4e3907_c187_4142_84a2_4c270c899c8b.JPG'
    opt_shape_param = [
        [
            [1, 3, 800, 800],  # min shape
            [1, 3, 800, 1333],  # optimize shape
            [1, 3, 1333, 1333],  # max shape
        ]
    ]
    # opt_shape_param = None
    trt_model = mmdet2trt(
        cfg_path, checkpoint, opt_shape_param=opt_shape_param, fp16_mode=True, device='cuda:0')
    torch.save(trt_model.state_dict(), save_path)

    trt_model = init_detector(save_path)

    with open(checkpoint.replace('.','_')+'.engine', 'wb') as f:
        f.write(trt_model.engine.serialize())

    result = inference_detector(trt_model, image_path, cfg_path, 'cuda:0')

    num_detections = result[0].item()
    trt_bbox = result[1][0]
    trt_score = result[2][0]
    trt_cls = result[3][0]

    image = cv2.imread(image_path)
    input_image_shape = image.shape
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
        cv2.rectangle(image, bbox[:2], bbox[2:], color, thickness=5)

    if input_image_shape[0] > 1280 or input_image_shape[1] > 720:
        scales = min(720 / image.shape[0], 1280 / image.shape[1])
        image = cv2.resize(image, (0, 0), fx=scales, fy=scales)
    cv2.imshow('image', image)
    cv2.waitKey()


if __name__ == '__main__':
    main()