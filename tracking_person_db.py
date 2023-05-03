# python interpreter searches these subdirectories for modules
import sys

from sort.sort import Sort

sys.path.insert(0, './yolov5')
sys.path.insert(0, './sort')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

# yolov5
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_file, \
    check_requirements, print_args, check_imshow, increment_path, LOGGER, colorstr, strip_optimizer
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator

# SORT
import skimage
from sort import *

# Predict
from predict_person_congestion import calculate_congestion

torch.set_printoptions(precision=3)

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def draw_boxes(bbox, identities=None, categories=None, summary_sum=None):
    for i, box in enumerate(bbox):
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        if cat == 0:
            summary_sum.append(id)

    return summary_sum


@torch.no_grad()
def run(
        weights='yolov5/yolov5s.pt',  # model.pt path(s)
        source='yolov5/data/images',  # file/dir/URL/glob, 0 for webcam
        data='yolov5/data/coco128_person.yaml',  # customDataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        dnn=False,  # use OpenCV DNN for ONNX inference

        sort_max_age=30,
        sort_min_hits=2,
        sort_iou_thresh=0.2,

        sum_time=4.0,
        filming_location='MainGate'
):
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Initialize SORT
    sort_tracker = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)  # {plug into parser}

    # Directory and CUDA settings for YOLOv5
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load YOLOv5 model
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Set Dataloader
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size

    # Init define
    summary_data = []
    summary_time = 0.0
    summary_frame = 0

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im, augment=augment, visualize=visualize)

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            s += '%gx%g ' % im.shape[2:]  # print string

            # Rescale boxes from img_size (temporarily downscaled size) to im0 (native) size
            det[:, :4] = scale_coords(
                im.shape[2:], det[:, :4], im0.shape).round()

            for c in det[:, -1].unique():  # for each unique object category
                n = (det[:, -1] == c).sum()  # number of detections per class
                s += f' - {n} {names[int(c)]}'

            dets_to_sort = np.empty((0, 6))

            # Pass detections to SORT
            # NOTE: We send in detected object class too
            for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))

            # Run SORT
            tracked_dets = sort_tracker.update(dets_to_sort)

            # draw boxes for visualization
            if len(tracked_dets) > 0:
                bbox_xyxy = tracked_dets[:, :4]
                identities = tracked_dets[:, 8]
                categories = tracked_dets[:, 4]
                summary_data = draw_boxes(bbox_xyxy, identities, categories, summary_data)

        # During time
        summary_time += time_sync() - t1
        summary_frame += 1
        if summary_time >= sum_time:
            predict_congestion, predict_person = calculate_congestion(summary_data, summary_frame, filming_location)
            print('Congestion Level: {},\t Waiting Person: {}'.format(predict_congestion, predict_person))
            summary_data = []
            summary_time = 0.0
            summary_frame = 0


def parse_opt():
    parser = argparse.ArgumentParser()

    # YOLOv5 params
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5/yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='yolov5/data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default='yolov5/data/coco128_person.yaml',
                        help='(optional) customDataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')

    # SORT params
    parser.add_argument('--sort-max-age', type=int, default=5,
                        help='keep track of object even if object is occluded or not detected in n frames')
    parser.add_argument('--sort-min-hits', type=int, default=2,
                        help='start tracking only after n number of objects detected')
    parser.add_argument('--sort-iou-thresh', type=float, default=0.1,
                        help='intersection-over-union threshold between two frames for association')

    # Detecting descript
    parser.add_argument('--sum-time', type=float, default=5.0, help='Designated as 4 seconds based on the image of '
                                                                    'FPS 30.')
    parser.add_argument('--filming-location', type=str, default='MainGate',
                        help='Congestion level filming location is \'MainGate\' or \'AI\'')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
