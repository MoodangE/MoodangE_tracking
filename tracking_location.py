# python interpreter searches these subdirectories for modules
import sys

sys.path.insert(0, './yolov5')
sys.path.insert(0, './sort')

import glob
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
from predict_location_Tfid import location_predict_vector

torch.set_printoptions(precision=3)

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # MoodangE_tracking root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


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


def scene_boxes(bbox, categories=None, names=None, offset=(0, 0), summary_sum=None):
    for i, box in enumerate(bbox):
        cat = int(categories[i]) if categories is not None else 0
        summary_sum += names[cat] + ' '
    return summary_sum


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5/yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'yolov5/data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'yolov5/models/yolov5s.yaml',  # customDataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        project=ROOT / 'inference_location',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        dnn=False,  # use OpenCV DNN for ONNX inference

        sort_max_age=5,
        sort_min_hits=2,
        sort_iou_thresh=0.2,

        start_point='AI',
        sum_time=4.0
):
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    video = False

    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Initialize SORT
    sort_tracker = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)  # {plug into parser}

    # Directory and CUDA settings for YOLOv5
    device = select_device(device)
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)  # delete output folder
    os.makedirs(save_dir)  # make new output folder
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
        video = True
        bs = 1  # batch_size

    # Init define
    predict_location = start_point
    summary_data = ''
    summary_time = 0.0

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        split_s = s.split()
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
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

            p = Path(p)  # to Path
            txt_path = str(save_dir / p.stem) + '.txt'  # im.txt
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

            # Detect data savet to summay_data
            if len(tracked_dets) > 0:
                bbox_xyxy = tracked_dets[:, :4]
                categories = tracked_dets[:, 4]
                summary_data = scene_boxes(bbox_xyxy, categories, names, summary_sum=summary_data)
                s += f'\t=> ({predict_location})'
                summary_time += time_sync() - t1

            # During time
            if summary_time >= sum_time:
                predict_location = location_predict_vector(summary_data, predict_location)
                print('Current : {}\tDuring time : {}'.format(predict_location, summary_time))
                summary_data = ''
                summary_time = 0.0

            # Save detect Data
            now = time.strftime('%X', time.localtime(time.time()))
            if len(tracked_dets) != 0:
                with open(txt_path, 'a') as f:
                    if video:
                        f.write(f'{split_s[2]} : {predict_location}\t=> ')
                    else:
                        f.write(f'[{now}] : {predict_location}\t=> ')
                    for j in range(len(tracked_dets)):
                        ca = int(tracked_dets[j][4])
                        id = int(tracked_dets[j][8])
                        f.write(f'{names[ca]}:{id}')
                        f.write('  ')
                    f.write('\n')


def parse_opt():
    parser = argparse.ArgumentParser()

    # YOLOv5 params
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5/yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'yolov5/data/images',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'yolov5/data/coco128.yaml',
                        help='(optional) customDataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--project', default=ROOT / 'inference_location', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')

    # SORT params
    parser.add_argument('--sort-max-age', type=int, default=5,
                        help='keep track of object even if object is occluded or not detected in n frames')
    parser.add_argument('--sort-min-hits', type=int, default=2,
                        help='start tracking only after n number of objects detected')
    parser.add_argument('--sort-iou-thresh', type=float, default=0.1,
                        help='intersection-over-union threshold between two frames for association')

    # Detecting descript
    parser.add_argument('--start-point', type=str, default='AI', help='start point\'s category : [MainGate, Tunnel, '
                                                                      'Education, EduMainLib, Student, AI, MainLib, '
                                                                      'Rotary, Art]')
    parser.add_argument('--sum-time', type=float, default=4.0, help='Designated as 4 seconds based on the image of '
                                                                    'FPS 30.')

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
