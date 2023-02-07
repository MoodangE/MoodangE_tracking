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


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0), congestion=None, summary_sum=None):
    cv2.putText(img, congestion, (10, 50), cv2.FONT_ITALIC, 2, (255, 255, 255), cv2.LINE_8, 2)
    real_time_count = 0
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # box text and bar
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0

        if cat == 0:
            color = compute_color_for_labels(id)
            summary_sum.append(id)

            label = f'{names[cat]} | {id}'
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(
                img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (x1, y1 +
                                     t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img, summary_sum


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
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='inference_person',  # save results to project/name
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
    save_img = not nosave and not source.endswith('.txt')  # save inference images
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
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Init define
    predict_congestion = 'None'
    summary_data = []
    summary_time = 0.0

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
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

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
            # print('\n')
            # print('Input into SORT:\n', dets_to_sort, '\n')

            # Run SORT
            tracked_dets = sort_tracker.update(dets_to_sort)

            # print('Output from SORT:\n', tracked_dets, '\n')

            # draw boxes for visualization
            if len(tracked_dets) > 0:
                bbox_xyxy = tracked_dets[:, :4]
                identities = tracked_dets[:, 8]
                categories = tracked_dets[:, 4]
                im0, summary_data = draw_boxes(im0, bbox_xyxy, identities, categories, names,
                                               congestion=predict_congestion, summary_sum=summary_data)
                summary_time += time_sync() - t1
            else:
                cv2.putText(im0, predict_congestion, (10, 50), cv2.FONT_ITALIC, 2, (255, 255, 255), cv2.LINE_8, 2)

            # During time
            if summary_time >= sum_time:
                predict_congestion = calculate_congestion(summary_data)
                summary_data = []
                summary_time = 0.0

            # Write detections to file. NOTE: Not MOT-compliant format.
            if save_txt and len(tracked_dets) != 0:
                for j, tracked_dets in enumerate(tracked_dets):
                    bbox_x1 = tracked_dets[0]
                    bbox_y1 = tracked_dets[1]
                    bbox_x2 = tracked_dets[2]
                    bbox_y2 = tracked_dets[3]
                    category = tracked_dets[4]
                    u_overdot = tracked_dets[5]
                    v_overdot = tracked_dets[6]
                    s_overdot = tracked_dets[7]
                    identity = tracked_dets[8]

                    with open(txt_path, 'a') as f:
                        f.write(
                            f'{frame_idx},{bbox_x1},{bbox_y1},{bbox_x2},{bbox_y2},{category},{u_overdot},{v_overdot},{s_overdot},{identity}\n')

            # print(f'{s} Done. ({t2 - t1})')
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

            # Progress
            print(f'{s} Done.')

            # Time taken per frame
            total_duration = time_sync() - t1
            print('\tTime taken per frame: {:.4f}'.format(total_duration))

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


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
    parser.add_argument('--iou-thres', type=float, default=0.4, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='inference_person', help='save results to project/name')
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
    parser.add_argument('--sum-time', type=float, default=5.0, help='Designated as 4 seconds based on the image of '
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
