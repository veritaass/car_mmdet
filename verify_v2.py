import imp
from pkg_resources import fixup_namespace_packages
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import numpy as np
import pathlib
from datetime import datetime

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file'
        , default='tools/car_test1.jpg')
    parser.add_argument('--config', help='Config file'
        , default='configs/htkim_car/htkim_yolox_s_8x8_300e_coco_3rd.py')
    parser.add_argument('--checkpoint', help='Checkpoint file'
        , default='work_dirs/htkim_yolox_s_8x8_300e_coco_3rd/epoch_2000.pth')
    parser.add_argument('--score-thr', type=float
        , default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args

def main(args):
    #config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    # config_file = 'configs/htkim_car/htkim_yolox_s_8x8_300e_coco_2nd.py'
    config_file = args.config
    # download the checkpoint from model zoo and put it in `checkpoints/`
    # url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
    #checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    #checkpoint_file = 'work_dirs/htkim_yolox_s_8x8_300e_coco_2nd/best_bbox_mAP_epoch_780.pth'
    # checkpoint_file = 'work_dirs/htkim_yolox_s_8x8_300e_coco_2nd/epoch_1500.pth'
    checkpoint_file = args.checkpoint
    #score_thr=0.3
    
    device = 'cuda:1'
    # init a detector
    model = init_detector(config_file, checkpoint_file, device=device)
    # inference the demo image
    img = args.img
    # img = '/raid/templates/farm-data/car/actual_testdata/car_test1.jpg'
    rst_inf = inference_detector(model, img)  # 'demo/demo.jpg')
    #print(rst_inf)
    # f_name = "res_" + "-".join(model.CLASSES) + "_score" + str(args.score_thr) + "_" + pathlib.Path(img).name
    out_path =  'htkim_result/' + pathlib.Path(config_file).name.split(".")[0] + '/' + pathlib.Path(checkpoint_file).name.split(".")[0] + '/'
    f_name = out_path + "res_" + pathlib.Path(checkpoint_file).name.split(".")[0] + "_score" + str(args.score_thr) + "_" + pathlib.Path(img).name
    show_result_pyplot2(model, img, rst_inf, score_thr=args.score_thr, f_name=f_name)

def show_result_pyplot2(model,
                       img,
                       result,
                       score_thr=0.3,
                       title='result',
                       wait_time=0,
                       f_name="resultfile"
                       ):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        title (str): Title of the pyplot figure.
        wait_time (float): Value of waitKey param.
                Default: 0.
    """
    # export result  to file
    # print(model.CLASSES) --->> ('license_plate',)
    #f_name = "res_" + "-".join(model.CLASSES) + "_" + pathlib.Path(img).name + "_score" + str(score_thr) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    # f_name = "res_" + "-".join(model.CLASSES) + "_score" + str(score_thr) + "_" + pathlib.Path(img).name

    # export result to image
    if hasattr(model, 'module'):
        model = model.module
    model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=False,
        wait_time=wait_time,
        win_name=title,
        out_file=f_name,
        font_size='9',
        bbox_color=(2, 10, 241),
        text_color=(241, 10, 2))

    f_content = str(result)
    o_file = open(f_name+".txt", "w")
    o_file.write(f_content)
    o_file.close()


if __name__ == '__main__':
    args=parse_args()
    main(args)
