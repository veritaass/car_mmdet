from mmdet.apis import init_detector, inference_detector
from mmdet.models import build_detector
from mmdet.apis import train_detector

config_file = 'configs/yolox/htkim_yolox_s_8x8_300e_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'checkpoints/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
rst_inf = inference_detector(model, 'demo/demo.jpg')
print(rst_inf)
