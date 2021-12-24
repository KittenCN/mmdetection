# import os
# os.system('python /root/mmdetection/tools/test.py /root/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py /root/mmdetection/work_dirs/faster_rcnn_r50_fpn_1x_coco/latest.pth --eval bbox')

from mmdet.apis import init_detector
from mmdet.apis import inference_detector
# from mmdet.apis import show_result
from mmdet.apis import show_result_pyplot
 
# 模型配置文件
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
 
# 预训练模型文件
checkpoint_file = 'work_dirs/result/normal_1x_1000epochs/latest.pth'
 
# 通过模型配置文件与预训练文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')
 
# 测试单张图片并进行展示
img = 'test2.jpg'
result = inference_detector(model, img)
# show_result(img, result, model.CLASSES)
# show_result_pyplot(model, img, result)
show_result_pyplot(model, img, result)
