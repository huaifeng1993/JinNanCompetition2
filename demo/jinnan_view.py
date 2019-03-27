# --------------------------------------------------------
# Written by SHEN HUIXIANG  (shhuixi@qq.com)
# Created On: 2019-3-9
# --------------------------------------------------------
from maskrcnn_benchmark.config import cfg
from predictor import JINNANDemo
import cv2
import os
import json
config_file = "configs/e2e_faster_rcnn_R_50_FPN_1x.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
jinnan_demo = JINNANDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.5,
)
# load image and then run prediction

images_dir='datasets/jinnan/test/'
images_path=os.listdir(images_dir)
images_path.sort(key = lambda x: int(x[:-4]))

jingnan_result=[]
for x in range(len(images_path)):
    print(x)
    image = cv2.imread(os.path.join(images_dir,images_path[x]))
  
    result = jinnan_demo.run_on_opencv_image(image)
    cv2.imshow("COCO detections", result)
    if cv2.waitKey() == 27:
            break  # esc to quit
cv2.destroyAllWindows()