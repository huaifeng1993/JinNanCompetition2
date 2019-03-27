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
    confidence_threshold=0.50,
)
# load image and then run prediction

images_dir='datasets/jinnan/testb/'
images_path=os.listdir(images_dir)
images_path.sort(key = lambda x: int(x[:-4]))

jingnan_result=[]
for x in range(len(images_path)):
    print(x)
    image = cv2.imread(os.path.join(images_dir,images_path[x]))
    prediction=jinnan_demo.compute_prediction(image)
    prediction = jinnan_demo.select_top_predictions(prediction)

    boxes = prediction.bbox.tolist()
    scores = prediction.get_field("scores").tolist()
    labels = prediction.get_field("labels").tolist()
    jingnan_result.extend(
            [
                {
                    "filename": images_path[x],
                    "rects": [{"xmin":int(info[0][0]),
                                "xmax":int(info[0][2]),
                                "ymin":int(info[0][1]),
                                "ymax":int(info[0][3]),
                                "label":info[1],
                                "confidence":info[2]} for info in zip(boxes,labels,scores)]
                }
            ]
        )
results={}
results['results']=jingnan_result
with open("./test_result.json",'w',encoding='utf-8') as json_file:
    json.dump(results,json_file,ensure_ascii=False)