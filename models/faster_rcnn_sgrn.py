from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg
import config as CFG
import os
import torch
import json
from utils.custom_trainer import CustomTrainer

import pdb 


# /home/huyen/projects/duyna/KGPNet-DetectionModel/logs/baseline/model_0004999.pth

def train(args):

    # pdb.set_trace()

    # model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    # pdb.set_trace()

    cfg = get_cfg()
    # pdb.set_trace()

    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    cfg.merge_from_file("/home/huyen/anaconda3/envs/huyen3/lib/python3.8/site-packages/detectron2/model_zoo/configs/COCO-Detection/faster_rcnn_sgrn_R_50_FPN_3x.yaml")

    cfg.OUTPUT_DIR = CFG.base_log + args.name + "_test1406"
    cfg.DATASETS.TRAIN = ("pills_train",)
    cfg.DATASETS.TEST = ("pills_test",)
    cfg.DATALOADER.NUM_WORKERS = args.n_workers
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr  # pick a good LR
    cfg.SOLVER.MAX_ITER = args.max_iters
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.SOLVER.CHECKPOINT_PERIOD = CFG.cpt_frequency
    # cfg.ROI_HEADS.NAME = "StandardROIHeads_wgraph"
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.n_classes  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.ROI_HEADS.FREEZE_FEAT = True
    cfg.MODEL.KEYPOINT_ON = False
    cfg.MODEL.BACKBONE.FREEZE = True
    cfg.MODEL.BACKBONE.FREEZE_P5 = True

    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    if args.save_best:
        cfg.TEST.EVAL_PERIOD = CFG.test_period
        trainer = CustomTrainer(cfg)
    else:
        trainer = DefaultTrainer(cfg) 

    # pdb.set_trace()
    # cfg.MODEL.WEIGHTS = '/home/huyen/projects/huyen/Test1/duy/logs/sgrn/model_0004999.pth'
    
    # cfg.MODEL.WEIGHTS = "/home/huyen/projects/huyen/Test1/duy/logs/sgrn/model_0019999.pth"

    # cfg.MODEL.WEIGHTS = "/home/huyen/projects/huyen/Test1/duy/logs/sgrn_test1206/model_0011999.pth"

    cfg.MODEL.WEIGHTS = "/home/huyen/projects/huyen/Test1/duy/logs/baseline/model_0019999.pth"

    trainer.resume_or_load(resume=args.resume)
    # save_model(trainer, args.name)
    trainer.train()
    # save_model(trainer, args.name)

def save_model(trainer, name):
    state_dict = trainer.model.state_dict()
    torch.save(state_dict, os.path.join(trainer.cfg.OUTPUT_DIR, name + ".pth"))

def test(args):
    cfg = get_cfg()

    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file("/home/huyen/anaconda3/envs/huyen3/lib/python3.8/site-packages/detectron2/model_zoo/configs/COCO-Detection/faster_rcnn_sgrn_R_50_FPN_3x.yaml")
    cfg.OUTPUT_DIR = CFG.base_log + args.name+ "_test1206"
    cfg.DATASETS.TEST = ("pills_test",)
    cfg.DATALOADER.NUM_WORKERS = args.n_workers
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch_size
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.n_classes  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.KEYPOINT_ON = False
    
    # cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 
    if args.resume_path == '':
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
    else:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, args.resume_path)
        
    # cfg.MODEL.WEIGHTS = '/home/huyen/projects/huyen/Test1/duy/logs/sgrn/temp/model_0004999.pth'
    # cfg.MODEL.WEIGHTS = '/home/huyen/projects/huyen/Test1/duy/logs/sgrn/model_0012999.pth'

    cfg.MODEL.WEIGHTS = "/home/huyen/projects/huyen/Test1/duy/logs/sgrn_test1206/model_0019999.pth"
    predictor = DefaultPredictor(cfg)

    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader
    from detectron2.data import DatasetCatalog
    from detectron2.utils.visualizer import Visualizer
    import cv2
    import matplotlib.pyplot as plt

    # visualization
    test_dict = DatasetCatalog.get("pills_test")
    d = test_dict[100]
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    print(d['annotations'])
    v = Visualizer(im[:, :, ::-1],
                    metadata=d,
                   #instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(out.get_image())
    plt.savefig(f'eval_{args.name}_100.png', dpi=300)

    pdb.set_trace()
    
    # # evaluation
    evaluator = COCOEvaluator("pills_test", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "pills_test")
    result = inference_on_dataset(predictor.model, val_loader, evaluator)

    pdb.set_trace()
    print(result)
    with open(cfg.OUTPUT_DIR + "/results.json", "w") as f:
        json.dump(result, f)
    