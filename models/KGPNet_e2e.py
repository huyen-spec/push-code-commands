from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg
import config as CFG
import os
import torch
import json
from models.modules import KGPStandardROIHeads
from utils.custom_trainer import CustomTrainer

def train(args):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.OUTPUT_DIR = CFG.base_log + args.name
    cfg.MODEL.TRAIN_GCN = args.train_gcn
    cfg.DATASETS.TRAIN = ("pills_train",)
    cfg.DATASETS.TEST = ("pills_test",)
    cfg.DATALOADER.NUM_WORKERS = args.n_workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = os.path.join(CFG.warmstart_path, "model_final.pth")
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr  # pick a good LR
    cfg.SOLVER.MAX_ITER = args.max_iters
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.SOLVER.CHECKPOINT_PERIOD = CFG.cpt_frequency
    cfg.MODEL.DEVICE = 'cuda' if args.cuda else 'cpu'
    cfg.MODEL.ROI_HEADS.NAME = 'KGPStandardROIHeads'
    cfg.MODEL.ROI_HEADS.GRAPH_EBDS_PATH = CFG.graph_ebds_path
    cfg.MODEL.ROI_HEADS.PREDICTOR_INPUT_SHAPE = 1024
    cfg.MODEL.ROI_HEADS.PREDICTOR_HIDDEN_SIZE = 128
    cfg.MODEL.ROI_HEADS.LINKING_LOSS_WEIGHT = args.linking_loss_weight
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.n_classes  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.KEYPOINT_ON = False
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    if args.save_best:
        cfg.TEST.EVAL_PERIOD = CFG.test_period
        trainer = CustomTrainer(cfg)
    else:
        trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=args.resume)
    # save_model(trainer, args.name)
    trainer.train()
    # save_model(trainer, args.name)

def save_model(trainer, name):
    state_dict = trainer.model.state_dict()
    torch.save(state_dict, os.path.join(trainer.cfg.OUTPUT_DIR, name + ".pth"))

def test(args):
    print(args)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.OUTPUT_DIR = CFG.base_log + args.name
    cfg.DATASETS.TEST = ("pills_test",)
    cfg.MODEL.TRAIN_GCN = args.train_gcn
    cfg.DATALOADER.NUM_WORKERS = args.n_workers
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.DEVICE = 'cuda' if args.cuda else 'cpu'
    cfg.MODEL.ROI_HEADS.NAME = 'KGPStandardROIHeads'
    cfg.MODEL.ROI_HEADS.PREDICTOR_INPUT_SHAPE = 1024
    cfg.MODEL.ROI_HEADS.GRAPH_EBDS_PATH = CFG.graph_ebds_path
    cfg.MODEL.ROI_HEADS.PREDICTOR_HIDDEN_SIZE = 128
    cfg.MODEL.ROI_HEADS.LINKING_LOSS_WEIGHT = args.linking_loss_weight
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch_size
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.n_classes  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    cfg.MODEL.KEYPOINT_ON = False

    if args.resume_path == '':
        print('Load last model')
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
    else:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, args.resume_path)

    predictor = DefaultPredictor(cfg)

    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader
    from detectron2.data import DatasetCatalog
    from detectron2.utils.visualizer import Visualizer
    import cv2
    import matplotlib.pyplot as plt

    # visualization
    test_dict = DatasetCatalog.get("pills_test")
    d = test_dict[5]
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    print(f'gtruth: {d["annotations"]}')
    print(f'predict: {outputs["instances"]}')
    v = Visualizer(im[:, :, ::-1],
                    metadata=d,
                   #instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(out.get_image())
    plt.savefig(f'eval_{args.name}.png', dpi=300)
    
    # # evaluation
    evaluator = COCOEvaluator("pills_test", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "pills_test")
    result = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(result)
    with open(cfg.OUTPUT_DIR + "/results.json", "w") as f:
        json.dump(result, f)
    