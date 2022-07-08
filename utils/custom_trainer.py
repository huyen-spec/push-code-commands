from detectron2.engine import DefaultTrainer
import os
from detectron2.evaluation import COCOEvaluator
from detectron2.engine.hooks import BestCheckpointer

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        import pdb
        pdb.set_trace()
        print("Bulding evaluator***************************************")

        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    def build_hooks(self):
        ret = super().build_hooks()
        ret.append(BestCheckpointer(self.cfg.TEST.EVAL_PERIOD, self.checkpointer, 'bbox/AP'))
        
        return ret

    