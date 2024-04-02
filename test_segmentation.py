import argparse
import os

from datasets import load_dataset
from diarizers.models.segmentation.hf_model import SegmentationModel
from pyannote.audio import Model
from diarizers.test import Test

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", help="", default="kamilakesbi/real_ami_ihm")

    parser.add_argument('--pretrained_or_finetuned', help="", default='finetuned', choices= ['finetuned', 'pretrained'])
    parser.add_argument('--checkpoint_path', help="", default='checkpoints/ami')
    args = parser.parse_args()

    test_dataset = load_dataset(str(args.dataset_name), split='test', num_proc=12)
    
    if str(args.pretrained_or_finetuned) == 'finetuned': 
        model = SegmentationModel()
        model = model.from_pretrained(str(args.checkpoint_path))
        model = model.to_pyannote_model()

    else:
        model = Model.from_pretrained(
                "pyannote/segmentation-3.0", use_auth_token=True
            )

    test = Test(test_dataset, model, step=2.5)
    metrics = test.compute_metrics()
    print(metrics)
