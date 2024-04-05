import argparse
import os

from datasets import load_dataset
from diarizers.models.segmentation.hf_model import SegmentationModel
from pyannote.audio import Model
from diarizers.test import Test
from diarizers.utils import train_val_test_split

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", help="", default="kamilakesbi/callhome_jpn")

    parser.add_argument('--pretrained_or_finetuned', help="", default='finetuned', choices= ['finetuned', 'pretrained'])
    parser.add_argument('--checkpoint_path', help="", default='checkpoints/cv_for_spd_ja_2k_rayleigh')
    parser.add_argument('--num_proc', help="", default=12)
    parser.add_argument('--do_split', help="", default=True)

    args = parser.parse_args()

    dataset = load_dataset(str(args.dataset_name), num_proc=int(args.num_proc))

    if args.do_split is True:  
        dataset = train_val_test_split(dataset['data'])
    
    test_dataset = dataset['test']
    
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
