import argparse
from datasets import load_dataset
from src.data.preprocess import Preprocess
from src.models.segmentation.hf_model import SegmentationModel

if __name__ == "__main__":

    ds = load_dataset("kamilakesbi/ami_spd_nobatch", num_proc=12)

    # Parameters to make the preprocesssing match the model hyperparameters: 
    hyperparameters = {
        'chunk_duration' : 10, 
        'max_speakers_per_frame' : 2, 
        'max_speakers_per_chunk': 3, 
        'min_duration' : None, 
        'warm_up' : (0.0, 0.0), 
        'weigh_by_cardinality': False
    }

    model = SegmentationModel(hyperparameters)

    preprocessed_dataset = Preprocess(ds, model).preprocess_dataset()
    # preprocessed_dataset.push_to_hub("kamilakesbi/real_ami_processed")