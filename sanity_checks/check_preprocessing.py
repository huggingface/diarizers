import numpy as np
from pyannote.audio import Model
from pyannote.audio.tasks import SpeakerDiarization
from pyannote.database import registry
from sklearn.metrics.pairwise import cosine_similarity

from datasets import load_dataset

from diarizers import Preprocess, SegmentationModel
import argparse


def get_chunk_from_pyannote(pyannote_task, file_id, start_time, duration):
    """Get a chunk from audio file using a pyannote task object. 
    Args:
        pyannote_task (pyannote.audio.tasks.segmentation.speaker_diarization.SpeakerDiarization): 
            pyannote SpeakerDiarization task object, with AMI__SpeakerDiarization__only_words as protocol.
        file_id (int): ID of the AMI dataset file.
        start_time (float): chunk start time.
        duration (float): chunk duration.

    Returns:
        chunk: dict containing: 
            'X': waveform tensor
            'y': pyannote SlidingWindowFeature with the target
            'meta': dict with metadata.
    """

    pyannote_task.prepare_data()
    pyannote_task.setup()

    chunk = pyannote_task.prepare_chunk(file_id, start_time, duration)

    return chunk


def test_pyannote_diarizers_preprocessing_equivalence(path_to_ami):
    """Check that preprocessing with diarizers and pyannote is equivalent 
        on a given 10 sec audio chunk.

    Args:
        path_to_ami (str): path to the local pyannote AMI dataset. 
    """

    # 1. Load the AMI dataset using pyannote:
    registry.load_database(path_to_ami + "/AMI-diarization-setup/pyannote/database.yml")
    ami_pyannote = registry.get_protocol("AMI.SpeakerDiarization.only_words")

    # Define the pyannote task used to preprocess the AMI dataet:
    pyannote_task = SpeakerDiarization(ami_pyannote, duration=10.0, max_speakers_per_chunk=3, max_speakers_per_frame=2)
    pretrained = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=True)
    pyannote_task.model = pretrained

    # Get chunk from 0s to 10s from file 9 (=IS1002c meeting file): 
    ami_pyannote_example = get_chunk_from_pyannote(pyannote_task, 9, 0, 10)

    # 2. Load the AMI dataset from the Hugging Face hub:
    ami_dataset_hub = load_dataset('diarizers-community/ami', 'ihm')

    # Prepare preprocessing:
    model = SegmentationModel.from_pyannote_model(pretrained)
    preprocessor = Preprocess(model.config)

    # Select the first example (= meeting IS1002c), preprocess it and extract the first chunk:
    ami_hub_example = ami_dataset_hub['train'].select(range(1))
    ami_hub_example = ami_hub_example.map(
        lambda file: preprocessor(file, random=False, overlap=0.0),
        num_proc=1,
        remove_columns=ami_hub_example.column_names,
        batched=True,
        batch_size=1,
        keep_in_memory=True
    )[0]

    # Compare labels and waveforms obtained with diarizers vs pyannote preprocessing:
    waveform_hub = np.array(ami_hub_example["waveforms"])
    labels_hub = np.array(ami_hub_example["labels"])

    labels_pyannote = ami_pyannote_example["y"].data
    waveform_pyannote = np.array(ami_pyannote_example["X"][0])

    similarity = cosine_similarity([waveform_hub], [waveform_pyannote])

    assert (labels_hub == labels_pyannote).all(), "labels are not matching"
    assert similarity > 0.95


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--path_to_ami", help='Specify path to the pyannote AMI dataset', required=True)
    args = parser.parse_args()

    test_pyannote_diarizers_preprocessing_equivalence(args.path_to_ami)
