import numpy as np
from datasets import load_dataset
from pyannote.database import registry
from sklearn.metrics.pairwise import cosine_similarity

from pyannote.audio import Model
from pyannote.audio.tasks import SpeakerDiarization


def get_chunk_from_pyannote(seg_task, file_id, start_time, duration):

    seg_task.prepare_data()
    seg_task.setup()

    chunk = seg_task.prepare_chunk(file_id, start_time, duration)

    return chunk


def sanity_checks():

    # Extract 10 second audio from meeting EN2001a (= file_id 124).
    # We choose start_time = 3.34 to match with the first 10 seconds of audio from the synthetic AMI.
    synthetic_ami_chunk = synthetic_ami_dataset_processed["train"][0]
    waveform_synthetic = np.array(synthetic_ami_chunk["waveforms"])
    synthetic_labels = np.array(synthetic_ami_chunk["labels"])
    index_positions = np.nonzero(waveform_synthetic)

    real_ami_chunk = get_chunk_from_pyannote(seg_task, 124, 3.34, 10)
    real_labels = real_ami_chunk["y"].data
    waveform_real = np.array(real_ami_chunk["X"][0])

    waveform_synthetic_without_zeros = waveform_synthetic[index_positions]
    waveform_real_without_zeros = waveform_real[index_positions]

    similarity_without_zeros = cosine_similarity(
        [waveform_synthetic_without_zeros], [waveform_real_without_zeros]
    )
    similarity_with_zeros = cosine_similarity([waveform_synthetic], [waveform_real])

    assert (synthetic_labels == real_labels).all(), "labels are not matching"
    assert similarity_without_zeros > 0.95
    assert similarity_with_zeros > 0.8

    # We choose start_time = 5.90 to get a sample that doesn't match with the first 10 seconds of audio from the synthetic AMI.
    real_ami_chunk = get_chunk_from_pyannote(seg_task, 124, 5.90, 10)
    real_labels = real_ami_chunk["y"].data
    waveform_real = np.array(real_ami_chunk["X"][0])
    waveform_real_without_zeros = waveform_real[index_positions]

    similarity_without_zeros = cosine_similarity(
        [waveform_synthetic_without_zeros], [waveform_real_without_zeros]
    )
    similarity_with_zeros = cosine_similarity([waveform_synthetic], [waveform_real])

    assert (synthetic_labels == real_labels).all() == False
    assert similarity_without_zeros < 0.01
    assert similarity_with_zeros < 0.01


if __name__ == "__main__":

    registry.load_database(
        "/home/kamil/datasets/AMI-diarization-setup/pyannote/database.yml"
    )
    ami = registry.get_protocol("AMI.SpeakerDiarization.only_words")

    seg_task = SpeakerDiarization(
        ami, duration=10.0, max_speakers_per_chunk=3, max_speakers_per_frame=2
    )
    pretrained = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=True)
    seg_task.model = pretrained

    synthetic_ami_dataset_processed = load_dataset(
        "kamilakesbi/ami_spd_nobatch_processed_sc"
    )

    sanity_checks()
