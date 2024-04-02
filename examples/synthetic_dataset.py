from datasets import load_dataset
from diarizers.data.synthetic import SyntheticDataset

if __name__ == '__main__': 

    config = {
        "audio_file_length": 1.1,
        "batch_size": 8, 
        "std_concatenate": 2,
        "sample_rate": 16000,
        "refine_with_vad": True,
        "denoise": True,
        "normalize": True,
        "augment": True,
        "silent_regions": {
            "silent_regions": True,
            "silence_duration": 10,
            "silence_proba": 0.5,
        },
        "bn_path": "/home/kamil/datasets/wham_noise/wham_noise/tr",
        "ir_path": "/home/kamil/datasets/MIT-ir-survey",
    }

    common_voice = load_dataset(
        "mozilla-foundation/common_voice_16_1", "en", num_proc=2
    )
    speaker_column_name = "client_id"
    audio_column_name = "audio"

    spd_dataset = SyntheticDataset(
        common_voice,
        speaker_column_name,
        audio_column_name,
        config,
    ).create_spd_dataset(num_proc=1)

    spd_dataset.push_to_hub("kamilakesbi/commonvoice_spd_synthetic")
