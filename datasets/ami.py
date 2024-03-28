import argparse
import random

import numpy as np
from audiomentations import AddBackgroundNoise, ApplyImpulseResponse, Compose
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

bn_path = ("/home/kamil/datasets/wham_noise/wham_noise/tr",)
ir_path = ("/home/kamil/datasets/MIT-ir-survey",)

augmentation_pipeline = Compose(
    [
        ApplyImpulseResponse(ir_path, p=0.5),
        AddBackgroundNoise(bn_path, 0, 50, p=0.5),
    ]
)


def add_silent_regions(
    audio_file, sr, file_timestamps_start, file_timestamps_end, duration, p
):

    if random.random() < p and len(file_timestamps_start) > 2:
        duration = np.maximum(np.random.normal(duration, 3.0), 1)

        insert_silence_index = random.randint(0, len(file_timestamps_start) - 2)

        silence_start = file_timestamps_end[insert_silence_index]
        silence_end = silence_start + duration
        silence_start_index = int(silence_start * sr)
        silence_end_index = int(silence_end * sr)

        relative_duration = silence_end - min(
            file_timestamps_start[insert_silence_index + 1 :]
        )
        file_timestamps_start[insert_silence_index + 1 :] += relative_duration
        file_timestamps_end[insert_silence_index + 1 :] += relative_duration

        new_length = int(relative_duration * sr) + len(audio_file)
        extended_audio_file = np.zeros(new_length)

        extended_audio_file[:silence_start_index] = audio_file[:silence_start_index]

        length_segment_end = max(1, len(extended_audio_file[silence_end_index:]))

        extended_audio_file[-length_segment_end:] = audio_file[-length_segment_end:]

    else:
        extended_audio_file = audio_file

    return extended_audio_file, file_timestamps_start, file_timestamps_end


def concatenate(
    files,
    augment=True,
    add_silences=True,
):

    """_summary_

    Returns:
        _type_: _description_
    """

    new_batch = {
        "audio": [],
        "speakers": [],
        "timestamps_start": [],
        "timestamps_end": [],
    }

    sr = files["audio"][0]["sampling_rate"]

    audio_duration = int(max(files["end_time"]) - files["begin_time"][0])

    audio_file = np.zeros(audio_duration * sr)

    files = [
        {key: values[i] for key, values in files.items()}
        for i in range(len(files["audio"]))
    ]

    chunk_start_timestamp = files[0]["begin_time"]
    chunk_start = int(chunk_start_timestamp * sr)
    chunk_end = int(chunk_start + audio_duration * sr)

    speakers = []

    file_timestamps_start = []
    file_timestamps_end = []

    for element in files:

        timestamp_start = element["begin_time"]
        timestamp_end = element["end_time"]

        samples_start = int(timestamp_start * sr)

        audio_segment = element["audio"]["array"]
        audio_length = len(audio_segment)

        speaker = element["speaker_id"]

        start_index = samples_start - chunk_start
        segment_length = min(chunk_end - samples_start, audio_length)

        if samples_start > chunk_end:
            break

        audio_file[start_index : start_index + segment_length] += audio_segment[
            :segment_length
        ]

        speakers.append(str(speaker))

        file_timestamps_start.append(timestamp_start - chunk_start_timestamp)
        file_timestamps_end.append(
            min(timestamp_end - chunk_start_timestamp, audio_duration)
        )

    new_batch["speakers"].append(speakers)

    if add_silences:
        audio_file, file_timestamps_start, file_timestamps_end = add_silent_regions(
            audio_file,
            sr,
            file_timestamps_start,
            file_timestamps_end,
            duration=15,
            p=0.5,
        )

    if augment:
        audio_file = augmentation_pipeline(samples=audio_file, sample_rate=sr)

    audio_file = {
        "array": audio_file,
        "sampling_rate": sr,
    }

    new_batch["audio"].append(audio_file)
    new_batch["timestamps_start"].append(file_timestamps_start)
    new_batch["timestamps_end"].append(file_timestamps_end)

    return new_batch


def create_spd_dataset(ds, batch_size, nb_meetings, augment=False):

    """Function to create a speaker Diarization dataset
    from the ami for ASR.

    Returns:
        ds: _description_
        nb_samples_per_meeting:
        batch_size:
        audio_duration:
    """

    subsets = ["train", "validation", "test"]

    speaker_diarization_dataset = DatasetDict(
        {
            "train": Dataset.from_dict({}),
            "validation": Dataset.from_dict({}),
            "test": Dataset.from_dict({}),
        }
    )

    for subset in subsets:

        meetings = (
            ds[str(subset)]
            .to_pandas()["meeting_id"]
            .unique()[: nb_meetings[str(subset)]]
        )

        concatenate_dataset = Dataset.from_dict(
            {"audio": [], "speakers": [], "timestamps_start": [], "timestamps_end": []}
        )

        print(subset)
        for meeting in meetings:
            print(meeting)
            dataset = ds[str(subset)].filter(
                lambda x: x["meeting_id"] == str(meeting), num_proc=24
            )

            dataset = dataset.sort("begin_time")

            result = dataset.map(
                lambda example: concatenate(example, augment=augment),
                batched=True,
                batch_size=batch_size,
                remove_columns=dataset.column_names,
                num_proc=24,
                keep_in_memory=True,
            )

            concatenate_dataset = concatenate_datasets([concatenate_dataset, result])

        speaker_diarization_dataset[str(subset)] = concatenate_dataset

    return speaker_diarization_dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", help="", default="32")
    parser.add_argument("--nb_meetings_train", help="", default="-1")
    parser.add_argument("--nb_meetings_val", help="", default="-1")
    parser.add_argument("--nb_meetings_test", help="", default="-1")
    parser.add_argument("--augment", help="", default="True")

    args = parser.parse_args()

    ds = load_dataset("edinburghcstr/ami", "ihm")

    nb_meetings = {
        "train": int(args.nb_meetings_train),
        "validation": int(args.nb_meetings_val),
        "test": int(args.nb_meetings_test),
    }
    augment = args.augment == "True"

    spk_dataset = create_spd_dataset(
        ds, batch_size=int(args.bs), nb_meetings=nb_meetings, augment=augment
    )
    spk_dataset.push_to_hub("kamilakesbi/ami_spd_augmented_silences2")
