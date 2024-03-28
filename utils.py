from collections import Counter

from datasets import load_dataset


def get_nb_speakers_per_chunks(
    processed_dataset="kamilakesbi/real_ami_ihm_processed",
):

    dataset_processed = load_dataset(processed_dataset)

    assert "nb_speakers" in dataset_processed["train"].features

    nb_speakers_in_chunk_train = [
        len(list) for list in dataset_processed[str("train")]["nb_speakers"]
    ]
    nb_speakers_in_chunk_val = [
        len(list) for list in dataset_processed[str("validation")]["nb_speakers"]
    ]
    nb_speakers_in_chunk_test = [
        len(list) for list in dataset_processed[str("test")]["nb_speakers"]
    ]

    return {
        "train": Counter(nb_speakers_in_chunk_train),
        "val": Counter(nb_speakers_in_chunk_val),
        "test": Counter(nb_speakers_in_chunk_test),
    }


if __name__ == "__main__":

    real_ami_processed = "kamilakesbi/real_ami_ihm_processed"
    results_real_ami = get_nb_speakers_per_chunks(real_ami_processed)

    print("Real AMI: ", results_real_ami)

    synthetic_ami_processed = "kamilakesbi/ami_spd_nobatch_full_processed"
    results_synthetic_ami = get_nb_speakers_per_chunks(synthetic_ami_processed)
    print("Synthetic AMI: ", results_synthetic_ami)

    synthetic_ami_processed_2 = "kamilakesbi/ami_spd_augmented_test2_processed"
    results_synthetic_ami = get_nb_speakers_per_chunks(synthetic_ami_processed_2)
    print("Synthetic AMI: ", results_synthetic_ami)
