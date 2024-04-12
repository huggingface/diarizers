import argparse
import copy
import glob
import os

from pydub import AudioSegment

from diarizers.data.speaker_diarization import SpeakerDiarizationDataset


def get_ami_files(path_to_ami, setup="only_words", hm_type="ihm"):

    """_summary_

    Returns:
        _type_: _description_
    """
    assert setup in ["only_words", "mini"]
    assert hm_type in ["ihm", "sdm"]

    rttm_files = {
        "train": glob.glob(path_to_ami + "/AMI-diarization-setup/{}/rttms/{}/*.rttm".format(setup, "train")),
        "validation": glob.glob(path_to_ami + "/AMI-diarization-setup/{}/rttms/{}/*.rttm".format(setup, "dev")),
        "test": glob.glob(path_to_ami + "/AMI-diarization-setup/{}/rttms/{}/*.rttm".format(setup, "test")),
    }

    audio_files = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for subset in rttm_files:

        rttm_list = copy.deepcopy(rttm_files[subset])

        for rttm in rttm_list:
            meeting = rttm.split("/")[-1].split(".")[0]
            if hm_type == "ihm":
                path = path_to_ami + "/AMI-diarization-setup/pyannote/amicorpus/{}/audio/{}.Mix-Headset.wav".format(
                    meeting, meeting
                )
                if os.path.exists(path):
                    audio_files[subset].append(path)
                else:
                    rttm_files[subset].remove(rttm)
            if hm_type == "sdm":
                path = path_to_ami + "/AMI-diarization-setup/pyannote/amicorpus/{}/audio/{}.Array1-01.wav".format(
                    meeting, meeting
                )
                if os.path.exists(path):
                    audio_files[subset].append(path)
                else:
                    rttm_files[subset].remove(rttm)

    return audio_files, rttm_files


def get_callhome_files(path_to_callhome, langage="jpn"):

    audio_files = glob.glob(path_to_callhome + "/callhome/{}/*.mp3".format(langage))

    audio_files = {
        "data": audio_files,
    }
    cha_files = {
        "data": [],
    }

    for subset in audio_files:
        for cha_path in audio_files[subset]:
            file = cha_path.split("/")[-1].split(".")[0]
            cha_files[subset].append(path_to_callhome + "/callhome/{}/{}.cha".format(langage, file))

    return audio_files, cha_files


def get_simsamu_files(path_to_simsamu):

    rttm_files = glob.glob(path_to_simsamu + "/simsamu/*/*.rttm")
    audio_files = glob.glob(path_to_simsamu + "/simsamu/*/*.m4a")

    for file in audio_files:
        sound = AudioSegment.from_file(file, format="m4a")
        file.split("/")
        file_hanlde = sound.export(file.split(".")[0] + ".wav", format="wav")

    audio_files = glob.glob(path_to_simsamu + "/simsamu/*/*.wav")

    audio_files = {"data": audio_files}

    rttm_files = {"data": rttm_files}

    return audio_files, rttm_files


def get_voxconverse_files(path_to_voxconverse):

    rttm_files = {
        "dev": glob.glob(path_to_voxconverse + "/voxconverse/dev/*.rttm"),
        "test": glob.glob(path_to_voxconverse + "/voxconverse/test/*.rttm"),
    }

    audio_files = {
        "dev": glob.glob(path_to_voxconverse + "/voxconverse/audio/*.wav"),
        "test": glob.glob(path_to_voxconverse + "/voxconverse/voxconverse_test_wav/*.wav"),
    }

    return audio_files, rttm_files


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--path_to_dataset", required=True)
    parser.add_argument("--setup", required=False, default="only_words")
    parser.add_argument("--push_to_hub", required=False, default=False)

    parser.add_argument("--hub_repository", required=False)
    args = parser.parse_args()

    if args.dataset == "ami":

        audio_files, rttm_files = get_ami_files(path_to_ami=args.path_to_dataset, setup=args.setup, hm_type="ihm")
        ami_dataset_ihm = SpeakerDiarizationDataset(audio_files, rttm_files).construct_dataset()
        if args.push_to_hub == "True":
            ami_dataset_ihm.push_to_hub(args.hub_repository, "ihm")

        audio_files, rttm_files = get_ami_files(path_to_ami=args.path_to_dataset, setup=args.setup, hm_type="sdm")
        ami_dataset_sdm = SpeakerDiarizationDataset(audio_files, rttm_files).construct_dataset()
        if args.push_to_hub == "True":
            ami_dataset_sdm.push_to_hub(args.hub_repository, "sdm")

    if args.dataset == "callhome":

        langages = ["eng", "jpn", "spa", "zho", "deu"]

        for langage in langages:
            audio_files, cha_files = get_callhome_files(args.path_to_dataset, langage=langage)
            dataset = SpeakerDiarizationDataset(
                audio_files, cha_files, annotations_type="cha", crop_unannotated_regions=True
            ).construct_dataset(num_proc=24)

            if args.push_to_hub == "True":
                dataset.push_to_hub(args.hub_repository, str(langage))

    if args.dataset == "simsamu":
        audio_files, rttm_files = get_simsamu_files(args.path_to_dataset)
        dataset = SpeakerDiarizationDataset(audio_files, rttm_files).construct_dataset()

        if args.push_to_hub == "True":
            dataset.push_to_hub(args.hub_repository)

    if args.dataset == "voxconverse":
        audio_files, rttm_files = get_voxconverse_files(args.path_to_dataset)
        dataset = SpeakerDiarizationDataset(audio_files, rttm_files).construct_dataset()
        print(dataset)
        if args.push_to_hub == "True":
            dataset.push_to_hub(args.hub_repository)
