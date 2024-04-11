import glob
from diarizers.data.speaker_diarization import SpeakerDiarizationDataset
from pydub import AudioSegment
import argparse

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()

    parser.add_argument("--path_to_simsamu", required=True)
    parser.add_argument("--push_to_hub", required=False, default=False)
    parser.add_argument("--hub_repository", required=False)

    args = parser.parse_args()

    rttm_files = glob.glob(args.path_to_simsamu + '/simsamu/*/*.rttm')
    audio_files = glob.glob(args.path_to_simsamu + '/simsamu/*/*.m4a')

    for file in audio_files: 
        sound = AudioSegment.from_file(file, format='m4a')
        file.split('/')
        file_hanlde = sound.export(file.split('.')[0] + '.wav', format='wav')

    audio_files = glob.glob(args.path_to_simsamu + '/simsamu/*/*.wav')

    audio_files = {
        'train':audio_files 
    }

    rttm_files = {
        'train':rttm_files 
    }

    dataset = SpeakerDiarizationDataset(audio_files, rttm_files).construct_dataset()

    if args.push_to_hub == 'True': 
        dataset.push_to_hub(args.hub_repository)
