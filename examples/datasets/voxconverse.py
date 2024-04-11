import glob
from diarizers.data.speaker_diarization import SpeakerDiarizationDataset
import argparse

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()

    parser.add_argument("--path_to_voxconverse", required=True)
    parser.add_argument("--push_to_hub", required=False, default=False)
    parser.add_argument("--hub_repository", required=False)

    args = parser.parse_args()

    
    rttm_files = {
        'dev': glob.glob(args.path_to_voxconverse + '/voxconverse/rttm_dev/*.rttm'),  
        'test': glob.glob(args.path_to_voxconverse + '/voxconverse/rttm_test/*.rttm') 
    }

    audio_files = {
        'dev': glob.glob(args.path_to_voxconverse + '/voxconverse/audio_dev/*.wav'), 
        'test': glob.glob(args.path_to_voxconverse + '/voxconverse/audio_test/*.wav') 
    }

    dataset = SpeakerDiarizationDataset(audio_files, rttm_files).construct_dataset()

    if args.push_to_hub == 'True': 
        dataset.push_to_hub(args.hub_repository)
