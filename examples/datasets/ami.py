import glob
from diarizers.data.speaker_diarization import SpeakerDiarizationDataset
import argparse
import os 
import copy

def get_ami_files(
        path_to_ami, 
        setup='only_words', 
        hm_type='ihm'
    ): 

    """_summary_

    Returns:
        _type_: _description_
    """
    assert setup in ['only_words', 'mini']
    assert hm_type in ['ihm', 'sdm']

    rttm_paths = {
        'train': glob.glob( path_to_ami + '/AMI-diarization-setup/{}/rttms/{}/*.rttm'.format(setup, 'train')), 
        'validation': glob.glob( path_to_ami + '/AMI-diarization-setup/{}/rttms/{}/*.rttm'.format(setup, 'dev')), 
        'test': glob.glob( path_to_ami + '/AMI-diarization-setup/{}/rttms/{}/*.rttm'.format(setup, 'test')), 
    }

    audio_paths = {
        'train': [], 
        'validation': [], 
        'test': [], 
    }
    
    for subset in rttm_paths: 

        rttm_list = copy.deepcopy(rttm_paths[subset])

        for rttm in rttm_list: 
            meeting = rttm.split('/')[-1].split('.')[0]
            if hm_type == 'ihm': 
                path = path_to_ami + '/AMI-diarization-setup/pyannote/amicorpus/{}/audio/{}.Mix-Headset.wav'.format(meeting, meeting)
                if os.path.exists(path):
                    audio_paths[subset].append(path)
                else: 
                    rttm_paths[subset].remove(rttm)
            if hm_type == 'sdm': 
                path = path_to_ami + '/AMI-diarization-setup/pyannote/amicorpus/{}/audio/{}.Array1-01.wav'.format(meeting, meeting)
                if os.path.exists(path):
                    audio_paths[subset].append(path)
                else: 
                    rttm_paths[subset].remove(rttm)

    return rttm_paths, audio_paths


if __name__ == '__main__': 

    parser = argparse.ArgumentParser()

    parser.add_argument("--path_to_ami", required=True)
    parser.add_argument("--setup", required=False, default='only_words')
    parser.add_argument("--push_to_hub", required=False, default=False)
    
    parser.add_argument("--hub_repository", required=False)

    args = parser.parse_args()

    rttm_paths, audio_paths = get_ami_files(path_to_ami=args.path_to_ami, setup=args.setup, hm_type = 'ihm')

    ami_dataset_ihm = SpeakerDiarizationDataset(audio_paths, rttm_paths).construct_dataset()
    
    if args.push_to_hub == 'True': 
        ami_dataset_ihm.push_to_hub(args.hub_repository,'ihm')

    rttm_paths, audio_paths = get_ami_files(path_to_ami=args.path_to_ami, setup=args.setup, hm_type = 'sdm')

    ami_dataset_sdm = SpeakerDiarizationDataset(audio_paths, rttm_paths).construct_dataset()
    
    if args.push_to_hub == 'True': 
        ami_dataset_sdm.push_to_hub(args.hub_repository, 'sdm')
