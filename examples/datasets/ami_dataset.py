import glob
from diarizers.data.speaker_diarization import SpeakerDiarizationDataset


def get_ami_files(): 
    rttm_paths = {
        'train': glob.glob('/home/kamil/datasets/AMI-diarization-setup/only_words/rttms/{}/*.rttm'.format('train')), 
        'validation': glob.glob('/home/kamil/datasets/AMI-diarization-setup/only_words/rttms/{}/*.rttm'.format('dev')), 
        'test': glob.glob('/home/kamil/datasets/AMI-diarization-setup/only_words/rttms/{}/*.rttm'.format('test')), 
    }

    audio_paths = {
        'train': [], 
        'validation': [], 
        'test': [], 
    }
    
    for subset in rttm_paths: 
        for rttm in rttm_paths[subset]: 
            meeting = rttm.split('/')[-1].split('.')[0]
            audio_paths[subset].append('/home/kamil/datasets/AMI-diarization-setup/pyannote/amicorpus/{}/audio/{}.Mix-Headset.wav'.format(meeting, meeting))

    return rttm_paths, audio_paths


if __name__ == '__main__': 

    rttm_paths, audio_paths = get_ami_files()
    ami_dataset = SpeakerDiarizationDataset(audio_paths, rttm_paths).construct_dataset()
    print(ami_dataset)
