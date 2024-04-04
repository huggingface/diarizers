import glob
from diarizers.data.speaker_diarization import SpeakerDiarizationDataset

rttm_files = {
    'dev': glob.glob('/home/kamil/datasets/voxconverse/rttm_dev/*.rttm'),  
    'test': glob.glob('/home/kamil/datasets/voxconverse/rttm_test/*.rttm') 
}

audio_files = {
    'dev': glob.glob('/home/kamil/datasets/voxconverse/audio_dev/*.wav'), 
    'test': glob.glob('/home/kamil/datasets/voxconverse/audio_test/*.wav') 
}

dataset = SpeakerDiarizationDataset(audio_files, rttm_files).construct_dataset()

dataset.push_to_hub('kamilakesbi/voxconverse')
