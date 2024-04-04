import glob
from diarizers.data.speaker_diarization import SpeakerDiarizationDataset
from pydub import AudioSegment

rttm_files = glob.glob('/home/kamil/datasets/simsamu/*/*.rttm')
audio_files = glob.glob('/home/kamil/datasets/simsamu/*/*.m4a')

for file in audio_files: 
    sound = AudioSegment.from_file(file, format='m4a')
    file.split('/')
    file_hanlde = sound.export(file.split('.')[0] + '.wav', format='wav')

audio_files = glob.glob('/home/kamil/datasets/simsamu/*/*.wav')

audio_files = {
    'train':audio_files 
}

rttm_files = {
    'train':rttm_files 
}

dataset = SpeakerDiarizationDataset(audio_files, rttm_files).construct_dataset()

dataset.push_to_hub('kamilakesbi/simsamu')