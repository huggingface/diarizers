import glob
from datasets import Dataset, Audio, DatasetDict
# from dotenv import load_dotenv
import os


class SpeakerDiarizationDataset: 

    def __init__(
        self, 
        audio_paths, 
        rttm_paths, 
    ):
        self.audio_paths = audio_paths
        self.rttm_paths = rttm_paths

    def process_rttm_file(self, path_to_rttm): 
        
        timestamps_start = []
        timestamps_end = []
        speakers = []

        with open(path_to_rttm, "r") as file:

            lines = file.readlines()

            for line in lines:

                fields = line.split()

                speaker = fields[-3]
                start_time = float(fields[3]) 
                end_time  = start_time + float(fields[4])    

                timestamps_start.append(start_time)
                speakers.append(speaker)
                timestamps_end.append(end_time)

        return timestamps_start, timestamps_end, speakers

    def construct_dataset(self): 

        self.spd_dataset = DatasetDict(
            {
                "train": Dataset.from_dict({}),
                "validation": Dataset.from_dict({}),
                "test": Dataset.from_dict({}),
            }
        )

        for subset in self.audio_paths: 

            timestamps_start = []
            timestamps_end = []
            speakers = []

            for rttm in self.rttm_paths[subset]: 

                timestamps_start_file, timestamps_end_file, speakers_file = self.process_rttm_file(rttm)    

                timestamps_start.append(timestamps_start_file)
                timestamps_end.append(timestamps_end_file)
                speakers.append(speakers_file)


            self.spd_dataset[subset] = Dataset.from_dict({
                "audio": audio_paths[subset], 
                "timestamps_start": timestamps_start, 
                "timestamps_end": timestamps_end, 
                "speakers": speakers, 

            }).cast_column("audio", Audio())

        return self.spd_dataset


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