# Adapted from https://github.com/hbredin/pyannote-db-callhome/blob/master/parse_transcripts.py
import glob
from datasets import Dataset, Audio, DatasetDict
import numpy as np 

def get_secs(x):
    return x * 4 * 2.0 / 8000

def represent_int(s):
    try:
        int(s)
        return True
    except ValueError as e:
        return False

def get_start_end(t1, t2):
    t1 = get_secs(t1)
    t2 = get_secs(t2)
    return t1, t2

def get_callhome_files(): 

    audio_paths = glob.glob('/home/kamil/datasets/callhome/jpn/jpn_mp3/*.mp3')

    audio_paths = {
        'data': audio_paths, 
    }
    cha_paths = {
        'data': [], 
    }

    for subset in audio_paths: 
        for cha_path in audio_paths[subset]: 
            file = cha_path.split('/')[-1].split('.')[0]
            cha_paths[subset].append('/home/kamil/datasets/callhome/jpn/jpn_cha/{}.cha'.format(file))
        
    return audio_paths, cha_paths


class CallHomeForSPDDataset: 

    def __init__(
        self, 
        audio_paths, 
        cha_paths, 
        sample_rate=16000,  
    ):
        self.audio_paths = audio_paths
        self.cha_paths = cha_paths
        self.sample_rate = sample_rate

    def crop_audio(self, files):
        # Load audio from path 
        new_batch = {
            "audio": [],
            'timestamps_start': [],  
            'timestamps_end': [], 
            "speakers": [], 
        }

        batch = [
            {key: values[i] for key, values in files.items()}
            for i in range(len(files["audio"]))
        ]

        for file in batch:
             # Crop audio based on timestamps (in samples)

            start_idx = int(file["timestamps_start"][0] * self.sample_rate)
            end_idx = int(max(file["timestamps_end"]) * self.sample_rate)

            waveform = file['audio']['array']

            audio = {
                "array": np.array(waveform[start_idx: end_idx]), 
                "sampling_rate": self.sample_rate, 
            }

            timestamps_start = [start - file["timestamps_start"][0] for start in file["timestamps_start"]]
            timestamps_end = [end - file["timestamps_start"][0] for end in file["timestamps_end"]]

            new_batch["audio"].append(audio)
            new_batch['timestamps_start'].append(timestamps_start)
            new_batch['timestamps_end'].append(timestamps_end)
            new_batch['speakers'].append(file['speakers'])
    
        return new_batch

    def process_cha_file(self, path_to_cha): 

        timestamps_start = []
        timestamps_end = []
        speakers = []

        line = open(path_to_cha, 'r').read().splitlines()
        for i, line in enumerate(line):
            if line.startswith('*'):
                id = line.split(':')[0][1:]
            splits = line.split(" ")
            if splits[-1].find('_') != -1:
                indexes = splits[-1].strip()
                start = indexes.split("_")[0].strip()[1:]
                end = indexes.split("_")[1].strip()[:-1]
                if represent_int(start) and represent_int(end):
                    start, end = get_start_end(int(start), int(end))

                    speakers.append(id)
                    timestamps_start.append(start)
                    timestamps_end.append(end)

        return timestamps_start, timestamps_end, speakers

    def construct_dataset(self, num_proc=1): 

        self.spd_dataset = DatasetDict()

        for subset in self.cha_paths: 

            timestamps_start = []
            timestamps_end = []
            speakers = []

            self.spd_dataset[str(subset)] = Dataset.from_dict({})

            for rttm in self.cha_paths[subset]: 

                timestamps_start_file, timestamps_end_file, speakers_file = self.process_cha_file(rttm)

                timestamps_start.append(timestamps_start_file)
                timestamps_end.append(timestamps_end_file)
                speakers.append(speakers_file)

            self.spd_dataset[subset] = Dataset.from_dict({
                "audio": self.audio_paths[subset], 
                "timestamps_start": timestamps_start, 
                "timestamps_end": timestamps_end, 
                "speakers": speakers, 

            }).cast_column("audio", Audio(sampling_rate=self.sample_rate))

            self.spd_dataset[subset] = self.spd_dataset[subset].map(
                lambda example: self.crop_audio(example),
                batched=True,
                batch_size=8,
                remove_columns=self.spd_dataset[subset].column_names,
                num_proc=num_proc,
            ).cast_column("audio", Audio(sampling_rate=self.sample_rate))

        return self.spd_dataset    


if __name__ == '__main__': 

    audio_paths, cha_paths = get_callhome_files()
    dataset = CallHomeForSPDDataset(audio_paths, cha_paths).construct_dataset(num_proc=24)
    dataset.push_to_hub('kamilakesbi/callhome_jpn')