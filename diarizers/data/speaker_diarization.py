from datasets import Dataset, Audio, DatasetDict


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
                "audio": self.audio_paths[subset], 
                "timestamps_start": timestamps_start, 
                "timestamps_end": timestamps_end, 
                "speakers": speakers, 

            }).cast_column("audio", Audio())

        return self.spd_dataset

