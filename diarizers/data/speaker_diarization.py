from datasets import Dataset, Audio, DatasetDict


class SpeakerDiarizationDataset: 
    """
    Convert a speaker diarization dataset made of <audio files, rttm files>
    into a HF dataset with the following features: 
        - "audio": Audio feature. 
        - "speakers": The list of audio speakers, with their order of appearance. 
        - "timestamps_start": A list of timestamps indicating the start of each speaker segment.
        - "timestamps_end": A list of timestamps indicating the end of each speaker segment.
    """

    def __init__(
        self, 
        audio_paths, 
        rttm_paths, 
        sample_rate=16000,  
    ):
        """
        Args:
            audio_paths (dict): A dict with keys (str): split subset - example: "train" and values: list of str paths to audio files.  
            rttm_paths (dict): A dict with keys (str): split subset - example: "train" and values: list of str paths to RTTM files.  
            sample_rate (int, optional): Audios sampling rate in the generated HF dataset. Defaults to 16000.
        """
        self.audio_paths = audio_paths
        self.rttm_paths = rttm_paths
        self.sample_rate = sample_rate

    def process_rttm_file(self, path_to_rttm): 
        """ extract the list of timestamps_start, timestamps_end and speakers
        from an RTTM file with path: path_to_rttm. 

        Args:
            path_to_rttm (str): path to the RTTM file. 

        Returns:
            timestamps_start (list):  A list of timestamps indicating the start of each speaker segment.
            timestamps_end (list): A list of timestamps indicating the end of each speaker segment.
            speakers (list): The list of audio speakers, with their order of appearance. 
        """

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
        """ Main method to construct the dataset 

        Returns:
            self.spd_dataset: HF dataset compatible with diarizers. 
        """

        self.spd_dataset = DatasetDict()

        for subset in self.audio_paths: 

            timestamps_start = []
            timestamps_end = []
            speakers = []

            self.spd_dataset[str(subset)] =  Dataset.from_dict({})

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

            }).cast_column("audio", Audio(sampling_rate=self.sample_rate))

        return self.spd_dataset

