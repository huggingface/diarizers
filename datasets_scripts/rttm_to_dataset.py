import glob
from datasets import Dataset, Audio, DatasetDict
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
LOCAL_PATH = os.environ.get('LOCAL_PATH')


def process_rttm_file(path_to_rttm): 
    
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


def get_subset_dataset(subset = 'train'): 

    rttm_paths = glob.glob(LOCAL_PATH + '/AMI-diarization-setup/only_words/rttms/{}/*.rttm'.format(subset))

    timestamps_start = []
    timestamps_end = []
    speakers = []
    audio_paths = []
    meetings = []

    for rttm in rttm_paths: 

        meeting = rttm.split('/')[-1].split('.')[0]
        meetings.append(meeting)
        timestamps_start_meeting, timestamps_end_meeting, speakers_meeting = process_rttm_file(rttm)    

        timestamps_start.append(timestamps_start_meeting)
        timestamps_end.append(timestamps_end_meeting)
        speakers.append(speakers_meeting)

        audio_paths.append(LOCAL_PATH + 'AMI-diarization-setup/pyannote/amicorpus/{}/audio/{}.Mix-Headset.wav'.format(meeting, meeting))

    audio_dataset = Dataset.from_dict({
        "audio": audio_paths, 
        "timestamps_start": timestamps_start, 
        "timestamps_end": timestamps_end, 
        "speakers": speakers, 
        "meeting_id": meetings, 

    }).cast_column("audio", Audio())

    return audio_dataset


speaker_diarization_dataset = DatasetDict(
    {
        "train": Dataset.from_dict({}),
        "validation": Dataset.from_dict({}),
        "test": Dataset.from_dict({}),
    }
    )

speaker_diarization_dataset['train'] = get_subset_dataset('train')
speaker_diarization_dataset['validation'] = get_subset_dataset('dev')
speaker_diarization_dataset['test'] = get_subset_dataset('test')


speaker_diarization_dataset.push_to_hub('real_ami_ihm')





