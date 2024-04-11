# Adapted from https://github.com/hbredin/pyannote-db-callhome/blob/master/parse_transcripts.py
import glob
import argparse
from diarizers.data.speaker_diarization import SpeakerDiarizationDataset

def get_callhome_files(
        path_to_callhome, 
        langage='jpn'
    ): 

    audio_paths = glob.glob(path_to_callhome + '/callhome/{}/*.mp3'.format(langage))

    audio_paths = {
        'data': audio_paths, 
    }
    cha_paths = {
        'data': [], 
    }

    for subset in audio_paths: 
        for cha_path in audio_paths[subset]: 
            file = cha_path.split('/')[-1].split('.')[0]
            cha_paths[subset].append(path_to_callhome + '/callhome/{}/{}.cha'.format(langage, file))
        
    return audio_paths, cha_paths


if __name__ == '__main__': 

    parser = argparse.ArgumentParser()

    parser.add_argument("--path_to_callhome", required=True)
    parser.add_argument("--push_to_hub", required=False, default=False)    
    parser.add_argument("--hub_repository", required=False)

    args = parser.parse_args()
    langages = ['eng', 'jpn', 'spa', 'zho', 'deu']

    for langage in langages: 

        audio_paths, cha_paths = get_callhome_files(args.path_to_callhome, langage=langage)
        dataset = SpeakerDiarizationDataset(
            audio_paths, 
            cha_paths, 
            annotations_type='cha', 
            crop_unannotated_regions=True
        ).construct_dataset(num_proc=24)

        if args.push_to_hub == 'True': 
            dataset.push_to_hub(args.hub_repository, str(langage))