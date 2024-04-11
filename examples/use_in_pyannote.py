
from diarizers.models.segmentation.hf_model import SegmentationModel
from pyannote.audio import Inference
from datasets import load_dataset
import torch
import argparse
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    # dataset arguments:
    parser.add_argument("--pipeline", help="", default="speaker_diarization", choices=['segmentation', 'speaker_diarization'])

    args = parser.parse_args()

    # Example input audio: 
    ds = load_dataset("kamilakesbi/real_ami_ihm", num_proc=12, split='test')
    
    waveform = torch.tensor(ds[0]['audio']['array']).unsqueeze(0).to(torch.float32)
    sample_rate =  ds[0]['audio']['sampling_rate']
    input = {'waveform': waveform, 'sample_rate': sample_rate}
    
    # Select the fine-tuned model: 
    model = SegmentationModel().from_pretrained('checkpoints/ami')

    model = model.to_pyannote_model()

    if str(args.pipeline)=='segmentation': 

        # Get the inference result: 
        spk_probability = Inference(model, step=2.5)(input)
    
    if str(args.pipeline)=='speaker_diarization': 
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
        )
        pipeline.segmentation_model = model
        pipeline.to(torch.device("cuda"))
        with ProgressHook() as hook:
            diarization = pipeline(input, hook=hook)

        print(diarization)







