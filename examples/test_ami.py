from datasets import load_dataset
from pyannote.audio import Model
from diarizers.test import Test


if __name__ == "__main__":
    test_dataset = load_dataset("kamilakesbi/real_ami_ihm", split="test")

    model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=True)

    test = Test(test_dataset, model, step=2.5)

    metrics = test.compute_metrics()
    print(metrics)

