# Robust Change Captioning
This repository contains the code and dataset for the following paper:

* DH. Park, T. Darrell, A. Rohrbach, *Robust Change Captioning.* in ICCV 2019. ([arXiv](https://arxiv.org/abs/1901.02527))

## Installation
1. Clone the repository (`git clone git@github.com:Seth-Park/RobustChangeCaptioning.git`)
2. `cd RobustChangeCaptioning`
3. Make virtual environment with Python 3.5 (`mkvirtualenv rcc -p python3.5`)
4. Install requirements (`pip install -r requirements.txt`)
5. Setup COCO caption eval tools ([github](https://github.com/tylin/coco-caption)) (Since the repo only supports Python 2.7, either create a separate virtual environment with Python 2.7 or modify the code to be compatible with Python 3.5).

## Data
1. Download data from here: [google drive link](https://drive.google.com/file/d/1HJ3gWjaUJykEckyb2M0MB4HnrJSihjVe/view?usp=sharing)
```
python google_drive.py 1HJ3gWjaUJykEckyb2M0MB4HnrJSihjVe clevr_change.tar.gz
tar -xzvf clevr_change.tar.gz
```
Extracting this file will create `data` directory and fill it up with CLEVR-Change dataset.

2. Preprocess data

We are providing the preprocessed data here: [google drive link](https://drive.google.com/file/d/1FA9mYGIoQ_DvprP6rtdEve921UXewSGF/view?usp=sharing).
You can skip the procedures explained below and just download them using the following command:
```
python google_drive.py 1FA9mYGIoQ_DvprP6rtdEve921UXewSGF ./data/clevr_change_features.tar.gz
cd data
tar -xzvf clevr_change_features.tar.gz
```

* Extract visual features using ImageNet pretrained ResNet-101:
```
# processing default images
python scripts/extract_features.py --input_image_dir ./data/images --output_dir ./data/features --batch_size 128

# processing semantically changes images
python scripts/extract_features.py --input_image_dir ./data/sc_images --output_dir ./data/sc_features --batch_size 128

# processing distractor images
python scripts/extract_features.py --input_image_dir ./data/nsc_images --output_dir ./data/nsc_features --batch_size 128
```

* Build vocab and label files using caption annotations:
```
python scripts/preprocess_captions.py --input_captions_json ./data/change_captions.json --input_neg_captions_json ./data/no_change_captions.json --input_image_dir ./data/images --split_json ./data/splits.json --output_vocab_json ./data/vocab.json --output_h5 ./data/labels.h5
```

## Training
To train the Dual Dynamic Attention Model (DUDA), run the following commands:
```
# create a directory or a symlink to save the experiments logs/snapshots etc.
mkdir experiments
# OR
ln -s $PATH_TO_DIR$ experiments

# this will start the visdom server for logging
# start the server on a tmux session since the server needs to be up during training
python -m visdom.server

# start training
python train.py --cfg configs/dynamic/dynamic.yaml 
```

The training script runs the model on the validation dataset every snapshot iteration and one can save the visualizations of dual attentions and dynamic attentions using the flag `visualize`:
```
python train.py --cfg configs/dynamic/dynamic.yaml --visualize
```

One can also control the strength of entropy regularization over the dynamic attention weights using the flag `entropy_weight`:
```
python train.py --cfg configs/dynamic/dynamic.yaml --visualize --entropy_weight 0.0001
```

## Testing/Inference
To test/run inference on the test dataset, run the following command
```
python test.py --cfg configs/dynamic/dynamic.yaml --visualize --snapshot 9000 --gpu 1
```
The command above will take the model snapshot at 9000th iteration and run inference using GPU ID 1, saving visualizations as well.

## Evaluation
* Caption evaluation
To evaluate captions, we need to first reformat the caption annotations into COCO eval tool format (only need to run this once). After setting up the COCO caption eval tools ([github](https://github.com/tylin/coco-caption)), make sure to modify `utils/eval_utils.py` so that the `COCO_PATH` variable points to the COCO eval tool repository. Then, run the following command:
```
python utils/eval_utils.py
```

After the format is ready, run the following command to run evaluation:
```
# This will run evaluation on the results generated from the validation set and print the best results
python evaluate.py --results_dir ./experiments/dynamic/eval_sents --anno ./data/total_change_captions_reformat.json --type_file ./data/type_mapping.json
```

Once the best model is found on the validation set, you can run inference on test set for that specific model using the command exlpained in the `Testing/Inference` section and then finally evaluate on test set:
```
python evaluate.py --results_dir ./experiments/dynamic/test_output/captions --anno ./data/total_change_captions_reformat.json --type_file ./data/type_mapping.json
```
The results are saved in `./experiments/dynamic/test_output/captions/eval_results.txt`

To evaluate based on IOUs, run the following command:
```
python evaluate_by_IOU.py --results_dir ./experiments/dynamic/test_output/captions --anno ./data/total_change_captions_reformat.json --type_file ./data/type_mapping.json --IOU_file ./data/change_captions_by_IOU.json
```
By default, this command will compute captioning scores on the top 25% difficult examples based on IOU. However, you can change the percentage or compute for easiest examples by modifying `evalaute_by_IOU.py`

* Pointing Game evaluation
Run the following command to run Pointing Game evaluation:
```
python evaluate_pointing.py --results_dir ./experiments/dynamic/test_output/attentions --anno ./data/change_captions_with_bbox.json --type_file ./data/type_mapping.json
```
The results are saved in `./experiments/dynamic/test_output/attentions/eval_results_pointing.txt`

You can also run Pointing Game evaluation by IOU:
```
python evaluate_pointing_by_IOU.py --results_dir ./experiments/dynamic/test_output/attentions --anno ./data/change_captions_with_bbox.json --type_file ./data/type_mapping.json --IOU_file ./data/change_captions_by_IOU.json
```

## Pretrained Model
We provide weights, training log, generated dual attention maps and captions here: [google drive link](https://drive.google.com/file/d/1DloxAUV19_WwUIeWPQDcHEGvGrW65Pu6/view?usp=sharing). You can download them using this command:
```
python google_drive.py 1DloxAUV19_WwUIeWPQDcHEGvGrW65Pu6 ./experiments/pretrained_DUDA.tar.gz
cd experiments
tar -xzvf pretrained_DUDA.tar.gz
```
