# Ko-GPT2 model
We use Ko-GPT2 model to generate cat diary from clip model's label

Our model weights is accessible from huggingface __larcane/kogpt2-cat-diary__

## Requirements
```
pip install transformer
```

## Datasets
We fine-tune our model by using cat diary corpus.
You find our dataset in ./data


## Train
```
# In this repo kogpt directory
python train.py --data-path data/cat_diary_data_2.3v.txt \
                --block-size 48 \
                --epochs 30 \
                --save-dir work_dirs/model_output \
```

