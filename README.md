# Super-transformers
> **Important**  
> This is not a formal implementation of the referenced papers. Instead, it's a simple experiment to explore these new ideas and test them out for fun.


## Overview
Super-transformers is an experimental implementation that explores modifications to the standard GPT-2 transformer architecture. This project combines three key architectural innovations:

- Scalable Softmax: Implementation of a modified attention mechanism using scalable softmax to improve computational efficiency and potentially enhance model performance on longer sequences.

- Positional Encoding Ablation: Exploring the model's capability to learn without explicit positional encodings, investigating whether contextual information alone is sufficient for sequence modeling.



## Experimental Results
Our preliminary experiments compare training and validation loss between:

- A model using scalable softmax without positional encodings
- A model using standard softmax with positional encodings

<p align="center">
  <img src="src/images/model_comparison.png" alt="Model Comparison" width="800"/>
</p>

<p align="center">
  <em>Figure 1: Comparison of training and validation losses between different architectural configurations.</em>
</p>

## Getting Started
```bash
git clone  https://github.com/dame-cell/Super-transformers.git
cd Super-transformers
pip install -r requirements.txt 
cd src
```

## Dataset
This project uses the [fineweb-small](https://huggingface.co/datasets/eliplutchok/fineweb-small-sample) dataset from Hugging Face. The preprocessing script handles downloading and preparing the data.


```bash
python3 data.py \
    --max_length 1024 \
    --sample_size 100000 \
    --data_name eliplutchok/fineweb-small-sample \
    --stride 256 \
    --split_ratio 0.9 \
    --batch_size 1000 \
    --output_dir .

```
## Training 

```bash 
python3 train.py \
    --train_data path_to_train_data \
    --test_data path_to_test_data \
    --size small \
    --ssmax True \
    --use_pos_enc False \
    --wandb False \
    --batch_size 2 \
    --generating_step 2000 \
    --validation_step 1000 \
    --save_model 1000 \
    --epoch 1 \
```
## Models 

> **Warning**  
> Please note that the model was only trained on 100k rows for a single epoch, so its generation quality may be limited.

You can find the Pretrained Models here: 
| **Model**             | **Link**                               |
|---------------------------|-----------------------------------------|
| Model(SSMax with no pos_enc)                      | [Model](https://huggingface.co/damerajee/super-transformers-model/blob/main/best_model.pth) 
| Model(no SSMax with pos_enc)                      |  [Model](https://huggingface.co/damerajee/super-transformers-model/blob/main/best_model_2.pth) 


### Citations

```bash
@inproceedings{Nakanishi2025ScalableSoftmax,
    title   = {Scalable-Softmax Is Superior for Attention},
    author  = {Ken M. Nakanishi},
    year    = {2025},
    url     = {https://arxiv.org/pdf/2501.19399}
}
```

```bash
@article{Irie2024WhyAP,
    title   = {Why Are Positional Encodings Nonessential for Deep Autoregressive Transformers? Revisiting a Petroglyph},
    author  = {Kazuki Irie},
    year    = {2024},
    url     = {https://arxiv.org/pdf/2501.00659}
}
```

```bash
@article{Yu2024TheSW,
    title   = {The Super Weight in Large Language Models},
    author  = {Mengxia Yu and De Wang and Qi Shan and Colorado Reed and Alvin Wan},
    year    = {2024},
    url     = {https://arxiv.org/pdf/2411.07191v1}
}
```
