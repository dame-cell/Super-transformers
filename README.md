# Super-transformers
We code a GPT2 transformers  but instead we use these three new ideas: 
- Scalable sofmax
- Try to train it without the positional encodings
- super weights for pruning during inference




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