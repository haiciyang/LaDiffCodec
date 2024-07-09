# LaDiffCodec
Open-sourced codes for paper - GENERATIVE DE-QUANTIZATION FOR NEURAL SPEECH CODEC VIA LATENT DIFFUSION (Accepted by ICASSP 2024)

Cite as: <i>Yang, Haici, Inseon Jang, and Minje Kim. "Generative De-Quantization for Neural Speech Codec Via Latent Diffusion." ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024.</i>
## Prerequisites
### Environment
<code> pip install -r requirements.txt </code>
### Data
Librispeech
### Dependencies
- EnCodec - [https://github.com/facebookresearch/encodec](https://github.com/facebookresearch/encodec)
- Descript Audio Codec (DAC) - [https://github.com/descriptinc/descript-audio-codec](https://github.com/descriptinc/descript-audio-codec)

##  Hyper-Parameters:

| Symbol | Description |
| --- | ----------- |
| run_diff          |  Running diffusion model|
| diff_dims          | Dimension of input feature to the diffusion model |
| cond_quantization          | Whether the condition features should be quantized . Turn it on when training diffusion model on codecs.|
| cond_bandwidth          | The designated bitrate of this codec model |
| scaling_feature                 | Apply scaling on each feature map only |
| scaling_global               |  Apply scaling globally |
| ratios   | The downsampling ratios of encoder (and decoder)  | 

## Pretrained Checkpoints:

We provided a pretrained LaDiffCodec checkpoint with scalable bitrates at [link](https://indiana-my.sharepoint.com/:f:/g/personal/hy17_iu_edu/Eo9tTiag-u9JtkswVUr5wWIBKrA6hyEJx-TTF2USOGsSVQ?e=MDPijk). The bitrates can be chosen from 1.5kbps, 3kbps, 6kbps, 9kbps, 12kbps.

To use the pretrained models -   
<code>python -m srcs.main --synthesis --load_model [path]/diffusor.amlt --continuous_AE [path]/continuous_AE.amlt  --discrete_AE [path]/discrete_AE.amlt --cond_bandwidth [BANDWIDTH] --scaling_feature --diff_dims 256 --input_dir [INPUT_DIR] --output_dir [OUTPUT_DIR]  </code>
