# LaDiffCodec
Cite as: <i> Haici Yang, Inseon Jang, and Minje Kim. "Generative De-Quantization for Neural Speech Codec Via Latent Diffusion." ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024.</i>

- Paper: https://arxiv.org/pdf/2311.08330
- Website: https://minjekim.com/research-projects/ladiffcodec/ (Audio samples and supplement)
  
## Prerequisites
### Environment
<code> pip -r install requirements.txt </code>
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
| enc_ratios   | The downsampling ratios of encoder (and decoder)  | 

## Pretrained Checkpoints:
We provided pretrained 16khz EnCodec and LaDiffCodec at 1.5kbps and 3kbps at [link](https://indiana-my.sharepoint.com/:f:/g/personal/hy17_iu_edu/Eo9tTiag-u9JtkswVUr5wWIBKrA6hyEJx-TTF2USOGsSVQ?e=MDPijk).
The downsampling rate of the provided LaDiffCodec is 8.

To use the pretrained models - 
- 3kbps
  
<code>python -m srcs.sample --model_for_cond 'EnCodec_libri_3kb/model_best.amlt' --model_path 'Ladiff_3kb_8/model_best.amlt' --run_diff --scaling_global --cond_bandwidth 3 --unet_scale_cond --input_dir [INPUT_DIR] --output_dir [OUTPUT_DIR] </code>

- 1.5kbps
  
<code>python -m srcs.sample --model_for_cond 'EnCodec_libri_1_5kb/model_best.amlt' --model_path 'Ladiff_1_5kb_8/model_best.amlt' --run_diff --scaling_global --cond_bandwidth 1.5 --unet_scale_cond --input_dir [INPUT_DIR] --output_dir [OUTPUT_DIR] </code>

## Training steps
### 1. Pre-train Codec (Discrete autoencoder)
The diffusion model is built upon pre-trained EnCodec or DAC codecs. 

- Encodec specific hyper-parameters:
  
| Symbol | Description | 
| --- | ----------- |
| rep_dims         |  Running diffusion model| 
| n_residual_layers | number of residual layers | 
| n_filters | feature dimension | 
| lstm | number of lstm layers | 

#### Examples:
<code> python -m srcs.train --lr 0.00005 --seq_len_p_sec 2.4 --rep_dims 128 --n_residual_layers 1 --enc_ratios 8 5 4 2 --quantization --bandwidth 1.5 --n_filters 32 --lstm 2 --model_type unet --seq_length 1200 --data_folder_path \[DATA FOLDER\] --use_disc --disc_freq 5</code>

### 2. Pre-train autoencoer (Continuous autoencoder)
#### Examples:
<code> python -m srcs.train --lr 0.00005 --seq_len_p_sec 2.4 --rep_dims 128 --n_residual_layers 1 --enc_ratios 8 4 --finetune_model \[PATH TO CONTINUOUS MODEL\] --n_filters 32 --lstm 2 --model_type unet --seq_length 1200 --data_folder_path \[DATA FOLDER\] </code>

### 3. Diffusion model training
#### Examples:
<code> python -m srcs.train --lr 0.00005 --seq_len_p_sec 2.4 --rep_dims 128 --diff_dims 256 --n_residual_layers 1 --enc_ratios 8 4 --finetune_model \[PATH TO CONTINUOUS MODEL\] --n_filters 32 --lstm 2 --model_for_cond \[PATH TO DISCRETE CODEC\] --exp_name \[EXPERIMENT NAME\] --run_diff --model_type unet --seq_length 1200 --data_folder_path \[DATA FOLDER\] --scaling_global --cond_quantization --cond_bandwidth 1 </code>
