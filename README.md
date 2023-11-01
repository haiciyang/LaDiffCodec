# LaDiffCodec
Open-sourced codes for paper - GENERATIVE DE-QUANTIZATION FOR NEURAL SPEECH CODEC VIA LATENT DIFFUSION (submitted to ICASSP 2024)
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


### 2. Pre-train autoencoer (Continuous autoencoder)
#### Examples:
<code> python -m srcs.train --lr 0.00005 --seq_len_p_sec 2.4 --rep_dims 128 --n_residual_layers 1 --enc_ratios 8 4 --finetune_model \[PATH TO CONTINUOUS MODEL\] --n_filters 32 --lstm 2 --model_type unet --seq_length 1200 --data_folder_path \[DATA FOLDER\] </code>

### 3. Diffusion model training
#### Examples:
<code> python -m srcs.train --lr 0.00005 --seq_len_p_sec 2.4 --rep_dims 128 --diff_dims 256 --n_residual_layers 1 --enc_ratios 8 4 --finetune_model \[PATH TO CONTINUOUS MODEL\] --n_filters 32 --lstm 2 --model_for_cond \[PATH TO DISCRETE CODEC\] --exp_name \[EXPERIMENT NAME\] --run_diff --model_type unet --seq_length 1200 --data_folder_path \[DATA FOLDER\] --scaling_global --cond_quantization --cond_bandwidth 1 </code>

## Model Evaluation
#### Examples:
<code> python -m srcs.sample --seq_len_p_sec 2.4 --rep_dims 128 --diff_dims 256 --n_residual_layers 1 --enc_ratios 8 4 --finetune_model \[PATH TO CONTINUOUS MODEL\] --n_filters 32 --lstm 2 --model_for_cond \[PATH TO DISCRETE CODEC\] --exp_name \[EXPERIMENT NAME\] --run_diff --model_type unet --seq_length 1200 --data_folder_path \[DATA FOLDER\] --scaling_global --cond_quantization --cond_bandwidth 1 </code>
