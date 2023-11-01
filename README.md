# LaDiffCodec
Open-source code for paper - GENERATIVE DE-QUANTIZATION FOR NEURAL SPEECH CODEC VIA LATENT DIFFUSION (submitted to ICASSP 2024)
## Prerequisites
### Environment
<code> pip -r install requirements.txt </code>

### Data
Librispeech
### Dependence
- EnCodec - [https://github.com/facebookresearch/encodec](https://github.com/facebookresearch/encodec)
- Descript Audio Codec (DAC) - [https://github.com/descriptinc/descript-audio-codec](https://github.com/descriptinc/descript-audio-codec)

## Important Parameters:

| Symbol | Description |
| --- | ----------- |
| run_diff          |  Running diffusion model|
| weight_src                   |  source separation loss scale|
| weight_mix                  |  remix loss scale |
| transfer_model               |  The model label to start transferring training from |
| trainset            | dataset. MUSDB or Slakh |
| train_loss               |  SDR or SDSDR |
| with_silent          | Whether or not using the data, has actually contains less number of sources than the number the model is designed on |
| baseline                   | Whether or not training baseline |
| ratio_on_rep                |  Whether or not having ratios applying on the representation feature space  |

## Training steps
### 1. Pre-train Codec (Discrete autoencoder)
The diffusion model is built upon pre-trained EnCodec or DAC codecs. 

### 2. Pre-train autoencoer (Continuous autoencoder)


### 3. Diffusion model training


#### Examples:
Scenario 1:  .
<code> python -m srcs.train --lr 0.00005 --seq_len_p_sec 2.4 --rep_dims 128 --diff_dims 256 --n_residual_layers 1 --enc_ratios 8 4 --finetune_model \[PATH TO CONTINUOUS MODEL\] --n_filters 32 --lstm 2 --model_for_cond \[PATH TO DISCRETE CODEC\] --exp_name \[EXPERIMENT NAME\] --run_diff --model_type unet --seq_length 1200 --data_folder_path '/N/project/SAIGE_shared/librispeech' --scaling_global --cond_quantization --cond_bandwidth 1 --debug

## Model Evaluation

To get the remix output of specific one track
<code>python3 eval_on_samples.py  --model_name 0826_123024 --n_src 4 --dataset MUSDB --loss_f SDSDR<code>
