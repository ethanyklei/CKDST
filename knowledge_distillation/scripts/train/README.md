# Training
The training of our model is divided into two steps：
1. pre-train mt model
2. training st model

We recommend setting the `REPO_ROOT` variable to avoid potential path errors.
### Step 1: Pre-train MT Model
We first pre-train a MT teacher model using additional MT data.
When start training, you need to set `DATA_ROOT` to the path where pre-processed MT data is saved and set `SAVE_ROOT` to the path where you want to save the model.

We leverage yaml to configure our training process, you can find it in `knowledge_distillation/config/mt`.

When everything is ready, you can use the following command to quickly start training.
```bash
bash train_mt.sh en_de_wmt de 
```

### Step 2: Training ST Model
After MT pre-training is complete, the next step of training can proceed.

First, you need to set `DATA_ROOT` to the path where pre-processed MUST-C data is saved and set `SAVE_ROOT` to the path where you want to save the model. Then, you need download [wav2vec](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt) model and set `wav2vec_path` to the wav2vec model. Finally, you need to set `load_pretrained_encoder_from` and `load_pretrained_decoder_from` in training config to the path where the pre-trained MT model is.

When everything is ready, you can use the following command to quickly start training.
```bash
bash train_st.sh wav_dkd_with_crd_extra de
```