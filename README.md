## How to change and modify the neural machine translation model
This is an implementation of the Transformer model for Machine Translation
#### Install 
`pip install -r requirements.txt`
#### Get the data
- IWSLT'15 EN-VI for English-Vietnamese translation
Download and pre-process the EN-VI data with the following commands:
```bash
sh scripts/en_vi.sh
sh preprocess_data.sh spm en vi
```
By default, the downloaded dataset in `./data/en_vi`.
As with the [official implementation](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py),
`spm` (`sentencepiece`) encoding is used to encode the raw text as data pre-processing. The encoded data is by default
in `./temp/run_en_vi_spm`. 

### Train and evaluate model ###
```bash
python transformer_main.py --run_mode=train_and_evaluate \
    --config-model=config_model
    --config-data=config_envi
```

* Specify `--output-dir` to dump model checkpoints and training logs to a desired directory.
  By default it is set to `./outputs`. 
* Specifying `--output-dir` will also restore the latest model checkpoint under the directory, if any checkpoint exists.
* Additionally, you can also specify `--load-checkpoint` to load a previously trained checkpoint from `output_dir`.

### Test a trained model ###
To only evaluate a model checkpoint without training, first load the checkpoint and generate samples:
```bash
python transformer_main.py \
    --run_-mode=test \
    --config-data=config_envi \
    --output-dir=./outputs
```

The latest checkpoint in `./outputs` is used. Generated samples are in the file `./outputs/test.output.hyp`, and
reference sentences are in the file `./outputs/test.output.ref`. The script shows the cased BLEU score as provided by
the [`tx.evals.file_bleu`](https://texar-pytorch.readthedocs.io/en/latest/code/evals.html#file-bleu) function. 

Alternatively, you can also compute the BLEU score with the raw sentences using the `bleu_main` script:

```bash
python bleu_main.py --reference=data/en_vi/test.vi --translation=temp/test.output.hyp
```

## Run Your Customized Experiments

Here is a hands-on tutorial on running Transformer with your own customized dataset.

### 1. Prepare raw data

Create a data directory and put the raw data in the directory. To be compatible with the data preprocessing in the next
step, you may follow the convention below:

* The data directory should be named as `data/${src}_${tgt}/`. Take the data downloaded with `scripts/iwslt15_en_vi.sh`
  for example, the data directory is `data/en_vi`.
* The raw data should have 6 files, which contain source and target sentences of training/dev/test sets, respectively.
  In the `iwslt15_en_vi` example, `data/en_vi/train.en` contains the source sentences of the training set, where each
  line is a sentence. Other files are `train.vi`, `dev.en`, `dev.vi`, `test.en`, `test.vi`. 

### 2. Preprocess the data

To obtain the processed dataset, run

```bash
preprocess_data.sh ${encoder} ${src} ${tgt} ${vocab_size} ${max_seq_length}
```
where

* The `encoder` parameter can be `bpe`(byte pairwise encoding), `spm` (sentence piece encoding), or
`raw`(no subword encoding).
* `vocab_size` is optional. The default is 32000. 
  - At this point, this parameter is used only when `encoder` is set to `bpe` or `spm`. For `raw` encoding, you'd have
    to truncate the vocabulary by yourself.
  - For `spm` encoding, the preprocessing may fail (due to the Python sentencepiece module) if `vocab_size` is too
    large. So you may want to try smaller `vocab_size` if it happens. 
* `max_seq_length` is optional. The default is 70.

In the `iwslt15_en_vi` example, the command is `sh preprocess_data.sh spm en vi`.

By default, the preprocessed data are dumped under `temp/run_${src}_${tgt}_${encoder}`. In the `iwslt15_en_vi` example,
the directory is `temp/run_en_vi_spm`.

If you choose to use `raw` encoding method, notice that:

- By default, the word embedding layer is built with the combination of source vocabulary and target vocabulary. For
  example, if the source vocabulary is of size 3K and the target vocabulary of size 3K and there is no overlap between
  the two vocabularies, then the final vocabulary used in the model is of size 6K.
- By default, the final output layer of transformer decoder (hidden_state -> logits) shares the parameters with the word
  embedding layer.

### 3. Specify data and model configuration

Customize the Python configuration files to config the model and data.

Please refer to the example configuration files `config_model.py` for model configuration and `config_iwslt15.py` for
data configuration.

### 4. Train the model

Train the model with the following command:

```bash
python transformer_main.py \
    --run-mode=train_and_evaluate \
    --config-model=<custom_config_model> \
    --config-data=<custom_config_data>
```
where the model and data configuration files are `custom_config_model.py` and `custom_config_data.py`, respectively.

Outputs such as model checkpoints are by default under `outputs/`.

### 5. Test the model

Test with the following command:

```bash
python transformer_main.py \
    --run-mode=test \
    --config-data=<custom_config_data> \
    --output-dir=./outputs
```

Generated samples on the test set are in `outputs/test.output.hyp`, and reference sentences are in
`outputs/test.output.ref`. If you've used `bpe` or `spm` encoding in the data preprocessing step, make sure to set
`encoding` in the data configuration to the appropriate encoding type. The generated output will be decoded using the
specified encoding.

Finally, to evaluate the BLEU score against the ground truth on the test set:

```bash
python bleu_main.py --reference=<your_reference_file> --translation=temp/test.output.hyp.final
```
For the `iwslt15_en_vi` example, use `--reference=data/en_vi/test.vi`.