<div align="center">
  <h2><b>(CIKM'25) BALM-TSF: Balanced Multimodal Alignment for LLM-Based Time Series Forecasting</b></h2>
</div>

## Introduction

BALM-TSF (Balanced Multimodal Alignment for LLM-Based Time Series Forecasting) is a lightweight time series forecasting framework that balances textual prompts and numeric data. Follow these steps to run it.

<p align="center">
<img src="./figs/framework.png" height = "360" alt="" align=center />
</p>

- The model has two branches: (1) the Text Branch constructs a prompt embedding by combining learnable prompt embedding with word-embedded time series statistics, then encodes it with a frozen LLM; (2) the Time Series Branch normalizes raw data, applies a patch encoder, and projects features into the LLM’s hidden space. In the Balanced Alignment module, textual embeddings are rescaled and then aligned with temporal embeddings via a contrastive learning. The aligned two representations are concatenated and fed to a lightweight projection head for final prediction.

## Requirements

* Python 3.11 (recommended via Miniconda)
* Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

## Datasets

Download the pre-processed datasets from [Google Drive](https://drive.google.com/file/d/1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP/view?usp=sharing), and place the extracted contents under `./dataset`. This is a public dataset sharelink from [Time-LLM](https://github.com/KimMeen/Time-LLM).

## Quick Start

1. Download and unzip datasets into `./dataset`.
2. Tune the model using the provided scripts.

### Long-term Forecasting

```bash
bash ./scripts_long_term/BALM_ETTh1_GPT2.sh
```

### Few-shot Forecasting

```bash
bash ./scripts_few_shot/BALM_ETTh1_GPT2.sh
```

## Model Design

See `models/BALM.py` for implementation details.

## Acknowledgements

We appreciate [Time-Series-Library](https://github.com/thuml/Time-Series-Library) and [Time-LLM](https://github.com/KimMeen/Time-LLM) for code references and datasets.
