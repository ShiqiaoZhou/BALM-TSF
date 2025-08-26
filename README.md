<div align="center">
  <h2><b>BALM-TSF: Balanced Multimodal Alignment for LLM-Based Time Series Forecasting</b></h2>
</div>

## Introduction

BALM-TSF (Balanced Multimodal Alignment for LLM-Based Time Series Forecasting) is a lightweight time series forecasting framework that balances textual prompts and numeric data. Follow these steps to run it.

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
