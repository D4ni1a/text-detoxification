# Text detoxification

Shulepin Danila (BS21-DS02)

d.shulepin@innopolis.university

## Task description

Text Detoxification Task is a process of transforming the text with toxic style into the text with the same meaning but with neutral style.

> Formal definition of text detoxification task can be found in [Text Detoxification using Large Pre-trained Neural Models by Dale et al., page 14](https://arxiv.org/abs/2109.08914)

The goal of the assignment is to create a solution for detoxing text with high level of toxicity.

## Data description

### Main raw dataset

The dataset is a subset of the ParaNMT corpus (50M sentence pairs) - the filtered ParaNMT-detox corpus (500K sentence pairs). This is the main dataset for the assignment detoxification task.

The data is given in the `.tsv` format, means columns are separated by `\t` symbol.

| Column | Type | Discription |
| ----- | ------- | ---------- |
| reference | str | First item from the pair |
| ref_tox | float | toxicity level of reference text |
| translation | str | Second item from the pair - paraphrazed version of the reference|
| trn_tox | float | toxicity level of translation text |
| similarity | float | cosine similarity of the texts |
| lenght_diff | float | relative length difference between texts |

### External dataset

The dataset is tsd-train dataset, which may be found [here](https://raw.githubusercontent.com/hexinz/SI630_final_project/main/Data/tsd_train.csv).

The data is given in the `.csv` format.

| Column | Type | Discription |
| ----- | ------- | ---------- |
| spans | list | List of chars ids for toxic phrases/words/tokens |
| text | str | The text containing toxic phrases |

## Structure of repository

```
text-detoxification
├── README.md # The top-level README
│
├── data
│   ├── external # Data from third party sources
│   ├── interim  # Intermediate data that has been transformed
│   └── raw      # The original, immutable data
│
├── models       # Trained and serialized models, final checkpoints
│
├── notebooks    #  Two Jupyter notebooks: Data loading and Model building            
│
├── references   # Reference list
│
├── reports      # Two PDF reports: Solution building procedure and Final model description
│   └── figures  # Final model architecture
│
├── requirements.txt # The requirements file for reproducing the analysis environment
└── src                 # Source code for use
    │                 
    ├── data            # Scripts to download or generate data
    │   ├── make_data_bin.py
    │   └── make_data_span.py
    │
    ├── model          # Scripts to train models and then use trained models to make predictions
    │   ├── predict_model.py
    │   ├── toxic_classifier.py
    │   ├── toxic_parafraser.py
    │   ├── toxic_tagger.py
    │   ├── train_span_model.py
    │   └── train_classifier_model.py
    │   
    └── visualization   # Scripts to create results visualizations
        └── visualize.py
```

## Requirements

Requirements may be founde in requirements.txt

Install by:

> pip install requirements.txt

## How to run?

In order to run the project, clone repository localy and install requirements.

In you `Python3` script do:

> from train_span_model import train_span_model
> 
> from predict_model import predict

> model = train_span_model()
>
> result = predict(sentence, model)
