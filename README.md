# SIGNAL
Dataset for Semantic and Inferred Grammar Neurological Analysis of Language

## Repository structure

This repository follows the next structure:
```
├── src                       # Source code
|   ├── z-scores_estimation   # Code for pairwise estimation of statistical difference between conditions in EEG data
|   └── draw_plots            # Code for visualisation
├── README.md                 # README file
└── requirements.txt          # A file with requirements 
```

## Dataset

In this paper, we present SIGNAL, a dataset for **S**emantic and **I**nferred **G**rammar **N**eurological **A**nalysis of **L**anguage. Our dataset contains a group of sentences with a combination of a fully acceptable sentence and a grammatically or/and semantically incongruent sentences. The dataset has been approved by native speakers and later used for an EEG experiment. In total, our dataset contains recordings of 11 participants, each of whom read 600 sentences. In addition, we present a pilot study where we compare EEG analysis with simple probing experiments. 

## EEG data

The preprocessed and epoched EEG data is available at https://huggingface.co/datasets/zhuravlevahana/SIGNAL/tree/main.
