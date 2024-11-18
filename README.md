# SIGNAL
Dataset for Semantic and Inferred Grammar Neurological Analysis of Language

## Repository structure

This repository follows the next structure:
```
├── EEG_processing                      # Source code for EEG data analysis
|   ├── z-scores_estimation             # Code for pairwise conditions comparison in EEG data
|   └── draw_plots                      # Code for visualisation
├── NN_processing                       # Source code for LLM data analysis
|   └── SIGNAL_PROBING_CHARTS           # Code for pairwise conditions comparison in LLM data and visualisation
├── README.md                           # README file
└── requirements.txt                    # A file with requirements 
```

## Dataset

In this paper, we present SIGNAL, a dataset for **S**emantic and **I**nferred **G**rammar **N**eurological **A**nalysis of **L**anguage. Our dataset contains a group of sentences with a combination of a fully acceptable sentence and a grammatically or/and semantically incongruent sentences. The dataset has been approved by native speakers and later used for an EEG experiment. In total, our dataset contains recordings of 11 participants, each of whom read 600 sentences. In addition, we present a pilot study where we compare EEG analysis with simple probing experiments. 

## EEG data

The preprocessed and epoched EEG data is available at https://huggingface.co/datasets/zhuravlevahana/SIGNAL/tree/main.
