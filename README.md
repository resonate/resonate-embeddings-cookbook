
# Embeddings Cookbook

This repository contains code examples for running machine learning models using Resonate Embeddings as features. It includes examples for training models with optuna and scikit-learn. We find that appropriate hyperparameter tuning is necessary to work with these data successfully, so we recommend adopting these patterns within your own stack.
This repository does not include any data.

## Overview

Resonate embeddings are a synthesis of up to 90 days of online behavior made available at the individual level. 
These embeddings may be used in isolation or in tandome with first party datasets, are may improve the predictive quality of your models, as well as their scale.

## Getting Started

### Prerequisites

Ensure you have the following Python packages installed:

- sklearn
- optuna (for hyperparameter optimization)
- Any other dependencies required by your scripts

### Installation

Clone this repository:

```bash
git clone https://github.com/resonate/resonate-embeddings-cookbook.git
cd embeddings-cookbook
```

Install the necessary Python packages:

```bash
pip install -r requirements.txt
```

## Usage

### Scikit-learn Example

To run the scikit-learn model:

```bash
python -m sklearn_cookbook.train_sklearn \
  --input-path {input_data} \
  --output-path {output_path} \
  --evkey {evkey} \
  --embeddings-path {embeddings_path} \
  --feature-selection False
```

### Example Values

- `input_path`: Path to the input data (local or S3), containing labels and IDs.
- `embeddings_path`: Path to the embeddings (local or S3), containing IDs and N-dimensional embeddings.
- `output_path`: Path to write the output data (local or S3).
- `evkey`: An example key, e.g., `E205932615`. This is a model identifier and needed for record keeping.
- `evaluations`: Number of evaluations for Optuna to explore the hyperparameter space (e.g., 150, but can be higher or lower).

## Input Requirements

### Input Path

This is a parquet file that contains the label matrix for a binary classification model. The schema for this file is:

- `rid`: ID for each data point.
- `evkey`: A model identifier to facilitate good governance of experiments and model use cases.
- `label`: Binary label indicating the outcome for the observation (e.g., this record churned or did not churn).

### Embeddings Path

This is a parquet file that contains the embeddings for a set of rids. The schema for this file is:

- `rid`: ID for each data point.
- `bottleneck`: 512-dimension embedding vector as a numpy array.

## Contributing

We welcome contributions to this project. Please submit a pull request or open an issue to discuss any changes.

## License

This project is licensed under the BSD-3-Clause License - see the LICENSE file for details.
