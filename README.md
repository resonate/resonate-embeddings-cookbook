
# Embeddings Cookbook

This repository contains code examples for running machine learning models using Resonate Embeddings as features. It includes examples for training models with LightGBM and scikit-learn.

## Getting Started

### Prerequisites

Ensure you have the following Python packages installed:

- lightgbm
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
- `evaluations`: Number of evaluations for Optuna (e.g., 150, but can be higher or lower).

## Input Requirements

### Input Path

This file contains the data for generating true churn predictions. The schema for this file is:

- `rid`: ID for each data point.
- `evkey`: Evkey for each data point.
- `label`: Binary label indicating whether there was churn or not.

### Embeddings Path

This file contains the embeddings for a set of rids. The schema for this file is:

- `rid`: ID for each data point.
- `bottleneck`: 512-dimension embedding vector as a numpy array.

## Contributing

We welcome contributions to this project. Please submit a pull request or open an issue to discuss any changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
