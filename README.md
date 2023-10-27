# LLM Feature Extractor

This is a simple feature extractor that uses LLMs to obtain tabular data in CSV format from free-form text.

This project also includes experiments with different types of XAI strategies and packages for the well-known Kaggle Restaurant dataset.

## Installation

```bash
poetry install --with dev
poe install-torch
poe install-transformers
poe install-captum
```

## Usage Example

```bash
python -m llm-feature-extractor.text_to_csv -t text.txt --features city_type P6 P9
```

This command should return something along the lines of:


```python
[[{'generated_text': 'city_type,P6,P9\nBig Cities,2,4'}]]
```

The package uses Dolly-V2-3B by default. To change the model you can switch it using `-m` flag. You need to provide a valid HuggingFace ID of a model for this to work.

Models that we've tried so far that had reasonably good performance at this task:
- `databricks/dolly-v2-12b`
- `databricks/dolly-v2-3b`

Models that we want to try but require fine-tuning on a few samples first:
- `mistralai/Mistral-7B-Instruct-v0.1`
