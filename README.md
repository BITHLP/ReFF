# ReFF: Reinforcing Format Faithfulness

The source code and data for AAAI 2025 accepted paper [ReFF: Reinforcing Format Faithfulness in Language Models across Varied Tasks](https://arxiv.org/abs/2412.09173).

## Contents

- [Dataset](FormatBench): Downloading the FormatBench compressed `.tar.gz` file.

- [Format Checkers](format_check.py): Checking whether a LLM-generated response adheres to the format requirements of the corresponding task.

- [Generation](prompts): Reproducing the results with the same prompts.

- Adaptation: Improving format faithfulness of LLMs.

    - [Fine-Tuning](run_ft.py): Fine-tuning the LLM using labeled data.
    - [ReFF](run_rl.py): Reinforcing format faithfulness without compromising general quality.