# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- initial inference pipeline
- `requirements.txt` and `setup.py` files.
- Handle different cache types under the `prepare_inputs_for_generation` method.
- Add `inference.ipynb` notebook to demonstrate usage of the inference pipeline.
- New utility functions for IO.
- Add `local_path` system and `force_redownload` parameter to the `load_models` method.
- New `refine_text_only` parameter to the `inference` method of `Chat` class.
- `sample_random_speaker` method to sample a random speaker from the dataset.
- UI for the inference pipeline demonstration.
- Changed the random speaker sampling the `generate_audio` method to use `sample_random_speaker`.
- Sample audio output assets in the new `assets` directory.
- `Sample Audio Outputs` section in the `README.md` file.
- Check for any invalid characters found in the given input text during inference.
- Text normalization with `nemo_text_processing`
- Language detection.
- Default speed for inference.

### Fixed

- Ensure that the `pad_token` and `pad_token_id` are properly set when the tokenizer is loaded.
- Ensure text passed for inference is always in a list.

### Changed

- Ensure that `pad_token` and `pad_token_id` properly set when the tokenizer is loaded.
- Upgraded `huggingface_hub` dependency from 0.26.3 to 0.28.1.

### Removed

- `nemo_text_processing` dependency, as no longer needed for text normalization.
