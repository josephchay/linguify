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
- New notebook script for more detailed inference demonstration.
- `torch` dependency in `requirements.txt`.
- `IPython` dependency in `requirements.txt`.
- Text Normalization for both Chinese and English text.
- `extended_inference.ipynb` notebook for more detailed inference demonstration.
- Removed `[uv_break]` from the `inference_code` method.
- `webui.py` for the web-based user interface.
- New `cmd.py` file for command-line interface.
- `stream` parameter to `inference_code` generation pipeline.
- Apple GPU checking for device selection. 
- `HomophonesReplacer` class for chinese / mandarin characters.
- `init_homophones_replacer` method to initialize the homophones replacer in `core.py`.
- New `thumbnail2.png` image for the `README.md` file.
- Support of `torch.compile` for Windows. 

### Fixed

- Ensure that the `pad_token` and `pad_token_id` are properly set when the tokenizer is loaded.
- Ensure text passed for inference is always in a list.
- Indentation error in the `Chat::init_normalizer` method.

### Changed

- Ensure that `pad_token` and `pad_token_id` properly set when the tokenizer is loaded.
- Upgraded `huggingface_hub` dependency from 0.26.3 to 0.28.1.
- Enhanced caching logic in the `prepare_inputs_for_generation` method to efficiently manage sequence lengths.
- Initialized `past_key_alues` as `None` to optimize caching and improve token generation efficiency.
- Updated installation and development instructions in the `README.md` documentation file.
- Improved tensor operation for `DVAE` class via better feature reshaping and decoding.
- Dependencies versions in the `requirements.txt` file.

### Removed

- `nemo_text_processing` dependency, as no longer needed for text normalization.
- `__init__.py` file from the directories in the `src` directory.
- `setup.py` file since TTS is now moved to its dedicated folder.
- `**kwargs` argument from the `AudioGenerator` class.
