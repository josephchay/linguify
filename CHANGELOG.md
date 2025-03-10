# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- initial inference pipeline
- `requirements.txt` and `setup.py` files.
- Handle different cache types under the `prepare_inputs_for_generation` method.

### Changed

- Ensure that `pad_token` and `pad_token_id` properly set when the tokenizer is loaded.
