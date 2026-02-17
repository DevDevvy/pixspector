# Changelog

All notable changes to **pixspector** are documented in this file.

## [Unreleased]

### Added
- Replaced watermark detector stubs with working heuristic detectors for SynthID, Stable Signature, and Adobe CC style periodic watermark signals.
- Added `tests/test_watermark.py` to validate detector behavior and confidence outputs.

### Changed
- Updated project documentation to explain watermark and AI-origin modules as probabilistic provenance signals.
- Clarified architecture notes around watermark confidence interpretation.
