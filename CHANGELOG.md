# Changelog

All notable changes to pixspector will be documented in this file.

## [0.1.1] - 2026-01-31

### Added

- **Enhanced CLI**: New `--max-size` option to limit file sizes processed
- **Enhanced CLI**: New `--continue` flag for robust batch processing
- **Enhanced CLI**: Progress indicators showing N/M during batch processing
- **Enhanced CLI**: Status column in results table (✓/✗)
- **New Command**: `pixspector summarize` to analyze multiple reports
- **Better Validation**: File type validation (checks extensions before processing)
- **Better Validation**: File size pre-checks to avoid loading huge files
- **Better Validation**: Comprehensive input validation with detailed error messages
- **Improved Error Handling**: Graceful degradation when analysis modules fail
- **Improved Error Handling**: Better timeout handling (60s per module, 5min total)
- **Improved Error Handling**: Continue-on-error support for batch processing
- **Enhanced Diagnostics**: `pixspector doctor` now shows more details
- **Performance**: Optimized SHA-256 computation with larger chunks (8MB)
- **Performance**: Better concurrency management (respects CPU limits)
- **Logging**: More detailed progress logging for debugging
- **Security**: Enhanced file validation before processing
- **Security**: Better bounds checking on all numeric inputs

### Changed

- CLI now shows exit codes: 0 (success), 1 (all failed), 2 (partial success)
- Error messages are now truncated to 100 chars for better readability
- Failed analyses no longer stop the entire pipeline by default
- Improved version command with usage hints
- Config loading now validates YAML syntax and reports errors clearly

### Fixed

- Better handling of missing/corrupt image files
- Improved error messages for file permission issues
- Fixed potential race conditions in concurrent analysis
- Better cleanup when analysis is interrupted (Ctrl+C)

## [0.1.0] - Initial Release

### Added

- Core image forensics pipeline
- AI detection module
- Classical forensic analyses (ELA, JPEG ghosts, DCT/Benford, etc.)
- C2PA verification support
- Watermark detection
- Rule-based scoring system
- JSON and PDF report generation
- CLI and GUI interfaces
