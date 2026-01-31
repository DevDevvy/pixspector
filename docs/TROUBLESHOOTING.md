# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### Problem: `ModuleNotFoundError: No module named 'numpy'`

**Solution:**

```bash
pip install -e .
# or
pip install -r requirements.txt
```

#### Problem: `ImportError: cannot import name 'Draft7Validator'`

**Solution:** Update jsonschema:

```bash
pip install --upgrade jsonschema
```

### Runtime Issues

#### Problem: "Permission denied creating output directory"

**Solution:**

- Check write permissions on the output directory
- Try a different output location: `pixspector analyze image.jpg --report ~/Desktop/out`
- Or run with appropriate permissions

#### Problem: "File exceeds sandbox size cap"

**Solution:**

- Use `--max-size 0` to disable size limits
- Or edit `config/defaults.yaml` to increase `sandbox.max_file_size_mb`

#### Problem: Analysis times out

**Solution:**

- Large images may timeout. Try reducing max dimension in config:
  ```yaml
  app:
    max_image_dim: 2048 # Reduce from 4096
  ```
- Or process smaller images

#### Problem: "Task failed: ai_detection"

**Possible causes:**

- Out of memory (reduce image size)
- Corrupted image file
- Missing dependencies

**Solution:**

```bash
# Check system resources
pixspector doctor

# Try with a smaller image
convert large.jpg -resize 2048x2048 smaller.jpg
pixspector analyze smaller.jpg
```

### Analysis Issues

#### Problem: High false positive rate for AI detection

**Solution:**

- AI detection is statistical and may flag real photos with unusual characteristics
- Check the detailed evidence in the JSON report
- Adjust `ai_component_gate` in config (higher = stricter)

#### Problem: C2PA verification always shows "not checked"

**Solution:**

- Install c2patool: `cargo install c2patool`
- Verify installation: `which c2patool`
- Check PATH is correct

#### Problem: PDF generation fails

**Solution:**

- Skip PDF: `pixspector analyze image.jpg --no-pdf`
- Check reportlab installation: `pip install --upgrade reportlab`
- Check write permissions

### Performance Issues

#### Problem: Batch processing is slow

**Solution:**

- Process images in smaller batches
- Reduce max_image_dim in config
- Skip PDF generation: `--no-pdf`
- Use continue-on-error: `--continue`

#### Problem: High memory usage

**Solution:**

- Reduce `sandbox.max_memory_mb` in config
- Process fewer images concurrently
- Reduce `sandbox.max_decode_pixels`

### File Format Issues

#### Problem: "Unsupported or unrecognized image format"

**Supported formats:** JPEG, PNG, TIFF, WEBP, BMP, HEIC/HEIF (if supported by your OpenCV build)

**Solution:**

- Convert to supported format: `convert input.xyz output.jpg`
- Check file extension matches content
- Verify file is not corrupted: `file image.jpg`

## Debugging Tips

### Enable Verbose Logging

Set environment variable:

```bash
export PIXSPECTOR_LOG_LEVEL=DEBUG
pixspector analyze image.jpg
```

### Check System Compatibility

```bash
pixspector doctor
```

### Validate Configuration

```bash
python -c "from pixspector.config import Config; Config.load('config/defaults.yaml')"
```

### Test with Known Good Image

```bash
# Use sample images
pixspector analyze examples/sample_images/real_photo.jpg
```

### Check JSON Report for Details

```bash
# Examine detailed analysis results
cat out/image_report.json | python -m json.tool
```

## Getting Help

1. Check this troubleshooting guide
2. Review the documentation in `docs/`
3. Check existing issues on GitHub
4. Run `pixspector doctor` for diagnostics
5. Include doctor output when reporting issues

## Reporting Bugs

When reporting bugs, please include:

- Output of `pixspector doctor`
- Command used
- Error message (full stack trace)
- Image format and size (without sharing the actual image)
- Operating system and Python version
