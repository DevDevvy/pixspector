# PIXSPECTOR CLI Guide

## Modern, Beautiful Command-Line Interface

PIXSPECTOR now features a modern, professional CLI with enhanced visual feedback, colored output, and intuitive navigation.

## ✨ Visual Enhancements

### ASCII Logo
Every command displays the stunning PIXSPECTOR ASCII art logo with gradient cyan-to-blue coloring:

```
____  ____  __ _____ ____  _____ ______________  ____
   / __ \/  _/ |/ / ___// __ \/ ___// ____/ ____/ /_/ __ \/ __ \
  / /_/ // / |   /\__ \/ /_/ /\__ \/ __/ / /   / __/ / / / /_/ /
 / ____// / /   |___/ / ____/___/ / /___/ /___/ /_/ /_/ / _, _/
/_/   /___//_/|_/____/_/    /____/_____/\____/\__/_____/_/ |_|
```

### Color-Coded Badges
- **Suspicion Scores**: Color-coded from 0-100
  - 🟢 0-40: Green (Low risk)
  - 🟡 41-70: Yellow (Medium risk)  
  - 🔴 71-100: Red (High risk)

- **Risk Buckets**: Clearly labeled badges
  - `HIGH` - Red background
  - `MEDIUM` - Yellow background
  - `LOW` - Green background

### Section Headers
Each operation has beautiful section headers with emojis:
- 🔍 Forensic Analysis Pipeline
- 📊 Report Summary
- 🔧 System Diagnostics

## 📋 Commands Overview

### 1. Main Entry
```bash
pixspector
```
Shows the logo, tagline, and available commands in a clean, centered layout.

### 2. Version Information
```bash
pixspector version
```
Displays:
- ASCII logo with tagline
- Rounded panel with version, Python version, and platform
- Quick usage hints

### 3. System Diagnostics
```bash
pixspector doctor
```
Features:
- Comprehensive system check
- Styled table with ✓/✗ icons
- Installation help for missing dependencies
- Clean status summary panel

### 4. Image Analysis
```bash
pixspector analyze "path/to/images/*.jpg" --report out
```
Enhanced with:
- Progress indicators for each image
- Real-time status updates
- Suspicion and bucket badges in results table
- Color-coded success/error messages
- Summary panel with statistics

Options:
- `--report, -r`: Output folder (default: `out`)
- `--config, -c`: Custom YAML config
- `--no-pdf`: Skip PDF generation
- `--max-size`: Max file size in MB (default: 50)
- `--continue`: Continue on errors

### 5. Report Summarization
```bash
pixspector summarize out/
```
Displays:
- Beautiful summary table with all reports
- Average suspicion score
- Risk distribution bar chart
- Formatted statistics panel

Options:
- `--sort-by`: Sort by `suspicion`, `name`, or `bucket`

## 🎨 Styling Features

### Rich Tables
All tables use:
- Rounded borders (`box.ROUNDED`)
- Cyan color scheme
- Bold headers
- Proper column alignment
- Clean separators

### Status Icons
- ✓ Success (green)
- ✗ Error (red)
- ○ Pending (yellow)
- ▶ In Progress (cyan)

### Progress Indicators
- Real-time progress counters
- Visual bars for distributions
- Color-coded status messages

## 🖥️ GUI Application

Launch the graphical interface:
```bash
pixspector
```
Then click the GUI button, or use the Python API:
```python
from pixspector.gui.app import main
main()
```

Features:
- Drag & drop interface
- Batch processing
- Live preview of forensic artifacts
- PDF report generation
- Real-time log output

## 🎯 Quick Examples

### Analyze a single image
```bash
pixspector analyze photo.jpg
```

### Batch process with custom config
```bash
pixspector analyze "photos/*.jpg" -c custom_config.yaml
```

### Generate reports without PDFs
```bash
pixspector analyze image.png --no-pdf
```

### View all reports sorted by suspicion
```bash
pixspector summarize out/ --sort-by suspicion
```

### Check system requirements
```bash
pixspector doctor
```

## 💡 Pro Tips

1. **Quote glob patterns**: Always quote shell globs like `"*.jpg"` to avoid shell expansion
2. **Use --continue**: For batch jobs, use `--continue` to process all images even if some fail
3. **Set max-size**: Use `--max-size` to skip very large files automatically
4. **Custom configs**: Create YAML configs for different analysis profiles
5. **Check doctor**: Run `pixspector doctor` before important analysis jobs

## 🔧 Customization

The CLI styling is powered by the `pixspector.branding` module, which provides:
- `print_logo()` - Display the ASCII logo
- `get_suspicion_badge()` - Color-coded suspicion scores
- `get_bucket_badge()` - Risk level badges
- `print_section_header()` - Formatted headers

You can import these in your own scripts:
```python
from pixspector.branding import print_logo, get_suspicion_badge

print_logo()
score = 75
badge = get_suspicion_badge(score)
console.print(f"Risk: {badge}")
```

## 📚 Additional Resources

- [Architecture Guide](architecture.md) - System design overview
- [Troubleshooting](../TROUBLESHOOTING.md) - Common issues
- [Changelog](../CHANGELOG.md) - Recent improvements
- [Improvements Summary](../IMPROVEMENTS_SUMMARY.md) - What's new

---

**PIXSPECTOR** - Classical Image Forensics Toolkit
*Professional, Modern, Reliable*
