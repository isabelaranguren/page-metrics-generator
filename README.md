# Core Web Vitals Checker with PDF Reports

This tool analyzes website performance using Google's Lighthouse, focusing on Core Web Vitals metrics. It generates detailed PDF reports with visualizations to make performance data easier to understand.

## Features

- Analyzes Core Web Vitals metrics (LCP, CLS, TBT/FID)
- Supports both single URL and batch analysis
- Generates professional PDF reports with charts and visualizations
- Color-coded ratings based on performance thresholds
- Includes batch summary reports for multiple URLs

## Installation

1. Make sure you have Python 3.6+ installed
2. Install Lighthouse globally via npm:
   ```
   npm install -g lighthouse
   ```
3. Install required Python dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Analyzing a Single URL

```bash
python webvitals_checker.py example.com
```

### Analyzing Multiple URLs from a File

Create a text file with one URL per line, then run:

```bash
python webvitals_checker.py urls.txt --file
```

### Additional Options

- `--device` or `-d`: Choose device emulation (`desktop` or `mobile`)
- `--output-dir` or `-o`: Specify custom output directory
- `--no-pdf`: Skip PDF report generation

Examples:
```bash
# Mobile device emulation
python webvitals_checker.py example.com --device mobile

# Custom output directory
python webvitals_checker.py example.com --output-dir my_reports

# Skip PDF generation
python webvitals_checker.py example.com --no-pdf
```

## Output Files

For each analyzed URL, the tool generates:
- JSON file with complete Lighthouse data
- CSV file with summarized metrics
- PDF report with visualizations (unless `--no-pdf` is specified)

For batch analysis, it additionally creates:
- Batch summary CSV with all URLs
- Batch summary PDF report with combined data and distribution charts

## Requirements

- Python 3.6+
- Node.js 14+ (for Lighthouse)
- Lighthouse (installed globally)
- ReportLab
- Matplotlib
- NumPy

## Understanding Core Web Vitals

- **Largest Contentful Paint (LCP)**: Measures loading performance. To provide a good user experience, LCP should occur within 2.5 seconds of when the page first starts loading.
  
- **Total Blocking Time (TBT)**: Measures interactivity. A good TBT score is under 200ms (used as a proxy for First Input Delay).
  
- **Cumulative Layout Shift (CLS)**: Measures visual stability. Pages should maintain a CLS of less than 0.1.