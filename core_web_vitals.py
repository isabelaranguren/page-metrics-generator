#!/usr/bin/env python3
"""
Core Web Vitals Checker - Analyzes performance metrics using Lighthouse
with PDF report generation
"""

import json
import csv
import os
import sys
import time
import subprocess
import argparse
from urllib.parse import urlparse
import tempfile

# Constants
REPORTS_DIR = "reports"

# Import the PDF report generator from a separate file
try:
    from pdf_generator import PDFReportGenerator
except ImportError:
    print("PDF generator module not found. Creating it...")
    # We'll create the file later, so this is expected on first run

class LighthouseRunner:
    """Runs Lighthouse to get Core Web Vitals metrics"""
    
    def __init__(self, device='desktop'):
        self.device = device.lower()
        if not os.path.exists(REPORTS_DIR):
            os.makedirs(REPORTS_DIR)
    
    def is_lighthouse_installed(self):
        """Check if Lighthouse is installed"""
        try:
            subprocess.run(
                ['lighthouse', '--version'], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=True
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def run_lighthouse(self, url):
        """Run Lighthouse and return Core Web Vitals metrics"""
        print(f"Running Lighthouse performance audit for {url}...")
        
        # First check if Lighthouse is installed
        if not self.is_lighthouse_installed():
            print("WARNING: Lighthouse is not installed.")
            print("To install Lighthouse, run: npm install -g lighthouse")
            return {
                'error': 'Lighthouse not installed'
            }
        
        # Create temporary file for report
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp:
            output_path = temp.name
        
        try:
            # Build Lighthouse command
            lighthouse_cmd = [
                'lighthouse',
                url,
                '--output=json',
                '--output-path', output_path,
                '--chrome-flags=--headless',
                f'--preset={self.device}',
                '--only-categories=performance'
            ]
            
            # Run Lighthouse
            process = subprocess.run(
                lighthouse_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False  # Don't raise exception on non-zero exit
            )
            
            # Check for errors
            if process.returncode != 0:
                error_msg = process.stderr.strip()
                print(f"Lighthouse error: {error_msg}")
                return {
                    'error': f"Lighthouse failed: {error_msg}"
                }
            
            # Read the report
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            # Extract Core Web Vitals metrics
            audits = data.get('audits', {})
            metrics = data.get('audits', {}).get('metrics', {}).get('details', {}).get('items', [{}])[0]
            
            # Create results dictionary
            results = {
                'performance_score': data.get('categories', {}).get('performance', {}).get('score', 0) * 100,
                'metrics': {
                    'first-contentful-paint': {
                        'value': audits.get('first-contentful-paint', {}).get('numericValue'),
                        'score': audits.get('first-contentful-paint', {}).get('score')
                    },
                    'largest-contentful-paint': {
                        'value': audits.get('largest-contentful-paint', {}).get('numericValue'),
                        'score': audits.get('largest-contentful-paint', {}).get('score')
                    },
                    'total-blocking-time': {
                        'value': audits.get('total-blocking-time', {}).get('numericValue'),
                        'score': audits.get('total-blocking-time', {}).get('score')
                    },
                    'cumulative-layout-shift': {
                        'value': audits.get('cumulative-layout-shift', {}).get('numericValue'),
                        'score': audits.get('cumulative-layout-shift', {}).get('score')
                    },
                    'speed-index': {
                        'value': audits.get('speed-index', {}).get('numericValue'),
                        'score': audits.get('speed-index', {}).get('score')
                    },
                    'time-to-interactive': {
                        'value': audits.get('interactive', {}).get('numericValue'),
                        'score': audits.get('interactive', {}).get('score')
                    }
                },
                'raw_metrics': metrics
            }
            
            # Format values for display
            for metric in results['metrics']:
                value = results['metrics'][metric]['value']
                if value is not None:
                    if metric == 'cumulative-layout-shift':
                        results['metrics'][metric]['formatted'] = f"{value:.3f}"
                    elif metric in ['first-contentful-paint', 'largest-contentful-paint', 'speed-index', 'time-to-interactive']:
                        results['metrics'][metric]['formatted'] = f"{value/1000:.2f}s"
                    else:
                        results['metrics'][metric]['formatted'] = f"{value:.0f}ms"
            
            # Add ratings based on thresholds
            if results['metrics']['largest-contentful-paint']['value'] is not None:
                lcp = results['metrics']['largest-contentful-paint']['value']
                if lcp <= 2500:
                    results['metrics']['largest-contentful-paint']['rating'] = 'Good'
                elif lcp <= 4000:
                    results['metrics']['largest-contentful-paint']['rating'] = 'Needs Improvement'
                else:
                    results['metrics']['largest-contentful-paint']['rating'] = 'Poor'
            
            if results['metrics']['total-blocking-time']['value'] is not None:
                tbt = results['metrics']['total-blocking-time']['value']
                # TBT as FID proxy
                if tbt <= 200:
                    results['metrics']['total-blocking-time']['rating'] = 'Good'
                elif tbt <= 600:
                    results['metrics']['total-blocking-time']['rating'] = 'Needs Improvement'
                else:
                    results['metrics']['total-blocking-time']['rating'] = 'Poor'
            
            if results['metrics']['cumulative-layout-shift']['value'] is not None:
                cls = results['metrics']['cumulative-layout-shift']['value']
                if cls <= 0.1:
                    results['metrics']['cumulative-layout-shift']['rating'] = 'Good'
                elif cls <= 0.25:
                    results['metrics']['cumulative-layout-shift']['rating'] = 'Needs Improvement'
                else:
                    results['metrics']['cumulative-layout-shift']['rating'] = 'Poor'
            
            return results
            
        except Exception as e:
            print(f"Error running Lighthouse: {e}")
            return {
                'error': str(e)
            }
        finally:
            # Clean up temporary file
            if os.path.exists(output_path):
                os.remove(output_path)

def load_urls_from_file(filename):
    """Load URLs from a file and normalize them"""
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return []
    
    urls = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue  # Skip empty lines and comments
                
                # Add https:// if missing
                if not line.startswith(('http://', 'https://')):
                    line = 'https://' + line
                
                # Basic URL validation
                try:
                    result = urlparse(line)
                    if all([result.scheme, result.netloc]):
                        urls.append(line)
                    else:
                        print(f"Warning: Invalid URL skipped: '{line}'")
                except Exception:
                    print(f"Warning: Invalid URL skipped: '{line}'")
    except Exception as e:
        print(f"Error reading file: {e}")
    
    return urls

def analyze_url(url, device='desktop'):
    """Analyze a single URL for Core Web Vitals"""
    result = {
        'url': url,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': device,
        'core_web_vitals': None
    }
    
    # Run Lighthouse
    lighthouse = LighthouseRunner(device)
    result['core_web_vitals'] = lighthouse.run_lighthouse(url)
    
    return result

def save_results(results, output_dir=REPORTS_DIR, generate_pdf=True):
    """Save analysis results to files"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    url_part = urlparse(results['url']).netloc.replace('.', '_').replace(':', '_')
    
    # Save full JSON report
    json_filename = os.path.join(output_dir, f"{url_part}_{timestamp}.json")
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save CSV summary
    csv_filename = os.path.join(output_dir, f"{url_part}_{timestamp}.csv")
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        header = ['URL', 'Performance Score', 'LCP', 'TBT (FID proxy)', 'CLS', 'FCP', 'TTI', 'Speed Index']
        writer.writerow(header)
        
        # Write data
        if 'error' not in results.get('core_web_vitals', {}):
            cwv = results['core_web_vitals']
            row = [
                results['url'],
                f"{cwv.get('performance_score', 0):.1f}",
                cwv.get('metrics', {}).get('largest-contentful-paint', {}).get('formatted', 'N/A'),
                cwv.get('metrics', {}).get('total-blocking-time', {}).get('formatted', 'N/A'),
                cwv.get('metrics', {}).get('cumulative-layout-shift', {}).get('formatted', 'N/A'),
                cwv.get('metrics', {}).get('first-contentful-paint', {}).get('formatted', 'N/A'),
                cwv.get('metrics', {}).get('time-to-interactive', {}).get('formatted', 'N/A'),
                cwv.get('metrics', {}).get('speed-index', {}).get('formatted', 'N/A')
            ]
        else:
            row = [results['url'], 'Error', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']
        
        writer.writerow(row)
    
    print(f"Report saved to {json_filename}")
    print(f"Summary saved to {csv_filename}")
    
    # Generate PDF report if requested
    pdf_filename = None
    if generate_pdf:
        try:
            pdf_generator = PDFReportGenerator(output_dir)
            pdf_filename = pdf_generator.generate_single_report(results)
        except ImportError:
            print("PDF generation skipped - required modules not installed")
            print("Install with: pip install reportlab matplotlib")
    
    return json_filename, csv_filename, pdf_filename

def print_results_summary(results):
    """Print a summary of the results to the console"""
    print("\n===== CORE WEB VITALS RESULTS =====")
    print(f"URL: {results['url']}")
    
    # Core Web Vitals Summary
    if 'error' in results.get('core_web_vitals', {}):
        print(f"Error: {results['core_web_vitals']['error']}")
    else:
        cwv = results['core_web_vitals']
        print(f"\nLighthouse Performance Score: {cwv.get('performance_score', 0):.1f}/100")
        
        # LCP
        lcp = cwv.get('metrics', {}).get('largest-contentful-paint', {})
        print(f"\nLargest Contentful Paint (LCP): {lcp.get('formatted', 'N/A')}")
        if 'rating' in lcp:
            rating = lcp['rating']
            if rating == 'Good':
                print(f"  Rating: ✅ {rating}")
            elif rating == 'Needs Improvement':
                print(f"  Rating: ⚠️ {rating}")
            else:
                print(f"  Rating: ❌ {rating}")
        
        # TBT (FID proxy)
        tbt = cwv.get('metrics', {}).get('total-blocking-time', {})
        print(f"\nTotal Blocking Time (FID proxy): {tbt.get('formatted', 'N/A')}")
        if 'rating' in tbt:
            rating = tbt['rating']
            if rating == 'Good':
                print(f"  Rating: ✅ {rating} (as FID proxy)")
            elif rating == 'Needs Improvement':
                print(f"  Rating: ⚠️ {rating} (as FID proxy)")
            else:
                print(f"  Rating: ❌ {rating} (as FID proxy)")
        
        # CLS
        cls = cwv.get('metrics', {}).get('cumulative-layout-shift', {})
        print(f"\nCumulative Layout Shift (CLS): {cls.get('formatted', 'N/A')}")
        if 'rating' in cls:
            rating = cls['rating']
            if rating == 'Good':
                print(f"  Rating: ✅ {rating}")
            elif rating == 'Needs Improvement':
                print(f"  Rating: ⚠️ {rating}")
            else:
                print(f"  Rating: ❌ {rating}")
        
        # Other metrics
        print(f"\nAdditional Metrics:")
        print(f"  First Contentful Paint (FCP): {cwv.get('metrics', {}).get('first-contentful-paint', {}).get('formatted', 'N/A')}")
        print(f"  Time to Interactive (TTI): {cwv.get('metrics', {}).get('time-to-interactive', {}).get('formatted', 'N/A')}")
        print(f"  Speed Index: {cwv.get('metrics', {}).get('speed-index', {}).get('formatted', 'N/A')}")

def batch_analyze_urls(urls, device='desktop', output_dir=REPORTS_DIR, generate_pdf=True):
    """Analyze multiple URLs and save results"""
    print(f"Analyzing {len(urls)} URLs...")
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Results for batch summary
    all_results = []
    
    # Process each URL
    for url in urls:
        print(f"\n--- Processing {url} ---")
        try:
            results = analyze_url(url, device)
            save_results(results, output_dir, generate_pdf=False)  # Skip individual PDFs for batch
            print_results_summary(results)
            all_results.append(results)
        except Exception as e:
            print(f"Error analyzing {url}: {e}")
    
    # Create batch summary CSV
    if all_results:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        batch_csv = os.path.join(output_dir, f"batch_summary_{timestamp}.csv")
        
        with open(batch_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            header = ['URL', 'Performance Score', 'LCP', 'TBT (FID proxy)', 'CLS', 'Rating']
            writer.writerow(header)
            
            # Write data for each URL
            for result in all_results:
                if 'error' not in result.get('core_web_vitals', {}):
                    cwv = result['core_web_vitals']
                    
                    # Determine overall rating
                    metrics = cwv.get('metrics', {})
                    ratings = []
                    
                    for metric in ['largest-contentful-paint', 'total-blocking-time', 'cumulative-layout-shift']:
                        if 'rating' in metrics.get(metric, {}):
                            ratings.append(metrics[metric]['rating'])
                    
                    if 'Poor' in ratings:
                        overall_rating = 'Poor'
                    elif 'Needs Improvement' in ratings:
                        overall_rating = 'Needs Improvement'
                    elif ratings and all(r == 'Good' for r in ratings):
                        overall_rating = 'Good'
                    else:
                        overall_rating = 'Unknown'
                    
                    row = [
                        result['url'],
                        f"{cwv.get('performance_score', 0):.1f}",
                        cwv.get('metrics', {}).get('largest-contentful-paint', {}).get('formatted', 'N/A'),
                        cwv.get('metrics', {}).get('total-blocking-time', {}).get('formatted', 'N/A'),
                        cwv.get('metrics', {}).get('cumulative-layout-shift', {}).get('formatted', 'N/A'),
                        overall_rating
                    ]
                else:
                    row = [result['url'], 'Error', 'N/A', 'N/A', 'N/A', 'Error']
                
                writer.writerow(row)
        
        print(f"\nBatch summary saved to {batch_csv}")
        
        # Generate PDF batch report if requested
        if generate_pdf:
            try:
                pdf_generator = PDFReportGenerator(output_dir)
                pdf_filename = pdf_generator.generate_batch_report(all_results)
                print(f"Batch PDF report saved to {pdf_filename}")
            except ImportError:
                print("PDF generation skipped - required modules not installed")
                print("Install with: pip install reportlab matplotlib")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Core Web Vitals Checker - Analyzes performance metrics using Lighthouse')
    parser.add_argument('input', help='URL to analyze or file containing URLs (with --file)')
    parser.add_argument('--file', '-f', action='store_true', help='Read URLs from a file (one per line)')
    parser.add_argument('--device', '-d', choices=['desktop', 'mobile'], default='desktop', help='Device to emulate for Lighthouse')
    parser.add_argument('--output-dir', '-o', default=REPORTS_DIR, help='Output directory for reports')
    parser.add_argument('--no-pdf', action='store_true', help='Skip PDF report generation')
    
    args = parser.parse_args()
    generate_pdf = not args.no_pdf
    
    # Process file or single URL
    if args.file:
        urls = load_urls_from_file(args.input)
        if urls:
            batch_analyze_urls(
                urls, 
                device=args.device,
                output_dir=args.output_dir,
                generate_pdf=generate_pdf
            )
        else:
            print(f"No valid URLs found in {args.input}")
    else:
        # Single URL analysis
        url = args.input
        # Add https:// if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Analyze URL
        try:
            results = analyze_url(
                url, 
                device=args.device
            )
            save_results(results, args.output_dir, generate_pdf=generate_pdf)
            print_results_summary(results)
        except Exception as e:
            print(f"Error analyzing {url}: {e}")

if __name__ == "__main__":
    main()