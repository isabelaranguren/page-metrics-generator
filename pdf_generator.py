#!/usr/bin/env python3
import os
import time
from datetime import datetime
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from urllib.parse import urlparse

# ReportLab imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

class PDFReportGenerator:
    """Generate PDF reports from Core Web Vitals data"""
    
    def __init__(self, output_dir="webvitals_reports"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.styles = getSampleStyleSheet()
    
        
        self.styles.add(ParagraphStyle(
            name='Subtitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            leading=18
        ))
        
        self.styles.add(ParagraphStyle(
            name='MetricGood',
            parent=self.styles['Normal'],
            textColor=colors.green
        ))
        
        self.styles.add(ParagraphStyle(
            name='MetricWarning',
            parent=self.styles['Normal'],
            textColor=colors.orange
        ))
        
        self.styles.add(ParagraphStyle(
            name='MetricPoor',
            parent=self.styles['Normal'],
            textColor=colors.red
        ))
    
    def _create_performance_gauge(self, score):
        """Create a gauge chart for performance score visualization"""
        # Create a figure with a single subplot
        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        
        # Set the limits
        ax.set_ylim(0, 10)
        ax.set_xlim(-np.pi/2, np.pi/2)
        
        # Remove grid lines and spines
        ax.grid(False)
        ax.spines['polar'].set_visible(False)
        
        # Remove tick labels
        ax.set_yticks([])
        ax.set_xticks([])
        
        # Get color based on score
        if score >= 90:
            color = 'green'
        elif score >= 50:
            color = 'orange'
        else:
            color = 'red'
        
        # Add score text
        ax.text(0, 0, f"{int(score)}", 
                fontsize=28, 
                ha='center', 
                va='center', 
                color=color,
                fontweight='bold')
        
        # Create the gauge arc
        angle = np.pi * (score / 100 - 0.5)
        theta = np.linspace(-np.pi/2, angle, 100)
        r = 8
        ax.plot(theta, [r] * len(theta), linewidth=10, color=color)
        
        # Add a circle at the end
        ax.scatter(angle, r, s=100, color=color, zorder=3)
        
        # Add placeholder ticks
        for tick in np.linspace(-np.pi/2, np.pi/2, 11):
            ax.plot([tick, tick], [7.5, 8.5], color='gray', alpha=0.3, linewidth=1)
        
        # Save to BytesIO
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        
        return buffer.getvalue()
    
    def _create_metrics_bar_chart(self, metrics_data):
        """Create a bar chart of metrics scores"""
        # Define metrics to show and their display names
        metric_display = {
            'largest-contentful-paint': 'LCP',
            'total-blocking-time': 'TBT',
            'cumulative-layout-shift': 'CLS',
            'first-contentful-paint': 'FCP',
            'time-to-interactive': 'TTI',
            'speed-index': 'SI'
        }
        
        # Get scores, labels, and colors for available metrics
        scores = []
        labels = []
        colors = []
        
        for metric, display in metric_display.items():
            if metric in metrics_data and metrics_data[metric].get('score') is not None:
                score = metrics_data[metric]['score']
                scores.append(score * 100)
                labels.append(display)
                
                # Determine color based on score
                if score >= 0.9:
                    colors.append('green')
                elif score >= 0.5:
                    colors.append('orange')
                else:
                    colors.append('red')
        
        # Skip chart creation if no data
        if not scores:
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Create bars
        bars = ax.bar(labels, scores, color=colors)
        
        # Set y-axis range and labels
        ax.set_ylim(0, 105)
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score (0-100)')
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f"{int(height)}",
                    ha='center', va='bottom')
        
        # Save chart to BytesIO
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        
        return buffer.getvalue()
    
    def generate_single_report(self, results):
        """Generate a PDF report for a single URL analysis"""
        # Check for errors in results
        if 'error' in results.get('core_web_vitals', {}):
            print(f"Error generating PDF: {results['core_web_vitals']['error']}")
            return None
        
        url = results['url']
        url_part = urlparse(url).netloc.replace('.', '_').replace(':', '_')
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        pdf_filename = os.path.join(self.output_dir, f"{url_part}_{timestamp}.pdf")
        
        # Create document
        doc = SimpleDocTemplate(
            pdf_filename,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Container for PDF elements
        elements = []
        
        # Title
        title = Paragraph(f"Core Web Vitals Report", self.styles['Title'])
        elements.append(title)
        
        # URL and Timestamp
        elements.append(Paragraph(f"<b>URL:</b> {url}", self.styles['Normal']))
        elements.append(Paragraph(f"<b>Device:</b> {results.get('device', 'desktop').capitalize()}", self.styles['Normal']))
        elements.append(Paragraph(f"<b>Date:</b> {results['timestamp']}", self.styles['Normal']))
        elements.append(Spacer(1, 12))
        
        # Performance Score
        score = results['core_web_vitals'].get('performance_score', 0)
        elements.append(Paragraph(f"Performance Score", self.styles['Subtitle']))
        
        # Create gauge image
        gauge_data = self._create_performance_gauge(score)
        if gauge_data:
            img = Image(BytesIO(gauge_data))
            img.drawHeight = 2*inch
            img.drawWidth = 2*inch
            elements.append(img)
        
        elements.append(Spacer(1, 12))
        
        # Core Web Vitals section
        elements.append(Paragraph("Core Web Vitals", self.styles['Subtitle']))
        
        # Function to get style based on rating
        def get_style(rating):
            if rating == 'Good':
                return self.styles['MetricGood']
            elif rating == 'Needs Improvement':
                return self.styles['MetricWarning']
            else:
                return self.styles['MetricPoor']
        
        # Create data for the Core Web Vitals table
        cwv = results['core_web_vitals']
        
        data = [
            ["Metric", "Value", "Score", "Rating"]
        ]
        
        # Add LCP row
        lcp = cwv.get('metrics', {}).get('largest-contentful-paint', {})
        if lcp:
            rating = lcp.get('rating', 'N/A')
            style = get_style(rating)
            data.append([
                Paragraph("Largest Contentful Paint (LCP)", self.styles['Normal']),
                Paragraph(lcp.get('formatted', 'N/A'), self.styles['Normal']),
                Paragraph(f"{lcp.get('score', 0)*100:.0f}/100", self.styles['Normal']),
                Paragraph(rating, style)
            ])
        
        # Add TBT row
        tbt = cwv.get('metrics', {}).get('total-blocking-time', {})
        if tbt:
            rating = tbt.get('rating', 'N/A')
            style = get_style(rating)
            data.append([
                Paragraph("Total Blocking Time (TBT)", self.styles['Normal']),
                Paragraph(tbt.get('formatted', 'N/A'), self.styles['Normal']),
                Paragraph(f"{tbt.get('score', 0)*100:.0f}/100", self.styles['Normal']),
                Paragraph(rating, style)
            ])
        
        # Add CLS row
        cls = cwv.get('metrics', {}).get('cumulative-layout-shift', {})
        if cls:
            rating = cls.get('rating', 'N/A')
            style = get_style(rating)
            data.append([
                Paragraph("Cumulative Layout Shift (CLS)", self.styles['Normal']),
                Paragraph(cls.get('formatted', 'N/A'), self.styles['Normal']),
                Paragraph(f"{cls.get('score', 0)*100:.0f}/100", self.styles['Normal']),
                Paragraph(rating, style)
            ])
        
        # Create the table
        t = Table(data, colWidths=[2.5*inch, 1*inch, 1*inch, 1.5*inch])
        
        # Add style to the table
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ]))
        
        elements.append(t)
        elements.append(Spacer(1, 24))
        
        # Add a metrics score graph
        elements.append(Paragraph("Metrics Scores", self.styles['Subtitle']))
        
        # Create bar chart
        chart_data = self._create_metrics_bar_chart(cwv.get('metrics', {}))
        if chart_data:
            img = Image(BytesIO(chart_data))
            img.drawHeight = 3*inch
            img.drawWidth = 6*inch
            elements.append(img)
        
        elements.append(Spacer(1, 12))
        
        # Additional Metrics
        elements.append(Paragraph("Additional Metrics", self.styles['Subtitle']))
        
        # Create data for the additional metrics table
        data = [
            ["Metric", "Value", "Score"]
        ]
        
        # Add FCP
        fcp = cwv.get('metrics', {}).get('first-contentful-paint', {})
        if fcp:
            data.append([
                "First Contentful Paint (FCP)",
                fcp.get('formatted', 'N/A'),
                f"{fcp.get('score', 0)*100:.0f}/100"
            ])
        
        # Add Speed Index
        si = cwv.get('metrics', {}).get('speed-index', {})
        if si:
            data.append([
                "Speed Index (SI)",
                si.get('formatted', 'N/A'),
                f"{si.get('score', 0)*100:.0f}/100"
            ])
        
        # Add TTI
        tti = cwv.get('metrics', {}).get('time-to-interactive', {})
        if tti:
            data.append([
                "Time to Interactive (TTI)",
                tti.get('formatted', 'N/A'),
                f"{tti.get('score', 0)*100:.0f}/100"
            ])
        
        # Create the table
        t = Table(data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
        
        # Add style to the table
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ]))
        
        elements.append(t)
        
        # Recommendations section
        elements.append(Spacer(1, 24))
        elements.append(Paragraph("Performance Recommendations", self.styles['Subtitle']))
        
        # Add recommendations based on scores
        recommendations = []
        
        if lcp and lcp.get('score', 1) < 0.9:
            recommendations.append(
                Paragraph("<b>Improve LCP:</b> Optimize largest element loading, reduce server response time, "
                          "prioritize critical resources, and implement proper image optimization.", self.styles['Normal'])
            )
        
        if tbt and tbt.get('score', 1) < 0.9:
            recommendations.append(
                Paragraph("<b>Improve TBT:</b> Reduce JavaScript execution time, split long tasks, "
                          "minimize main thread work, and optimize third-party scripts.", self.styles['Normal'])
            )
        
        if cls and cls.get('score', 1) < 0.9:
            recommendations.append(
                Paragraph("<b>Improve CLS:</b> Set size attributes on images/videos, avoid inserting content above "
                          "existing content, and use transform animations instead of layout-triggering properties.", self.styles['Normal'])
            )
        
        # If all scores are good, add a general note
        if not recommendations:
            recommendations.append(
                Paragraph("All Core Web Vitals metrics are in the good range. Continue monitoring to maintain performance.", 
                        self.styles['MetricGood'])
            )
        
        # Add recommendations to PDF
        for rec in recommendations:
            elements.append(rec)
            elements.append(Spacer(1, 6))
        
        # Build the PDF
        doc.build(elements)
        
        print(f"PDF report saved to {pdf_filename}")
        return pdf_filename
    
    def generate_batch_report(self, all_results):
        """Generate a PDF report for multiple URLs"""
        if not all_results:
            print("No results to generate a batch report")
            return None
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        pdf_filename = os.path.join(self.output_dir, f"batch_summary_{timestamp}.pdf")
        
        # Create document - landscape for batch report
        doc = SimpleDocTemplate(
            pdf_filename,
            pagesize=landscape(letter),
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        
        # Container for elements
        elements = []
        
        # Title
        title = Paragraph(f"Core Web Vitals Batch Report", self.styles['Title'])
        elements.append(title)
        
        # Timestamp
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal']))
        elements.append(Paragraph(f"Number of URLs analyzed: {len(all_results)}", self.styles['Normal']))
        elements.append(Spacer(1, 12))
        
        # Create summary table
        elements.append(Paragraph("Summary Results", self.styles['Subtitle']))
        
        # Create data for the table
        data = [
            ["URL", "Performance Score", "LCP", "TBT (FID proxy)", "CLS", "Overall Rating"]
        ]
        
        # Add data rows
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
                    style = self.styles['MetricPoor']
                elif 'Needs Improvement' in ratings:
                    overall_rating = 'Needs Improvement'
                    style = self.styles['MetricWarning']
                elif ratings and all(r == 'Good' for r in ratings):
                    overall_rating = 'Good'
                    style = self.styles['MetricGood']
                else:
                    overall_rating = 'Unknown'
                    style = self.styles['Normal']
                
                # Limit URL length for display
                display_url = result['url']
                if len(display_url) > 50:
                    display_url = display_url[:47] + "..."
                
                data.append([
                    Paragraph(display_url, self.styles['Normal']),
                    Paragraph(f"{cwv.get('performance_score', 0):.1f}", self.styles['Normal']),
                    Paragraph(cwv.get('metrics', {}).get('largest-contentful-paint', {}).get('formatted', 'N/A'), self.styles['Normal']),
                    Paragraph(cwv.get('metrics', {}).get('total-blocking-time', {}).get('formatted', 'N/A'), self.styles['Normal']),
                    Paragraph(cwv.get('metrics', {}).get('cumulative-layout-shift', {}).get('formatted', 'N/A'), self.styles['Normal']),
                    Paragraph(overall_rating, style)
                ])
            else:
                data.append([
                    Paragraph(result['url'], self.styles['Normal']),
                    Paragraph('Error', self.styles['MetricPoor']),
                    Paragraph('N/A', self.styles['Normal']),
                    Paragraph('N/A', self.styles['Normal']),
                    Paragraph('N/A', self.styles['Normal']),
                    Paragraph('Error', self.styles['MetricPoor'])
                ])
        
        # Create the table
        t = Table(data, colWidths=[4*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1*inch, 1.2*inch])
        
        # Add style to the table
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ]))
        
        elements.append(t)
        elements.append(Spacer(1, 24))
        
        # Add performance score distribution chart
        elements.append(Paragraph("Performance Score Distribution", self.styles['Subtitle']))
        
        # Collect scores for chart
        scores = []
        for result in all_results:
            if 'error' not in result.get('core_web_vitals', {}):
                scores.append(result['core_web_vitals'].get('performance_score', 0))
        
        if scores:
            # Create histogram of performance scores
            fig, ax = plt.subplots(figsize=(8, 4))
            bins = [0, 50, 90, 100]
            labels = ['Poor (0-49)', 'Needs Improvement (50-89)', 'Good (90-100)']
            
            # Count scores in each bin
            hist, _ = np.histogram(scores, bins=bins)
            
            # Create bar chart with custom colors
            bars = ax.bar(
                range(len(hist)), 
                hist, 
                tick_label=labels,
                color=['red', 'orange', 'green']
            )
            
            # Add count labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width()/2., 
                        height + 0.1, 
                        f"{int(height)}", 
                        ha='center', 
                        va='bottom'
                    )
            
            # Add labels and grid
            ax.set_ylabel('Number of URLs')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save to BytesIO
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            plt.close(fig)
            
            # Add to document
            img = Image(BytesIO(buffer.getvalue()))
            img.drawHeight = 3*inch
            img.drawWidth = 6*inch
            elements.append(img)
        
        # Core Web Vitals metrics status chart
        if len(all_results) > 1:
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("Core Web Vitals Status", self.styles['Subtitle']))
            
            # Count ratings for each metric
            status_data = {
                'LCP': {'Good': 0, 'Needs Improvement': 0, 'Poor': 0},
                'TBT': {'Good': 0, 'Needs Improvement': 0, 'Poor': 0},
                'CLS': {'Good': 0, 'Needs Improvement': 0, 'Poor': 0}
            }
            
            # Collect data
            for result in all_results:
                if 'error' not in result.get('core_web_vitals', {}):
                    metrics = result['core_web_vitals'].get('metrics', {})
                    
                    # LCP
                    lcp_rating = metrics.get('largest-contentful-paint', {}).get('rating')
                    if lcp_rating:
                        status_data['LCP'][lcp_rating] += 1
                        
                    # TBT
                    tbt_rating = metrics.get('total-blocking-time', {}).get('rating')
                    if tbt_rating:
                        status_data['TBT'][tbt_rating] += 1
                        
                    # CLS
                    cls_rating = metrics.get('cumulative-layout-shift', {}).get('rating')
                    if cls_rating:
                        status_data['CLS'][cls_rating] += 1
            
            # Create stacked bar chart
            fig, ax = plt.subplots(figsize=(8, 4))
            
            metrics = list(status_data.keys())
            good_values = [status_data[m]['Good'] for m in metrics]
            needs_improvement_values = [status_data[m]['Needs Improvement'] for m in metrics]
            poor_values = [status_data[m]['Poor'] for m in metrics]
            
            # Create bars
            bar_width = 0.6
            indices = np.arange(len(metrics))
            
            # Plot stacked bars
            p1 = ax.bar(indices, good_values, bar_width, color='green', label='Good')
            p2 = ax.bar(indices, needs_improvement_values, bar_width, bottom=good_values, color='orange', label='Needs Improvement')
            p3 = ax.bar(indices, poor_values, bar_width, bottom=[good_values[i] + needs_improvement_values[i] for i in range(len(metrics))], color='red', label='Poor')
            
            # Customize chart
            ax.set_ylabel('Number of URLs')
            ax.set_xlabel('Core Web Vitals Metrics')
            ax.set_title('Distribution of Metrics Ratings')
            ax.set_xticks(indices)
            ax.set_xticklabels(metrics)
            ax.legend()
            
            # Add value labels
            for i, metric in enumerate(metrics):
                total = sum(status_data[metric].values())
                if total > 0:
                    # Calculate percentages
                    good_pct = good_values[i] / total * 100
                    needs_pct = needs_improvement_values[i] / total * 100
                    poor_pct = poor_values[i] / total * 100
                    
                    # Add labels if values are significant
                    if good_values[i] > 0:
                        ax.text(i, good_values[i]/2, f"{good_pct:.0f}%", ha='center', va='center', color='white', fontweight='bold')
                    
                    if needs_improvement_values[i] > 0:
                        ax.text(i, good_values[i] + needs_improvement_values[i]/2, f"{needs_pct:.0f}%", ha='center', va='center', color='white', fontweight='bold')
                    
                    if poor_values[i] > 0:
                        ax.text(i, good_values[i] + needs_improvement_values[i] + poor_values[i]/2, f"{poor_pct:.0f}%", ha='center', va='center', color='white', fontweight='bold')
            
            # Save to BytesIO
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            plt.close(fig)
            
            # Add to document
            img = Image(BytesIO(buffer.getvalue()))
            img.drawHeight = 3*inch
            img.drawWidth = 6*inch
            elements.append(img)
        
        # Add recommendations section
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Core Web Vitals Recommendations", self.styles['Subtitle']))
        
        recommendations = [
            "<b>Largest Contentful Paint (LCP)</b>: Optimize largest content elements such as images, videos, or large blocks of text. Consider server response time, resource load time, and client-side rendering.",
            "<b>Total Blocking Time (TBT)</b>: Reduce JavaScript execution time, break up long tasks, optimize your JavaScript and minimize main thread work.",
            "<b>Cumulative Layout Shift (CLS)</b>: Always include size attributes on images and videos, avoid inserting content above existing content, and use transform animations instead of animations that trigger layout changes."
        ]
        
        for rec in recommendations:
            elements.append(Paragraph(rec, self.styles['Normal']))
            elements.append(Spacer(1, 6))
        
        # Build the PDF
        doc.build(elements)
        
        return pdf_filename