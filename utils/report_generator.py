# Report Generation Module for Medical X-ray AI System

# Optional imports for report generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
import streamlit as st
from PIL import Image
import io
import base64
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import tempfile
import os

def get_report_settings():
    """Get report settings from session state or defaults"""
    try:
        if 'settings_manager' in st.session_state:
            settings = st.session_state.settings_manager.load_settings()
            return settings.get('reports', {})
    except:
        pass
    
    # Return default report settings
    return {
        'include_metadata': True,
        'include_preprocessing_info': False,
        'include_gradcam': True,
        'default_format': 'PDF',
        'auto_download': False,
        'compress_reports': True
    }

class MedicalReportGenerator:
    """Generate professional medical reports for X-ray analysis results"""
    
    def __init__(self):
        if REPORTLAB_AVAILABLE:
            self.styles = getSampleStyleSheet()
            self.setup_custom_styles()
        else:
            self.styles = None
        
        # Load report settings
        self.report_settings = get_report_settings()
    
    def setup_custom_styles(self):
        """Setup custom styles for the medical report"""
        if not REPORTLAB_AVAILABLE:
            return
            
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#2E86AB'),
            spaceAfter=30,
            alignment=1  # Center alignment
        ))
        
        # Header style
        self.styles.add(ParagraphStyle(
            name='CustomHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#A23B72'),
            spaceAfter=12,
            spaceBefore=20
        ))
        
        # Body style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            leftIndent=20
        ))
        
        # Findings style
        self.styles.add(ParagraphStyle(
            name='Findings',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#2E86AB'),
            spaceAfter=10,
            leftIndent=10,
            fontName='Helvetica-Bold'
        ))

def generate_pdf_report(results: Dict[str, Any]) -> bytes:
    """
    Generate PDF report for X-ray analysis results
    
    Args:
        results: Dictionary containing analysis results
        
    Returns:
        bytes: PDF report data
    """
    if not REPORTLAB_AVAILABLE:
        st.error("PDF generation not available. Please install reportlab: pip install reportlab")
        return generate_simple_pdf_report(results)
        
    try:
        # Create PDF buffer
        buffer = io.BytesIO()
        
        # Create PDF document
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        
        # Initialize report generator
        report_gen = MedicalReportGenerator()
        
        # Title
        title = Paragraph("Medical X-ray AI Analysis Report", report_gen.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Report metadata
        metadata_data = [
            ['Report Generated:', results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')],
            ['Analysis Type:', results.get('type', 'Unknown').title()],
            ['Model Used:', results.get('model_used', 'Unknown')],
            ['Report ID:', f"XR-{results['timestamp'].strftime('%Y%m%d%H%M%S')}"]
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 3*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f2f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(metadata_table)
        story.append(Spacer(1, 20))
        
        # Analysis Results Section
        results_header = Paragraph("Analysis Results", report_gen.styles['CustomHeader'])
        story.append(results_header)
        
        # Main findings
        prediction_text = f"<b>Diagnosis:</b> {results['prediction']}"
        confidence_text = f"<b>Confidence Level:</b> {results['confidence']:.1%}"
        
        findings_para = Paragraph(prediction_text, report_gen.styles['Findings'])
        confidence_para = Paragraph(confidence_text, report_gen.styles['CustomBody'])
        
        story.append(findings_para)
        story.append(confidence_para)
        story.append(Spacer(1, 15))
        
        # Confidence interpretation
        confidence_interpretation = get_confidence_interpretation(results['confidence'])
        interp_para = Paragraph(f"<b>Interpretation:</b> {confidence_interpretation}", 
                               report_gen.styles['CustomBody'])
        story.append(interp_para)
        story.append(Spacer(1, 20))
        
        # Images Section
        images_header = Paragraph("Image Analysis", report_gen.styles['CustomHeader'])
        story.append(images_header)
        
        # Save original image temporarily
        original_img_path = save_temp_image(results['original_image'], 'original')
        if original_img_path:
            original_img = RLImage(original_img_path, width=3*inch, height=3*inch)
            story.append(original_img)
            story.append(Paragraph("Original X-ray Image", report_gen.styles['CustomBody']))
            story.append(Spacer(1, 10))
        
        # Add Grad-CAM image if available
        if 'gradcam' in results and results['gradcam']:
            gradcam_path = save_temp_image(results['gradcam'], 'gradcam')
            if gradcam_path:
                story.append(Spacer(1, 10))
                gradcam_img = RLImage(gradcam_path, width=3*inch, height=3*inch)
                story.append(gradcam_img)
                story.append(Paragraph("Grad-CAM Visualization (Areas of AI Focus)", 
                                     report_gen.styles['CustomBody']))
        
        # Clinical Recommendations
        recommendations_header = Paragraph("Clinical Recommendations", report_gen.styles['CustomHeader'])
        story.append(recommendations_header)
        
        recommendations = generate_clinical_recommendations(results)
        for rec in recommendations:
            rec_para = Paragraph(f"• {rec}", report_gen.styles['CustomBody'])
            story.append(rec_para)
        
        story.append(Spacer(1, 20))
        
        # Disclaimer
        disclaimer_header = Paragraph("Important Disclaimer", report_gen.styles['CustomHeader'])
        story.append(disclaimer_header)
        
        disclaimer_text = """
        This report is generated by an AI system for educational and research purposes only. 
        The results should be interpreted by qualified medical professionals and should not 
        be used as the sole basis for clinical decisions. Always consult with a radiologist 
        or appropriate medical specialist for definitive diagnosis and treatment recommendations.
        """
        
        disclaimer_para = Paragraph(disclaimer_text, report_gen.styles['CustomBody'])
        story.append(disclaimer_para)
        
        # Build PDF
        doc.build(story)
        
        # Clean up temp files
        cleanup_temp_files()
        
        # Return PDF data
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        st.error(f"Error generating PDF report: {str(e)}")
        return generate_simple_pdf_report(results)

def generate_simple_pdf_report(results: Dict[str, Any]) -> bytes:
    """
    Generate a simple text-based PDF report when ReportLab is not available
    
    Args:
        results: Dictionary containing analysis results
        
    Returns:
        bytes: Simple PDF report data
    """
    try:
        # Create a simple text report
        report_text = f"""
MEDICAL X-RAY AI ANALYSIS REPORT
================================

Report Generated: {results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
Analysis Type: {results.get('type', 'Unknown').title()}
Model Used: {results.get('model_used', 'Unknown')}
Report ID: XR-{results['timestamp'].strftime('%Y%m%d%H%M%S')}

ANALYSIS RESULTS
================
Diagnosis: {results['prediction']}
Confidence Level: {results['confidence']:.1%}
Interpretation: {get_confidence_interpretation(results['confidence'])}

CLINICAL RECOMMENDATIONS
========================
"""
        
        recommendations = generate_clinical_recommendations(results)
        for i, rec in enumerate(recommendations, 1):
            report_text += f"{i}. {rec}\n"
        
        report_text += """

IMPORTANT DISCLAIMER
====================
This report is generated by an AI system for educational and research purposes only. 
The results should be interpreted by qualified medical professionals and should not 
be used as the sole basis for clinical decisions. Always consult with a radiologist 
or appropriate medical specialist for definitive diagnosis and treatment recommendations.
"""
        
        # For now, return the text as bytes (in production, you'd use a PDF library)
        return report_text.encode('utf-8')
        
    except Exception as e:
        st.error(f"Error generating simple PDF report: {str(e)}")
        return b"Error generating report"

def generate_html_report(results: Dict[str, Any]) -> str:
    """
    Generate HTML report for X-ray analysis results
    
    Args:
        results: Dictionary containing analysis results
        
    Returns:
        str: HTML report content
    """
    try:
        # Convert images to base64 for embedding
        original_img_b64 = image_to_base64(results['original_image'])
        gradcam_img_b64 = ""
        
        if 'gradcam' in results and results['gradcam']:
            gradcam_img_b64 = image_to_base64(results['gradcam'])
        
        # Get clinical recommendations
        recommendations = generate_clinical_recommendations(results)
        recommendations_html = "".join([f"<li>{rec}</li>" for rec in recommendations])
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Medical X-ray AI Analysis Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    color: #333;
                }}
                
                .header {{
                    text-align: center;
                    color: #2E86AB;
                    border-bottom: 3px solid #2E86AB;
                    padding-bottom: 10px;
                    margin-bottom: 30px;
                }}
                
                .section {{
                    margin-bottom: 30px;
                }}
                
                .section h2 {{
                    color: #A23B72;
                    border-left: 4px solid #A23B72;
                    padding-left: 10px;
                }}
                
                .metadata-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                
                .metadata-table th, .metadata-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                
                .metadata-table th {{
                    background-color: #f0f2f6;
                    font-weight: bold;
                }}
                
                .prediction-box {{
                    background-color: #f8f9fa;
                    border-left: 5px solid #2E86AB;
                    padding: 15px;
                    margin: 20px 0;
                }}
                
                .confidence-high {{ color: #28a745; }}
                .confidence-medium {{ color: #ffc107; }}
                .confidence-low {{ color: #dc3545; }}
                
                .image-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                
                .image-container img {{
                    max-width: 300px;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                }}
                
                .disclaimer {{
                    background-color: #fff3cd;
                    border: 1px solid #ffeaa7;
                    border-radius: 5px;
                    padding: 15px;
                    margin-top: 30px;
                }}
                
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Medical X-ray AI Analysis Report</h1>
                <p>Automated Medical Image Analysis Report</p>
            </div>
            
            <div class="section">
                <h2>Report Information</h2>
                <table class="metadata-table">
                    <tr>
                        <th>Report Generated</th>
                        <td>{results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</td>
                    </tr>
                    <tr>
                        <th>Analysis Type</th>
                        <td>{results.get('type', 'Unknown').title()}</td>
                    </tr>
                    <tr>
                        <th>Model Used</th>
                        <td>{results.get('model_used', 'Unknown')}</td>
                    </tr>
                    <tr>
                        <th>Report ID</th>
                        <td>XR-{results['timestamp'].strftime('%Y%m%d%H%M%S')}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Analysis Results</h2>
                <div class="prediction-box">
                    <h3>Diagnosis: {results['prediction']}</h3>
                    <p><strong>Confidence Level:</strong> 
                        <span class="{get_confidence_class(results['confidence'])}">
                            {results['confidence']:.1%}
                        </span>
                    </p>
                    <p><strong>Interpretation:</strong> {get_confidence_interpretation(results['confidence'])}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Image Analysis</h2>
                <div class="image-container">
                    <img src="data:image/png;base64,{original_img_b64}" alt="Original X-ray Image">
                    <p><strong>Original X-ray Image</strong></p>
                </div>
                
                {f'''
                <div class="image-container">
                    <img src="data:image/png;base64,{gradcam_img_b64}" alt="Grad-CAM Visualization">
                    <p><strong>Grad-CAM Visualization</strong><br>
                    <em>Red areas indicate regions the AI model focused on for diagnosis</em></p>
                </div>
                ''' if gradcam_img_b64 else ''}
            </div>
            
            <div class="section">
                <h2>Clinical Recommendations</h2>
                <ul>
                    {recommendations_html}
                </ul>
            </div>
            
            <div class="disclaimer">
                <h3>Important Disclaimer</h3>
                <p>This report is generated by an AI system for educational and research purposes only. 
                The results should be interpreted by qualified medical professionals and should not 
                be used as the sole basis for clinical decisions. Always consult with a radiologist 
                or appropriate medical specialist for definitive diagnosis and treatment recommendations.</p>
            </div>
            
            <div class="footer">
                <p>Generated by Medical X-ray AI Classification System</p>
                <p>Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """
        
        return html_content
        
    except Exception as e:
        st.error(f"Error generating HTML report: {str(e)}")
        return "<html><body><h1>Error generating report</h1></body></html>"

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    try:
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    except:
        return ""

def save_temp_image(image: Image.Image, prefix: str) -> Optional[str]:
    """Save image to temporary file and return path"""
    try:
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"{prefix}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
        image.save(temp_path)
        return temp_path
    except:
        return None

def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        temp_dir = tempfile.gettempdir()
        for filename in os.listdir(temp_dir):
            if filename.startswith(('original_', 'gradcam_')) and filename.endswith('.png'):
                try:
                    os.remove(os.path.join(temp_dir, filename))
                except:
                    pass  # Ignore errors during cleanup
    except:
        pass

def get_confidence_interpretation(confidence: float) -> str:
    """Get interpretation text for confidence level"""
    if confidence > 0.9:
        return "Very high confidence. The AI model is very certain about this diagnosis."
    elif confidence > 0.8:
        return "High confidence. The prediction is likely accurate but clinical review is recommended."
    elif confidence > 0.6:
        return "Moderate confidence. Additional examination or expert review may be beneficial."
    elif confidence > 0.4:
        return "Low confidence. Manual review by a qualified professional is strongly recommended."
    else:
        return "Very low confidence. This result should not be relied upon without expert evaluation."

def get_confidence_class(confidence: float) -> str:
    """Get CSS class for confidence level"""
    if confidence > 0.7:
        return "confidence-high"
    elif confidence > 0.5:
        return "confidence-medium"
    else:
        return "confidence-low"

def generate_clinical_recommendations(results: Dict[str, Any]) -> list:
    """Generate clinical recommendations based on analysis results"""
    recommendations = []
    
    prediction = results['prediction']
    confidence = results['confidence']
    analysis_type = results.get('type', 'unknown')
    
    # General recommendations based on confidence
    if confidence < 0.6:
        recommendations.append("Low confidence prediction - recommend manual review by radiologist")
        recommendations.append("Consider additional imaging or alternative diagnostic methods")
    
    # Specific recommendations based on diagnosis type
    if analysis_type == 'bone':
        if 'Fracture' in prediction:
            recommendations.append("Consider orthopedic consultation for fracture management")
            recommendations.append("Evaluate need for immobilization or surgical intervention")
            recommendations.append("Follow-up imaging may be required to monitor healing")
        else:
            recommendations.append("No fracture detected, but clinical correlation is advised")
            recommendations.append("Consider other causes if patient symptoms persist")
    
    elif analysis_type == 'chest':
        if 'Pneumonia' in prediction:
            recommendations.append("Consider antibiotic therapy based on clinical presentation")
            recommendations.append("Monitor patient response and consider follow-up imaging")
            recommendations.append("Evaluate for complications if symptoms are severe")
        elif 'Cardiomegaly' in prediction:
            recommendations.append("Recommend cardiology consultation for further evaluation")
            recommendations.append("Consider echocardiogram to assess cardiac function")
            recommendations.append("Evaluate for underlying cardiac conditions")
        else:
            recommendations.append("Normal chest X-ray findings")
            recommendations.append("Clinical correlation recommended if symptoms present")
    
    elif analysis_type == 'knee':
        if 'Osteoporosis' in prediction:
            recommendations.append("Consider bone density evaluation (DEXA scan)")
            recommendations.append("Evaluate for fracture risk and fall prevention")
            recommendations.append("Consider calcium and vitamin D supplementation")
        elif 'Arthritis' in prediction:
            recommendations.append("Consider rheumatology or orthopedic consultation")
            recommendations.append("Evaluate pain management strategies")
            recommendations.append("Physical therapy may be beneficial")
        else:
            recommendations.append("Normal knee joint appearance")
            recommendations.append("Clinical correlation advised for any ongoing symptoms")
    
    # Always include these general recommendations
    recommendations.append("This AI analysis should supplement, not replace, clinical judgment")
    recommendations.append("Consider patient history, physical examination, and other diagnostic findings")
    
    return recommendations

def create_summary_statistics(results_list: list) -> Dict[str, Any]:
    """Create summary statistics from multiple analysis results"""
    if not results_list:
        return {}
    
    total_cases = len(results_list)
    
    # Count predictions by type
    prediction_counts = {}
    confidence_levels = []
    
    for result in results_list:
        prediction = result['prediction']
        confidence = result['confidence']
        
        prediction_counts[prediction] = prediction_counts.get(prediction, 0) + 1
        confidence_levels.append(confidence)
    
    # Calculate statistics
    avg_confidence = np.mean(confidence_levels) if confidence_levels else 0
    high_confidence_cases = sum(1 for c in confidence_levels if c > 0.8)
    low_confidence_cases = sum(1 for c in confidence_levels if c < 0.6)
    
    summary = {
        'total_cases': total_cases,
        'prediction_distribution': prediction_counts,
        'average_confidence': avg_confidence,
        'high_confidence_cases': high_confidence_cases,
        'low_confidence_cases': low_confidence_cases,
        'confidence_levels': confidence_levels
    }
    
    return summary

# Example usage and testing
if __name__ == "__main__":
    print("Report generation module loaded successfully!")
    
    # Test with dummy data
    dummy_results = {
        'prediction': 'Fracture Detected',
        'confidence': 0.85,
        'type': 'bone',
        'model_used': 'Bone Fracture Detection Model',
        'timestamp': datetime.now(),
        'original_image': Image.new('RGB', (224, 224), 'white')
    }
    
    print("Test data created for report generation")