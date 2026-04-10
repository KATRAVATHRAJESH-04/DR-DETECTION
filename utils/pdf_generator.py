from fpdf import FPDF
import tempfile
import base64
from PIL import Image
import os
from io import BytesIO
import re

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Diabetic Retinopathy AI Detection Report', align='C', ln=True)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def generate_pdf_report(username, original_img_b64, heatmap_img_b64, diagnosis, confidence, explanation):
    pdf = PDFReport()
    pdf.add_page()
    
    # Patient Info
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f'Patient Username: {username}', ln=True)
    
    # Diagnosis Info
    pdf.cell(0, 10, f'Diagnosis: {diagnosis}', ln=True)
    confidence_pct = confidence * 100
    pdf.cell(0, 10, f'Confidence: {confidence_pct:.1f}%', ln=True)
    pdf.ln(5)
    
    # Images
    orig_path = None
    heatmap_path = None
    try:
        # Create temp files for images
        orig_img_data = base64.b64decode(original_img_b64)
        orig_img = Image.open(BytesIO(orig_img_data)).convert('RGB')
        
        orig_fd, orig_path = tempfile.mkstemp(suffix='.jpg')
        os.close(orig_fd)
        orig_img.save(orig_path)
        
        heatmap_img_data = base64.b64decode(heatmap_img_b64)
        heatmap_img = Image.open(BytesIO(heatmap_img_data)).convert('RGB')
        
        heatmap_fd, heatmap_path = tempfile.mkstemp(suffix='.jpg')
        os.close(heatmap_fd)
        heatmap_img.save(heatmap_path)
        
        # Add images side by side
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(95, 10, 'Original Retinal Scan', align='C')
        pdf.cell(95, 10, 'Grad-CAM Heatmap', align='C', ln=True)
        
        pdf.image(orig_path, x=10, y=pdf.get_y(), w=85)
        pdf.image(heatmap_path, x=105, y=pdf.get_y(), w=85)
        
        # Move cursor below images (approx 85 units down)
        pdf.set_y(pdf.get_y() + 90)
        
    except Exception as e:
        pdf.cell(0, 10, f'(Error loading images: {str(e)})', ln=True)
    finally:
        if orig_path and os.path.exists(orig_path):
            os.remove(orig_path)
        if heatmap_path and os.path.exists(heatmap_path):
            os.remove(heatmap_path)
        
    pdf.ln(5)
    
    # Explanation
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'AI Medical Explanation:', ln=True)
    pdf.set_font('Arial', '', 11)
    
    safe_explanation = re.sub(r'\*\*(.*?)\*\*', r'\1', explanation)
    safe_explanation = re.sub(r'\*(.*?)\*', r'\1', safe_explanation)
    safe_explanation = safe_explanation.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"').replace("\u2013", "-").replace("\u2014", "-")
    pdf.multi_cell(0, 8, safe_explanation)
    
    return pdf.output()
