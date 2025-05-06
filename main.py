from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import re
import httpx
import uuid
from PyPDF2 import PdfReader
from docx import Document
from typing import Dict
import json
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT
from dotenv import load_dotenv
from reportlab.lib.units import inch  # Add this import at the top
from reportlab.lib import colors      # For color definitions

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# OpenRouter API Configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

# File storage setup
UPLOAD_DIR = "uploads"
PREVIEW_DIR = "previews"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

# Helper functions for API calls and processing
async def deepseek_api_call(prompt: str, json_mode: bool = False) -> str:
    """Universal OpenRouter API caller using Microsoft MAI DS R1"""
    async with httpx.AsyncClient() as client:
        payload = {
            "model": "microsoft/mai-ds-r1:free",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        }
        try:
            response = await client.post(
                OPENROUTER_API_URL,
                headers=HEADERS,
                json=payload,
                timeout=30.0,
            )
            response.raise_for_status()
            response_data = response.json()
            print("OpenRouter Response:", response_data)

            if "choices" not in response_data:
                raise ValueError(f"Unexpected response format: {response_data}")

            content = response_data["choices"][0]["message"]["content"]

            # Remove the JSON block syntax (```json and ```), which is causing the issue
            content = content.replace("```json", "").replace("```", "").strip()

            # If json_mode is True, attempt to parse content as JSON
            if json_mode:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON. Response content: {content}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to parse JSON from LLM response: {content}",
                    )
            return content
        except httpx.HTTPStatusError as e:
            error_detail = f"HTTP Error {e.response.status_code}: {e.response.text}"
            print(error_detail)
            raise HTTPException(status_code=e.response.status_code, detail=error_detail)
        except Exception as e:
            error_detail = f"OpenRouter API error: {str(e)}"
            print(error_detail)
            raise HTTPException(status_code=500, detail=error_detail)

def clean_resume_text(enhanced_resume):
    """
    Improved text cleaner that:
    1. Removes all # symbols and other markdown
    2. Better detects headings and sections
    3. Preserves proper spacing
    """
    lines = enhanced_resume.split('\n')
    cleaned_lines = []
    skip_next = False  # For handling multi-line sections
    
    # Common heading indicators
    heading_keywords = [
        'profile', 'summary', 'skills', 'experience', 
        'education', 'projects', 'certifications',
        'work history', 'technical', 'academic'
    ]
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Remove ALL markdown symbols and clean text
        clean_line = re.sub(r'[#*_-]', '', line).strip()
        
        # Detect name (first non-empty line in title case)
        if len(cleaned_lines) == 0 and clean_line == clean_line.title():
            cleaned_lines.append((clean_line, "name"))
            continue
            
        # Detect contact info
        if ("@" in clean_line or 
            any(x in clean_line.lower() for x in ['http', 'www', 'linkedin', 'github', 'phone'])):
            cleaned_lines.append((clean_line, "contact"))
            continue
            
        # Detect headings (lines containing keywords or ALL CAPS)
        # The is_heading function to check headings based on length and case
        is_heading = (
            (1 <= len(clean_line.split()) <= 3)  # 1-3 words
            or clean_line == clean_line.upper()  # Uppercase
            or clean_line.istitle()  # Title case
)

        
        if is_heading:
            cleaned_lines.append((clean_line, "heading"))
            skip_next = True  # Skip the next line if it's a separator
        elif skip_next and not clean_line:
            skip_next = False
            continue
        elif line.startswith(('-', '•', '*')):
            cleaned_lines.append((clean_line, "bullet"))
        else:
            cleaned_lines.append((clean_line, "normal"))
    
    return cleaned_lines

def extract_text_from_file(file_path: str, content_type: str) -> str:
    text = ""
    try:
        if content_type == "application/pdf":
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif "word" in content_type.lower():
            doc = Document(file_path)
            text = "\n".join(para.text.strip() for para in doc.paragraphs if para.text.strip())
        return text
    except Exception as e:
        print(f"Extraction error: {e}")
        return ""
    
def detect_headings(text):
    """
    Detect headings based on:
    1. Line has 1-3 words (configurable)
    2. Is followed by a newline
    3. Typically bold (represented by asterisks/markdown)
    4. Often slightly larger font (represented by # symbols/markdown)
    """
    lines = text.split('\n')
    headings = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Heading characteristics
        is_short = len(line.split()) <= 3  # 1-3 words
        has_formatting = (line.startswith('**') and line.endswith('**')) or \
                        (line.startswith('*') and line.endswith('*')) or \
                        (line.startswith('## '))  # Markdown-style bold/heading
        next_line_empty = (i < len(lines)-1) and (not lines[i+1].strip())
        
        if is_short and (has_formatting or next_line_empty):
            # Clean formatting markers
            clean_line = line.replace('**', '').replace('*', '').replace('## ', '')
            headings.append((i, clean_line))
    
    return headings

async def save_uploaded_file(upload_file: UploadFile, directory: str) -> str:
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{uuid.uuid4()}_{upload_file.filename}")
    with open(file_path, "wb") as f:
        content = await upload_file.read()
        f.write(content)
    return file_path

async def analyze_jd_with_llm(job_description: str) -> Dict:
    prompt = f"""
    Analyze this job description and return JSON with:
    1. technical_skills: Top 10 hard skills
    2. power_verbs: Top 5 action verbs  
    3. competencies: 3 industry terms

    Job Description:
    {job_description}

    Return the result as a **valid JSON string only, no need of extra text or explanation.
    """
    result = await deepseek_api_call(prompt, json_mode=True)
    return result if isinstance(result, dict) else json.loads(result)

async def enhance_resume_with_llm(resume_text: str, jd_analysis: Dict) -> str:
    prompt = f"""
    Enhance this resume by:
    1. Adding missing: {jd_analysis.get('technical_skills', [])}
    2. Using verbs: {jd_analysis.get('power_verbs', [])}
    3. Keeping ALL original content

    Resume:
    {resume_text}
    """
    return await deepseek_api_call(prompt)

def create_pdf(styled_content, session_id):
    pdf_path = f"generated_resumes/professional_resume_{session_id}.pdf"
    os.makedirs("generated_resumes", exist_ok=True)
    
    # Initialize document with tighter spacing
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    
    # Get default stylesheet
    styles = getSampleStyleSheet()
    
    # Custom styles with tighter spacing
    styles.add(ParagraphStyle(
        name='ResumeName',
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=18,
        spaceAfter=4,  # Reduced from 6
        alignment=TA_LEFT
    ))
    
    styles.add(ParagraphStyle(
        name='ResumeHeader',
        fontName='Helvetica-Bold',  # Ensures bold
        fontSize=12,
        leading=14,
        textColor=colors.HexColor('#11557C'),
        spaceBefore=10,  # Reduced from 12
        spaceAfter=4,    # Reduced from 6
        alignment=TA_LEFT
    ))
    
    styles.add(ParagraphStyle(
        name='ResumeBullet',
        fontName='Helvetica',
        fontSize=10,
        leading=12,
        leftIndent=18,
        bulletIndent=9,
        spaceAfter=2,   # Reduced from 4
        alignment=TA_LEFT
    ))
    
    styles.add(ParagraphStyle(
        name='ResumeNormal',
        fontName='Helvetica',
        fontSize=10,
        leading=12,
        spaceAfter=4,   # Reduced from 6
        alignment=TA_LEFT
    ))
    
    # Build content with optimized spacing
    content = []
    prev_style = None
    
    for item in styled_content:
        if isinstance(item, tuple):
            text, style_key = item
        else:
            text = item
            style_key = "normal"
        
        # Add minimal spacing between sections
        if prev_style == "heading" and style_key != "heading":
            content.append(Spacer(1, 4))  # Reduced spacing
        
        # Apply styles
        if style_key == "name":
            para = Paragraph(text, styles['ResumeName'])
        elif style_key == "contact":
            para = Paragraph(text, styles['ResumeNormal'])
        elif style_key == "heading":
            para = Paragraph(text.upper(), styles['ResumeHeader'])
        elif style_key == "bullet":
            para = Paragraph("• " + text, styles['ResumeBullet'])
        else:
            para = Paragraph(text, styles['ResumeNormal'])
        
        content.append(para)
        prev_style = style_key
    
    doc.build(content)
    return pdf_path

@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/process")
async def process_resume(
    request: Request,
    resume_file: UploadFile = File(...),
    jd_text: str = Form(...),
):
    try:
        # Validate inputs
        if not resume_file or not jd_text:
            raise HTTPException(
                status_code=400,
                detail="Both resume file and job description are required"
            )

        # Validate file type
        allowed_types = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]
        if resume_file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload PDF or DOCX"
            )

        session_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = await save_uploaded_file(resume_file, UPLOAD_DIR)
        
        try:
            # Extract text
            resume_text = extract_text_from_file(file_path, resume_file.content_type)
            if not resume_text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract text from file"
                )

            # Process with LLM
            jd_analysis = await analyze_jd_with_llm(jd_text)
            enhanced_resume_raw = await enhance_resume_with_llm(resume_text, jd_analysis)
            enhanced_resume_cleaned = clean_resume_text(enhanced_resume_raw)

            # Create PDF
            pdf_path = create_pdf(enhanced_resume_cleaned, session_id)

            return templates.TemplateResponse(
                "result.html",
                {
                    "request": request,
                    "original_text": resume_text,
                    "enhanced_text": enhanced_resume_raw,
                    "jd_analysis": jd_analysis,
                    "download_url": f"/download/{session_id}",
                }
            )
        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
                
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )

import shutil
from fastapi import BackgroundTasks

async def cleanup_files(session_id: str):
    """Clean up generated files after download"""
    pdf_path = os.path.join("generated_resumes", f"professional_resume_{session_id}.pdf")
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

@app.get("/download/{session_id}")
async def download_resume(session_id: str, background_tasks: BackgroundTasks):
    file_path = os.path.join("generated_resumes", f"professional_resume_{session_id}.pdf")
    if os.path.exists(file_path):
        background_tasks.add_task(cleanup_files, session_id)
        return FileResponse(
            file_path,
            media_type="application/pdf",
            filename="optimized_resume.pdf",
        )
    return RedirectResponse("/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
