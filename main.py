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
import mistune
from weasyprint import HTML
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
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
            logger.info("OpenRouter Response: %s", response_data)

            if "choices" not in response_data:
                raise ValueError(f"Unexpected response format: {response_data}")

            content = response_data["choices"][0]["message"]["content"]
            content = content.replace("```json", "").replace("```", "").strip()

            if json_mode:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON. Response content: %s", content)
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to parse JSON from LLM response: {content}",
                    )
            return content
        except httpx.HTTPStatusError as e:
            error_detail = f"HTTP Error {e.response.status_code}: {e.response.text}"
            logger.error(error_detail)
            raise HTTPException(status_code=e.response.status_code, detail=error_detail)
        except Exception as e:
            error_detail = f"OpenRouter API error: {str(e)}"
            logger.error(error_detail)
            raise HTTPException(status_code=500, detail=error_detail)

def preprocess_to_markdown(text: str) -> str:
    """Preprocess raw resume text to infer Markdown structure and normalize bullets."""
    if not text.strip():
        logger.error("Input text is empty")
        return ""
    
    lines = text.split('\n')
    markdown_lines = []
    heading_keywords = [
        'profile', 'summary', 'skills', 'experience', 
        'education', 'projects', 'certifications',
        'work history', 'technical', 'academic'
    ]
    in_profile_summary = False
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            in_profile_summary = False
            continue
        
        clean_line = re.sub(r'[#*_-]', '', line).strip()
        is_heading = (
            any(keyword in clean_line.lower() for keyword in heading_keywords) and
            len(clean_line.split()) <= 3 and
            clean_line == clean_line.title()
        )
        
        # Normalize bullet points (remove duplicates like - -, ••)
        if line.startswith(('-', '•', '*')):
            normalized_line = re.sub(r'^[-•*]\s*[-•*]+\s*', '- ', line)
            markdown_lines.append(normalized_line)
        elif is_heading and any(keyword in clean_line.lower() for keyword in ['profile', 'summary']):
            in_profile_summary = True
            markdown_lines.append(f"## {clean_line}")
        elif in_profile_summary:
            markdown_lines.append(clean_line)
            in_profile_summary = len(clean_line.split()) > 10
        elif is_heading:
            markdown_lines.append(f"## {clean_line}")
        else:
            markdown_lines.append(line)
    
    result = '\n'.join(markdown_lines)
    logger.info("Preprocessed Markdown: %s", result[:100])
    return result

def clean_resume_text(enhanced_resume: str) -> list:
    """Clean enhanced resume and classify lines, ensuring single bullets."""
    if not enhanced_resume.strip():
        logger.error("Enhanced resume is empty")
        return []
    
    markdown_parser = mistune.create_markdown()
    markdown_text = preprocess_to_markdown(enhanced_resume)
    if not markdown_text:
        logger.error("Markdown text is empty")
        return []
    
    logger.info("Markdown text: %s", markdown_text[:100])
    
    lines = markdown_text.split('\n')
    cleaned_lines = []
    skip_next = False
    in_profile_summary = False
    in_education = False
    in_experience = False
    
    heading_keywords = [
        'profile', 'summary', 'skills', 'experience', 
        'education', 'projects', 'certifications',
        'work history', 'technical', 'academic'
    ]
    
    # Regex for education and experience duration: year ranges, single years, or month-year formats
    month_names = r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December'
    duration_pattern = r'(?:(?:' + month_names + r')\s+\d{4}\s*[-–—]\s*(?:' + month_names + r')\s+\d{4})|' + \
                      r'(?:' + month_names + r')\s+\d{4}\s+(?:' + month_names + r')\s+\d{4}|' + \
                      r'(?:' + month_names + r')\d{4}\s+(?:' + month_names + r')\d{4}|' + \
                      r'(?:' + month_names + r')\s+\d{4}\s*[-–—]\s*Present|' + \
                      r'(?:' + month_names + r')\d{4}\s*[-–—]\s*Present|' + \
                      r'\d{4}\s*[-–—]\s*\d{4}|\d{4}\s*[-–—]\s*Present|\d{4}'
    education_regex = r'^(.*?)\s+(' + duration_pattern + r')$'
    experience_regex = r'^(.*?)\s+(' + duration_pattern + r')$'
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            in_profile_summary = False
            in_education = False
            in_experience = False
            continue
        
        clean_line = re.sub(r'[#*_-]', '', line).strip()
        # Normalize spaces and dashes for education and experience lines
        if in_education or in_experience:
            clean_line = re.sub(r'\s+', ' ', clean_line)  # Normalize multiple spaces to single
            clean_line = re.sub(r'[–—]', '-', clean_line)  # Normalize dashes to hyphen
            # Remove extra commas
            clean_line = re.sub(r',\s*,', ',', clean_line)
            clean_line = re.sub(r',\s*$', '', clean_line)
            # Add dash for Month YYYY Month YYYY if missing
            clean_line = re.sub(
                r'(' + month_names + r')\s+(\d{4})\s+(' + month_names + r')\s+(\d{4})',
                r'\1 \2 - \3 \4',
                clean_line,
                flags=re.IGNORECASE
            )
        
        if len(cleaned_lines) == 0 and clean_line == clean_line.title():
            cleaned_lines.append((clean_line, "name"))
            continue
            
        if ("@" in clean_line or 
            any(x in clean_line.lower() for x in ['http', 'www', 'linkedin', 'github', 'phone'])):
            cleaned_lines.append((clean_line, "contact"))
            continue
            
        if line.startswith('##'):
            in_education = 'education' in clean_line.lower()
            in_experience = 'experience' in clean_line.lower() or 'work history' in clean_line.lower()
            in_profile_summary = any(keyword in clean_line.lower() for keyword in ['profile', 'summary'])
            cleaned_lines.append((clean_line, "heading"))
            skip_next = True
        elif in_profile_summary:
            cleaned_lines.append((clean_line, "normal"))
            in_profile_summary = len(clean_line.split()) > 10
        elif in_education:
            match = re.match(education_regex, clean_line, re.IGNORECASE)
            if match:
                institute, duration = match.groups()
                cleaned_lines.append(((institute.strip(), duration.strip()), "education"))
            else:
                cleaned_lines.append((clean_line, "normal"))
        elif in_experience:
            match = re.match(experience_regex, clean_line, re.IGNORECASE)
            if match:
                company_location, duration = match.groups()
                cleaned_lines.append(((company_location.strip(), duration.strip()), "experience"))
            else:
                cleaned_lines.append((clean_line, "normal"))
        elif any(keyword in clean_line.lower() for keyword in heading_keywords) and len(clean_line.split()) <= 3:
            cleaned_lines.append((clean_line, "heading"))
            skip_next = True
        elif skip_next and not clean_line:
            skip_next = False
            continue
        elif line.startswith('-'):
            # Ensure single bullet
            normalized_line = re.sub(r'^[-•*]\s*[-•*]+\s*', '- ', line)
            cleaned_lines.append((normalized_line, "bullet"))
        else:
            cleaned_lines.append((clean_line, "normal"))
    
    logger.info("Cleaned resume lines: %s", cleaned_lines)
    return cleaned_lines

def extract_text_from_file(file_path: str, content_type: str) -> str:
    """Extract text from PDF or DOCX."""
    text = ""
    try:
        if content_type == "application/pdf":
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif "word" in content_type.lower():
            doc = Document(file_path)
            text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        logger.info("Extracted text: %s", text[:100])
        if not text.strip():
            logger.error("Extracted text is empty")
        return text
    except Exception as e:
        logger.error("Extraction error: %s", str(e))
        return ""

async def save_uploaded_file(upload_file: UploadFile, directory: str) -> str:
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{uuid.uuid4()}_{upload_file.filename}")
    with open(file_path, "wb") as f:
        content = await upload_file.read()
        f.write(content)
    logger.info("Saved file: %s", file_path)
    return file_path

async def analyze_jd_with_llm(job_description: str) -> Dict:
    prompt = f"""
    Analyze this job description and return JSON with:
    1. technical_skills: All technical skills mentioned in the job description
    2. power_verbs: Top 5 action verbs
    3. competencies: 3 industry terms

    Job Description:
    {job_description}

    Return the result as a **valid JSON string only, no need of extra text or explanation.
    """
    result = await deepseek_api_call(prompt, json_mode=True)
    logger.info("JD analysis result: %s", result)
    return result if isinstance(result, dict) else json.loads(result)

async def enhance_resume_with_llm(resume_text: str, jd_analysis: Dict) -> str:
    if not resume_text.strip():
        logger.error("Resume text is empty before LLM enhancement")
        return ""
    
    prompt = f"""
    You are an expert resume reviewer with extensive experience in tailoring resumes to specific job descriptions. Your task is to enhance the provided resume to make it perfectly aligned with the job description, focusing on the Skills section. Follow these instructions strictly:

    1. **Incorporate All Technical Skills with Categorization**: You MUST include EVERY SINGLE technical skill listed in the job description: {jd_analysis.get('technical_skills', [])}. Add these skills to the Skills section of the resume, even if they are missing from the original resume, ensuring NO skill is omitted. Categorize the skills into the following groups: Programming Languages, Machine Learning/AI, Data Frameworks, Visualization & Tools, Cloud Platforms, and Other (for skills that don't fit these categories). Each category must be a separate bullet point in the format '- Category: Skill1, Skill2, Skill3'. List only the skills, separated by commas, with NO qualifiers (e.g., 'Proficient in', 'Skilled in') or descriptions (e.g., 'Advanced proficiency in data analysis'). If the Skills section does not exist, create one. Preserve any existing skills in the resume and place them in the appropriate categories. Cross-check that all technical skills from the job description are included in the Skills section.
    2. **Paraphrase Projects and Experience**: Rephrase project and work experience descriptions to emphasize relevance to the job description, using the following action verbs where appropriate: {jd_analysis.get('power_verbs', [])}. Maintain the original project details, outcomes, and structure (e.g., company, role, dates).
    3. **Preserve Original Content**: Retain ALL original content, including education, certifications, publications, and work history, exactly as provided. Do NOT add new certifications, publications, or qualifications not present in the original resume.
    4. **Maintain Structure**: Keep the resume’s structure (e.g., sections like Profile, Skills, Experience, Education) intact, enhancing content within these sections. Ensure the Skills section is prominent and clearly organized by categories.
    5. **Exclude Summaries**: Do NOT include any summary of changes, explanations, or additional commentary. Return only the enhanced resume text, ending with the last valid resume section (e.g., Experience, Education).

    Job Description Analysis:
    - Technical Skills: {jd_analysis.get('technical_skills', [])}
    - Power Verbs: {jd_analysis.get('power_verbs', [])}
    - Competencies: {jd_analysis.get('competencies', [])}

    Resume:
    {resume_text}

    Return the enhanced resume as plain text, maintaining the original format as closely as possible with categorized bullet points for technical skills (e.g., '- Programming Languages: Python, SQL'). Do not append any summary, explanation, or commentary.
    """
    result = await deepseek_api_call(prompt)
    logger.info("Enhanced resume raw: %s", result[:100])
    
    # Post-process to remove any trailing summary
    lines = result.split('\n')
    valid_sections = [
        'profile', 'summary', 'skills', 'experience', 'education',
        'projects', 'certifications', 'work history', 'technical', 'academic'
    ]
    cleaned_lines = []
    in_valid_section = True
    
    for line in lines:
        clean_line = re.sub(r'[#*_-]', '', line).strip()
        is_section_heading = any(keyword in clean_line.lower() for keyword in valid_sections)
        # Stop if we encounter non-resume content (e.g., summary of changes)
        if in_valid_section and not (clean_line.lower().startswith(('added', 'modified', 'enhanced', 'summary of'))):
            cleaned_lines.append(line)
        if is_section_heading:
            in_valid_section = True
        elif clean_line and not line.startswith(('-', '•', '*')) and not in_valid_section:
            in_valid_section = False
    
    cleaned_result = '\n'.join(cleaned_lines).strip()
    logger.info("Cleaned enhanced resume: %s", cleaned_result[:100])
    return cleaned_result

def create_pdf(styled_content, session_id):
    """Generate PDF using WeasyPrint from styled HTML with horizontal lines."""
    if not styled_content:
        logger.error("Styled content is empty")
        raise HTTPException(status_code=500, detail="No content to generate PDF")
    
    pdf_path = f"generated_resumes/professional_resume_{session_id}.pdf"
    os.makedirs("generated_resumes", exist_ok=True)
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {
                size: Letter;
                margin: 0.4in;
            }
            body {
                font-family: Helvetica, sans-serif;
                margin: 0;
                font-size: 10pt;
                line-height: 1.2;
            }
            .name {
                font-size: 16pt;
                font-weight: bold;
                margin-bottom: 4pt;
                text-align: center;
            }
            .heading {
                font-size: 12pt;
                font-weight: bold;
                color: #11557C;
                margin-top: 10pt;
                margin-bottom: 4pt;
                text-transform: uppercase;
            }
            hr {
                border: 1px solid #11557C;
                margin: 8pt 0;
                width: 100%;
                box-sizing: border-box;
            }
            ul {
                list-style-type: disc;
                margin-left: 18pt;
                padding-left: 0;
            }
            .skills ul {
                margin-left: 0;
                padding-left: 0;
                list-style-position: outside;
            }
            li {
                margin-bottom: 2pt;
                white-space: normal;
                word-wrap: break-word;
                max-width: 100%;
            }
            .education {
                display: flex;
                justify-content: space-between;
                margin-bottom: 4pt;
                width: 100%;
            }
            .education span:first-child {
                flex: 1;
                text-align: left;
            }
            .education span:last-child {
                text-align: right;
                white-space: nowrap;
            }
            .experience {
                display: flex;
                justify-content: space-between;
                margin-bottom: 4pt;
                width: 100%;
            }
            .experience span:first-child {
                flex: 1;
                text-align: left;
            }
            .experience span:last-child {
                text-align: right;
                white-space: nowrap;
            }
            .normal, .contact {
                margin-bottom: 4pt;
            }
            .spacer {
                height: 4pt;
            }
        </style>
    </head>
    <body>
    """
    
    prev_style = None
    in_bullet_list = False
    section_open = False
    in_skills_section = False
    
    for i, item in enumerate(styled_content):
        is_last_item = i == len(styled_content) - 1
        if isinstance(item, tuple):
            if item[1] == "education":
                institute, duration = item[0]
                text = (institute, duration)
                style_key = "education"
            elif item[1] == "experience":
                company_location, duration = item[0]
                text = (company_location, duration)
                style_key = "experience"
            else:
                text, style_key = item
        else:
            text = item
            style_key = "normal"
        
        if prev_style == "heading" and style_key != "heading":
            html_content += '<div class="spacer"></div>'
        
        if style_key == "heading":
            in_skills_section = 'skills' in text.lower()
            if section_open and not is_last_item:
                html_content += '<hr>'
            section_open = True
        
        if isinstance(text, tuple):
            if style_key == "education":
                institute, duration = text
                html_content += f'<div class="education"><span>{institute}</span><span>{duration}</span></div>'
            elif style_key == "experience":
                company_location, duration = text
                html_content += f'<div class="experience"><span>{company_location}</span><span>{duration}</span></div>'
        else:
            text = text.replace('&', '&').replace('<', '<').replace('>', '>')
            if style_key == "name":
                html_content += f'<div class="name">{text}</div>'
            elif style_key == "contact":
                html_content += f'<div class="contact">{text}</div>'
            elif style_key == "heading":
                if in_bullet_list:
                    html_content += '</ul>'
                    if in_skills_section:
                        html_content += '</div>'
                    in_bullet_list = False
                html_content += f'<div class="heading">{text}</div>'
                section_open = True
            elif style_key == "bullet":
                if not in_bullet_list:
                    html_content += '<div class="skills">' if in_skills_section else ''
                    html_content += '<ul>'
                    in_bullet_list = True
                html_content += f'<li>{text.lstrip("- ").strip()}</li>'
            else:
                if in_bullet_list:
                    html_content += '</ul>'
                    if in_skills_section:
                        html_content += '</div>'
                    in_bullet_list = False
                html_content += f'<div class="normal">{text}</div>'
        
        prev_style = style_key
    
    if in_bullet_list:
        html_content += '</ul>'
        if in_skills_section:
            html_content += '</div>'
    
    if section_open:
        html_content += '<hr>'
    
    html_content += """
    </body>
    </html>
    """
    
    logger.info("HTML content: %s", html_content[:200])
    
    temp_html_path = f"generated_resumes/temp_{session_id}.html"
    # Ensure old temp file is removed
    if os.path.exists(temp_html_path):
        os.remove(temp_html_path)
    
    with open(temp_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    try:
        HTML(temp_html_path).write_pdf(pdf_path, zoom=1)
        logger.info("Generated PDF at: %s", pdf_path)
    except Exception as e:
        logger.error("WeasyPrint error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"WeasyPrint failed: {str(e)}")
    
    if os.path.exists(temp_html_path):
        os.remove(temp_html_path)
    
    return pdf_path

# @app.get("/", response_class=HTMLResponse)
# async def upload_form(request: Request):
#     return templates.TemplateResponse("upload.html", {"request": request})

@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    """Render upload form and clear generated_resumes folder."""
    folder_path = "generated_resumes"
    try:
        os.makedirs(folder_path, exist_ok=True)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                logger.info("Deleted: %s", file_path)
    except Exception as e:
        logger.error("Error clearing %s: %s", folder_path, str(e))
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/process")
async def process_resume(
    request: Request,
    resume: UploadFile = File(...),
    job_description: str = Form(...),
):
    try:
        logger.info("Received form data: resume=%s, job_description=%s", 
                    resume.filename, job_description[:50])
        
        if not resume or not job_description:
            logger.error("Missing resume or job_description")
            raise HTTPException(
                status_code=400,
                detail="Both resume file and job description are required"
            )

        allowed_types = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]
        if resume.content_type not in allowed_types:
            logger.error("Unsupported file type: %s", resume.content_type)
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload PDF or DOCX"
            )

        session_id = str(uuid.uuid4())
        file_path = await save_uploaded_file(resume, UPLOAD_DIR)
        
        try:
            resume_text = extract_text_from_file(file_path, resume.content_type)
            if not resume_text.strip():
                logger.error("No text extracted from resume")
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract text from file"
                )

            jd_analysis = await analyze_jd_with_llm(job_description)
            logger.info("JD analysis: %s", jd_analysis)
            enhanced_resume_raw = await enhance_resume_with_llm(resume_text, jd_analysis)
            if not enhanced_resume_raw.strip():
                logger.error("Enhanced resume is empty")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to enhance resume"
                )
            
            enhanced_resume_cleaned = clean_resume_text(enhanced_resume_raw)
            if not enhanced_resume_cleaned:
                logger.error("Cleaned resume is empty")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to process enhanced resume"
                )

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
            if os.path.exists(file_path):
                os.remove(file_path)
                
    except HTTPException as he:
        logger.error("HTTPException in process_resume: %s", str(he))
        raise he
    except Exception as e:
        logger.error("Unexpected error in process_resume: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )

@app.get("/download/{session_id}")
async def download_resume(session_id: str):
    file_path = os.path.join("generated_resumes", f"professional_resume_{session_id}.pdf")
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="application/pdf",
            filename="optimized_resume.pdf",
        )
    return RedirectResponse("/")


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)

@app.get("/favicon.ico")
async def favicon():
    """Handle favicon.ico request to suppress 404."""
    logger.info("Favicon requested, returning empty response")
    return {"detail": "No favicon available"}, 404

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=2)