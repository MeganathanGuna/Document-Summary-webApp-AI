import streamlit as st
import fitz  # PyMuPDF
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
import torch

# ────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="LexiSumm - Legal Document Summarizer",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ LexiSumm – Indian Legal Document Summarizer")
st.markdown("**Upload any Indian legal document (Judgment, Contract, Agreement, FIR, Notice) and get instant clean summary**")

# ────────────────────────────────────────────────
# DISCLAIMER
# ────────────────────────────────────────────────
st.warning(
    "⚠️ This tool is for reference and educational purposes only. "
    "It is NOT legal advice. Always consult a qualified lawyer."
)

# ────────────────────────────────────────────────
# MODEL LOADING (cached) - MANUAL VERSION
# ────────────────────────────────────────────────
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# ────────────────────────────────────────────────
# HELPER FUNCTIONS
# ────────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text")
        doc.close()
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()


def chunk_text(text, max_words=450):
    words = text.split()
    chunks = []
    current = []
    for word in words:
        current.append(word)
        if len(current) >= max_words:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


def generate_summary(text, summary_level="Medium"):
    text = clean_text(text)
    
    if len(text.split()) < 50:
        return "Document is too short to summarize meaningfully."
    
    level_map = {
        "Short":    (80,  130),
        "Medium":   (150, 250),
        "Detailed": (250, 400)
    }
    max_len, min_len = level_map[summary_level]
    
    chunks = chunk_text(text)
    summaries = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, chunk in enumerate(chunks):
        if len(chunk.split()) > 30:
            try:
                inputs = tokenizer(
                    chunk,
                    return_tensors="pt",
                    max_length=1024,
                    truncation=True
                )
                
                summary_ids = model.generate(
                    inputs["input_ids"],
                    max_length=max_len,
                    min_length=min_len,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0
                )
                
                summary_text = tokenizer.decode(
                    summary_ids[0],
                    skip_special_tokens=True
                ).strip()
                
                summaries.append(summary_text)
            except Exception as e:
                st.warning(f"Chunk {i+1} failed: {str(e)[:80]}...")
                summaries.append(chunk[:400] + " … [error]")
        else:
            summaries.append(chunk.strip())
        
        progress_bar.progress((i + 1) / len(chunks))
        status_text.text(f"Processing chunk {i+1}/{len(chunks)} …")
    
    status_text.empty()
    
    combined = " ".join(summaries).strip()
    
    # Optional: one final summarization pass if still very long
    if len(combined.split()) > 350:
        try:
            inputs = tokenizer(
                combined,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            )
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=350,
                min_length=150,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            final_summary = tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            ).strip()
            return final_summary
        except:
            pass  # keep combined if final pass fails
    
    return combined


def extract_key_highlights(text):
    if not text:
        return {}
    
    highlights = {}
    
    # Dates
    dates = re.findall(r'\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\b', text)
    if dates:
        highlights["Important Dates"] = list(set(dates))[:8]
    
    # Legal sections (India-focused)
    sections = re.findall(r'(?i)(?:section|sec\.?|article|art\.?|clause|cl\.?|§)\s*(\d+[A-Za-z]?(?:\s*(?:to|and|-)\s*\d+[A-Za-z]?)?(?:\s*\([A-Za-z0-9]+\))?)?', text)
    if sections:
        highlights["Key Sections"] = list(set(s.strip() for s in sections if s.strip()))[:10]
    
    # Parties
    parties_patterns = [
        r'(?i)(?:petitioner|respondent|plaintiff|defendant|appellant|appellate|accused|complainant|vs\.?|v\.?|versus)\s*[:\-]?\s*([A-Za-z0-9\s\.,&\'\(\)]{5,80})',
        r'(?i)(?:between|and|by)\s+([A-Za-z0-9\s\.,&\'\(\)]{5,100})'
    ]
    parties = []
    for pat in parties_patterns:
        parties.extend(re.findall(pat, text))
    if parties:
        cleaned = [p.strip() for p in parties if len(p.strip()) > 5]
        highlights["Parties Involved"] = list(set(cleaned))[:6]
    
    return highlights


# ────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose PDF (Judgment / Contract / Agreement / Notice / FIR)",
        type=["pdf"]
    )
    
    summary_level = st.radio(
        "Summary Length",
        ["Short", "Medium", "Detailed"],
        index=1,
        horizontal=True
    )
    
    st.markdown("---")
    st.info("First run downloads ~320 MB model (~10–90 s depending on connection)")


# ────────────────────────────────────────────────
# MAIN LOGIC
# ────────────────────────────────────────────────
if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)
    
    if raw_text and len(raw_text) > 80:
        col1, col2 = st.columns([1, 1.15])
        
        with col1:
            st.subheader("📄 Original Document (first part)")
            st.text_area(
                "Preview (first ~1800 characters)",
                raw_text[:1800] + " …",
                height=520
            )
        
        with col2:
            st.subheader("✨ Generated Summary")
            start_time = time.time()
            
            with st.spinner("Summarizing document... (this may take 30–180 seconds)"):
                summary = generate_summary(raw_text, summary_level)
            
            time_taken = round(time.time() - start_time, 1)
            st.success(f"Generated in {time_taken} seconds")
            
            st.markdown(summary)
            
            st.download_button(
                label="📥 Download Summary (.txt)",
                data=summary,
                file_name=f"LexiSumm_{summary_level}_{uploaded_file.name.rsplit('.',1)[0]}.txt",
                mime="text/plain"
            )
        
        # ── Highlights ────────────────────────────────────────
        st.subheader("🔍 Extracted Key Information")
        highlights = extract_key_highlights(raw_text)
        
        if highlights:
            for category, items in highlights.items():
                if items:
                    st.markdown(f"**{category}**")
                    st.write(" • " + " • ".join(items))
        else:
            st.info("No clear dates/sections/parties detected in this document.")
        
        st.info(
            "💡 Next steps: clause risk scoring, Hindi support, "
            "chat-with-document (RAG), comparison of two contracts…"
        )
    
    else:
        st.error("Could not extract readable text. The PDF may be scanned / image-only.")

else:
    st.info("👆 Upload a legal PDF to start")
    st.markdown("""
    ### Currently supported document types
    • Supreme Court & High Court judgments  
    • Contracts, Agreements, MoUs  
    • Legal Notices  
    • FIRs, Charge sheets  
    """)

# ────────────────────────────────────────────────
# FOOTER
# ────────────────────────────────────────────────
st.markdown("---")
st.caption("LexiSumm v1.2 • Manual model loading • transformers 4.44.2 • Made with ❤️ • 2026")