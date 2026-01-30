%%writefile main.py
import os
import shutil
import json
import time
import requests
import random
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings # Colab mei ye chalta hai
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import edge_tts
import PyPDF2
from concurrent.futures import ThreadPoolExecutor
NGROK_URL = "https://melanie-flourishing-donte.ngrok-free.dev"
os.environ["GOOGLE_API_KEY"] = "enter key here"
GEMINI_KEY = "enter key here"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

VOICE_MAP = {
    "English": {"male": "en-IN-PrabhatNeural", "female": "en-IN-NeerjaNeural"},
    "Hindi":   {"male": "hi-IN-MadhurNeural", "female": "hi-IN-SwaraNeural"},
    "Hinglish": {"male": "en-IN-PrabhatNeural", "female": "hi-IN-SwaraNeural"}
}

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception: pass
    return text

def generate_content_with_gemini(prompt):
    try:
        genai.configure(api_key=GEMINI_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        if response.text: return response.text
    except: return None
    return None

def generate_quiz(text):
    prompt = f"""Generate 3 MCQs based on this text. Output JSON strictly: [ {{"question": "...", "options": ["A", "B", "C", "D"], "answer": "Option A"}} ] Context: {text[:2000]}"""
    res = generate_content_with_gemini(prompt)
    if res:
        try:
            start, end = res.find('['), res.rfind(']')
            if start != -1 and end != -1: return json.loads(res[start:end+1])
        except: pass
    return []

def generate_image_hf(prompt, filename):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        search_term = prompt[:100].replace(" ", "+")
        url = f"https://lexica.art/api/v1/search?q={search_term}"
        response = requests.get(url, headers=headers, timeout=5)
        image_url = ""
        if response.status_code == 200:
            data = response.json()
            if data and "images" in data: image_url = data["images"][0]["src"]
        if not image_url: image_url = f"https://picsum.photos/seed/{random.randint(1,1000)}/800/600"
        img_data = requests.get(image_url, headers=headers, timeout=5).content
        file_path = f"static/{filename}"
        with open(file_path, "wb") as f: f.write(img_data)
        return f"{NGROK_URL}/static/{filename}"
    except: return "https://via.placeholder.com/800?text=Image+Error"

async def generate_single_audio(text, voice, filepath):
    try:
        comm = edge_tts.Communicate(text, voice)
        await comm.save(filepath)
    except: pass

vector_store = None

@app.post("/process-pdf/")
async def process_pdf(file: UploadFile = File(...), mood: str = Form("Funny"), language: str = Form("Hinglish"), character: str = Form("Two Hosts")):
    global vector_store
    file_path = f"static/{file.filename}"
    with open(file_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
    raw_text = extract_text_from_pdf(file_path)

    # Memory Build
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(raw_text)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        print("✅ Memory Ready!")
    except Exception as e: print(f"❌ Memory Error: {e}")

    prompt = f"""Create a short podcast (only 6 dialogue excahnges) script (JSON). Topic: Based on text. SETTINGS: Mood: {mood} | Lang: {language} | Char: {character}. STRICT JSON: [ {{"speaker": "Host 1", "script": "Dialogue", "image_prompt": "Visual"}} ] Context: {raw_text[:2500]}"""
    ai_res = generate_content_with_gemini(prompt)
    segments = []
    if ai_res:
        try:
            start, end = ai_res.find('['), ai_res.rfind(']')
            if start != -1 and end != -1: segments = json.loads(ai_res[start:end+1])
        except: pass
    if not segments: segments = [{"speaker": "AI", "script": "Error generating script.", "image_prompt": "Error"}]

    final_data = []
    ts = int(time.time())
    audio_tasks = []
    lang_config = VOICE_MAP.get(language, VOICE_MAP["English"])
    for i, item in enumerate(segments):
        if isinstance(item, str): item = {"speaker": "Host", "script": item}
        audio_file = f"audio_{ts}_{i}.mp3"
        item['audio_url'] = f"{NGROK_URL}/static/{audio_file}"
        item['image_filename'] = f"image_{ts}_{i}.jpg"
        voice = lang_config["male"] if i % 2 == 0 else lang_config["female"]
        if item.get('script'): audio_tasks.append(generate_single_audio(item['script'], voice, f"static/{audio_file}"))
        final_data.append(item)
    if audio_tasks: await asyncio.gather(*audio_tasks)
    with ThreadPoolExecutor(max_workers=4) as ex:
        def process_img(item):
            item['image_url'] = generate_image_hf(item.get('image_prompt',""), item['image_filename'])
            return item
        final_data = list(ex.map(process_img, final_data))
    return {"status": "success", "background_music_url": f"{NGROK_URL}/static/background.mp3", "quiz": generate_quiz(raw_text), "data": final_data}

@app.post("/chat/")
async def chat_with_pdf(question: str = Form(...)):
    global vector_store
    if not vector_store: return {"answer": "Upload PDF first."}
    try:
        docs = vector_store.similarity_search(question)
        context = "\n".join([d.page_content for d in docs])
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        res = llm.invoke(f"Context: {context}\nQuestion: {question}")
        return {"answer": res.content}

    except Exception as e: return {"answer": str(e)}

