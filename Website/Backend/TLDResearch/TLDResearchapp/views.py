from django.shortcuts import render, HttpResponse

# Create your views here.
def home(request):
    if request.user.is_authenticated:
        return render(request, "summarize.html")
    else:
        return render(request, "landing.html")

def main(request):
    if request.user.is_authenticated:
        return render(request, "summarize.html")
    else:
        return render(request, "landing.html")

from datetime import datetime
import os
import fitz  # PyMuPDF for PDF parsing
import google.generativeai as genai
import google.auth
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from google.auth import default
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Configure Google Generative AI with service account
credentials, _ = google.auth.load_credentials_from_file(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
genai.configure(credentials=credentials)

db_user, db_password = os.getenv('MONGODB_USER'), os.getenv('MONGODB_PASS')

mongouri = f'mongodb+srv://{db_user}:{db_password}@cluster0.ugz5x.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'

mongo_client = MongoClient(mongouri)

db = mongo_client['TLDResearch']
summaries_collection = db['summaries']

from django.contrib.auth.decorators import login_required

def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text.strip()

def generate_gemini_response(text):
    """Sends extracted text to Gemini 1.5 Pro and returns response."""
    model = genai.GenerativeModel("gemini-1.5-pro")

    prompt = f"""
    You are a summarization tool, your job is to take in research papers, and summarizes them into 2-3 sentences capturing the main ideas and outputting them back to the user. Your output should be a good representation of what the reader expects the paper to cover.

    To help you with this task, here is an example input research paper and an optimal summary. You should try to mimic the style of the example output summary in the summary you generate.

    Example of an input research paper:
    Key equatorial climate phenomena such as QBO and ENSO have never been adequately explained as deterministic processes. This in spite of recent research showing growing evidence of predictable behavior. This study applies the fundamental Laplace tidal equations with simplifying assumptions along the equator — i.e. no Coriolis force and a small angle approximation. The solutions to the partial differential equations are highly non-linear related to Navier-Stokes and only search approaches can be used to fit to the data.

    Example of an optimal output summary:
    Analytical Formulation of Equatorial Standing Wave Phenomena: Application to QBO and ENSO.

    Examples done. Summarize the research paper adhering to everything you were just told:
    {text}

    Summary:
    """

    response = model.generate_content(prompt)
    return response.text if response else "No response from Gemini."

@csrf_exempt
def handle_pdf_upload(request):
    if request.method == 'POST' and request.FILES.get('file'):
        if not request.user.is_authenticated:
            return JsonResponse({"error": "User not authenticated"}, status=401)
    
        uploaded_file = request.FILES['file']

        # Save the file temporarily
        file_path = os.path.join('uploads', uploaded_file.name)
        file_name = default_storage.save(file_path, ContentFile(uploaded_file.read()))
        full_path = default_storage.path(file_name)  # Get absolute path

        try:
            # Extract text from the PDF
            extracted_text = extract_text_from_pdf(full_path)

            # Send extracted text to Gemini AI
            ai_response = generate_gemini_response(extracted_text)
            
            username = request.user.username

            document = {
                "username": username,
                "file_name": file_name,
                "summary": ai_response,
                "timestamp": datetime.utcnow().isoformat()
            }

            insert = summaries_collection.insert_one(document)

            return JsonResponse({
                "message": "File processed successfully",
                #"filename": uploaded_file.name,
                #"extracted_text": extracted_text[:500],  # Return a preview
                "ai_response": ai_response
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({'error': 'No file uploaded'}, status=400)

@login_required  # Ensures only authenticated users can access this route
def get_summaries(request):
    try:
        username = request.user.username  # Get the logged-in user's username

        # Query MongoDB for all summaries by this user
        summaries = list(summaries_collection.find({"username": username}, {"_id": 0}))  # Exclude _id field

        return JsonResponse({"summaries": summaries}, status=200)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
