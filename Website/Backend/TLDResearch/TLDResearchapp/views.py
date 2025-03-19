from django.shortcuts import render, HttpResponse

# Create your views here.
def home(request):
    return HttpResponse("hello world!!")

import os
import fitz
import google.generativeai as genai
import google.auth
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from google.auth import default

# Configure Google Generative AI with service account
print(os.getenv("GOOGLE_APPLICATION_CREDENTIAL"))
credentials, _ = google.auth.load_credentials_from_file(os.getenv("GOOGLE_APPLICATION_CREDENTIAL"))
genai.configure(credentials=credentials)

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

            return JsonResponse({
                "message": "File processed successfully",
                #"filename": uploaded_file.name,
                #"extracted_text": extracted_text[:500],  # Return a preview
                "ai_response": ai_response
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({'error': 'No file uploaded'}, status=400)
