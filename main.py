import os
from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
from PyPDF2 import PdfReader

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# Load policy.pdf once
reader = PdfReader("policy.pdf")
policy_text = "\n".join([page.extract_text() for page in reader.pages])

# FastAPI app
app = FastAPI(title="InsightGenie Minimal")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    prompt = f"""
    You are a helpful HR policy assistant. You need to understand the question and use the policy document to answer the question. People may ask indirect questions also. based on the policy accumulate the answer.
    Answer the question based on the following policy text:

    {policy_text}

    Question: {request.question}
    """
    try:
        response = model.generate_content(prompt)
        return QueryResponse(answer=response.text)
    except Exception as e:
        # Print the error to console for debugging
        print("Gemini API Error:", str(e))
        return QueryResponse(answer=f"Error: {str(e)}")
