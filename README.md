# EduFlow AI - Integrated Django Edition

This project is an all-in-one (Monolithic) implementation of EduFlow AI using Django. Both the backend logic (RAG, OCR) and frontend (Templates, CSS) are contained within a single directory structure.

## Features

- **OCR Integration**: Tesseract-based text extraction for Vietnamese and English.
- **Auto-Sorting**: Move files based on keywords into Theory, Examples, and Exercises folders.
- **RAG Powered Chat**: Talk to your study materials using Gemini 1.5.
- **Mock Exam Generator**: AI-driven exam creation from your documents.
- **Premium UI**: Dark-mode glassmorphism interface built with Django Templates.

## Tech Stack

- **Framework**: Django
- **Frontend**: Django Templates + Vanilla CSS (Premium Tokens)
- **AI/LLM**: Google Gemini API, LangChain
- **Vector Database**: ChromaDB
- **OCR**: Pytesseract, pdf2image

## Setup

1. **Requirements**:
   ```bash
   pip install -r requirements.txt
   brew install tesseract poppler
   ```
2. **Environment**:
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_key_here
   ```
3. **Database**:
   ```bash
   python manage.py migrate
   ```
4. **Run**:
   ```bash
   python manage.py runserver
   ```

## Folder Structure

- `eduflow/`: Project configuration.
- `study/`: Main app containing views, templates, and static assets.
- `media/`: Physical storage for processed documents, organized by subject and category.
- `data/`: Chroma vector store storage.
