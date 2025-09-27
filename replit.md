# Overview

This is a hands-free electronic medical record (Prontuário Eletrônico Hands-Free) system built with FastAPI. It provides real-time clinical documentation through voice transcription and automated clinical analysis. The system transcribes medical consultations, extracts clinical facts using NLP, and displays structured clinical panels to assist healthcare providers during patient encounters.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Backend Architecture
- **FastAPI Framework**: RESTful API with WebSocket support for real-time communication
- **Asynchronous Processing**: Uses asyncio for handling concurrent audio processing and transcription tasks
- **In-Memory Storage**: MVP implementation stores data in Python dictionaries (transcripts, panels, audio buffers)
- **Modular Design**: Separated into distinct modules for ASR (Automatic Speech Recognition), NLP (Natural Language Processing), and clinical panels

## Frontend Architecture  
- **Static HTML/CSS/JavaScript**: Single-page application served from the static directory
- **WebSocket Client**: Real-time bidirectional communication for audio streaming and panel updates
- **Responsive Design**: Medical interface optimized for clinical workflows

## Audio Processing Pipeline
- **Real-time Audio Capture**: WebSocket-based audio streaming from browser
- **OpenAI Whisper Integration**: Cloud-based speech-to-text transcription service
- **Audio Buffer Management**: Temporary storage and processing of audio chunks

## Clinical NLP Pipeline
- **Portuguese Language Processing**: Uses spaCy with Portuguese models (pt_core_news_sm)
- **Clinical Entity Extraction**: Identifies symptoms, findings, medications, allergies, and medical history
- **Negation Detection**: negspacy integration for handling negated clinical statements
- **Clinical Rule Engine**: Deterministic rules for generating diagnoses, identifying information gaps, and suggesting management plans

## Clinical Panels System
- **Four Real-time Panels**: 
  - Syndromic summary (clinical overview)
  - Diagnostic hypotheses with confidence scores
  - Suggested questions to ask patients
  - Recommended clinical management
- **Dynamic Updates**: Panels refresh automatically as new clinical information is processed

## Data Models
- **Pydantic Models**: Type-safe data structures for clinical facts, panel states, and encounter data
- **Structured Clinical Data**: Organized extraction of symptoms, examination findings, medications, allergies, and red flags

# External Dependencies

## AI/ML Services
- **OpenAI API**: Whisper model for speech-to-text transcription
- **spaCy**: Open-source NLP library for Portuguese language processing
- **negspacy**: Negation detection extension for medical text

## Python Framework Dependencies
- **FastAPI**: Modern web framework with automatic API documentation
- **Uvicorn**: ASGI server for running the FastAPI application
- **Pydantic**: Data validation and serialization
- **WebSocket**: Real-time communication protocol support

## Development Dependencies
- **Logging**: Built-in Python logging for system monitoring
- **Asyncio**: Asynchronous programming support
- **Tempfile**: Temporary file handling for audio processing

Note: The system currently uses in-memory storage for MVP purposes but is architected to easily integrate with persistent databases like PostgreSQL in future iterations.