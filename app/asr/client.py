"""
Cliente para serviço de transcrição de áudio (ASR) usando OpenAI Whisper
"""

import asyncio
import io
import logging
import os
import tempfile
from typing import Optional

from openai import OpenAI
from pydub import AudioSegment

logger = logging.getLogger(__name__)

# Initialize OpenAI client
# the newest OpenAI model is "gpt-5" which was released August 7, 2025.
# do not change this unless explicitly requested by the user
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


async def transcribe_audio_chunk(audio_data: bytes) -> str:
    """
    Transcreve um chunk de áudio usando OpenAI Whisper
    
    Args:
        audio_data: Dados de áudio em bytes (formato WebM/Opus ou WAV)
        
    Returns:
        Texto transcrito em português
    """
    try:
        # Tentar converter o áudio WebM/Opus para WAV
        try:
            # Criar buffer de memória com os dados
            audio_buffer = io.BytesIO(audio_data)
            
            # Carregar áudio (pydub detecta formato automaticamente)
            audio = AudioSegment.from_file(audio_buffer)
            
            # Converter para WAV com taxa de amostragem de 16kHz
            wav_buffer = io.BytesIO()
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(wav_buffer, format="wav")
            wav_data = wav_buffer.getvalue()
            
            logger.debug(f"Áudio convertido: {len(audio_data)} bytes WebM -> {len(wav_data)} bytes WAV")
            
        except Exception as conv_error:
            logger.warning(f"Não foi possível converter áudio, usando original: {conv_error}")
            wav_data = audio_data
        
        # Criar arquivo temporário para o áudio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(wav_data)
            temp_file_path = temp_file.name
        
        try:
            # Usar a função do loop de eventos para não bloquear
            loop = asyncio.get_event_loop()
            transcription = await loop.run_in_executor(
                None, 
                _transcribe_sync, 
                temp_file_path
            )
            
            logger.info(f"Transcrição realizada: {len(transcription)} caracteres")
            return transcription
            
        finally:
            # Limpar arquivo temporário
            os.unlink(temp_file_path)
            
    except Exception as e:
        logger.error(f"Erro na transcrição de áudio: {e}")
        return ""


def _transcribe_sync(audio_file_path: str) -> str:
    """
    Função síncrona para transcrição (executa em thread separada)
    """
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = openai_client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                language="pt"  # Português
            )
        return response.text
    except Exception as e:
        logger.error(f"Erro na transcrição síncrona: {e}")
        return ""


async def transcribe_audio_file(file_path: str) -> str:
    """
    Transcreve um arquivo de áudio completo
    
    Args:
        file_path: Caminho para o arquivo de áudio
        
    Returns:
        Texto transcrito em português
    """
    try:
        # Usar a função do loop de eventos para não bloquear
        loop = asyncio.get_event_loop()
        transcription = await loop.run_in_executor(
            None, 
            _transcribe_sync, 
            file_path
        )
        
        logger.info(f"Transcrição de arquivo realizada: {len(transcription)} caracteres")
        return transcription
        
    except Exception as e:
        logger.error(f"Erro na transcrição de arquivo: {e}")
        return ""


def validate_audio_format(audio_data: bytes) -> bool:
    """
    Valida se os dados de áudio estão em formato suportado
    
    Args:
        audio_data: Dados de áudio em bytes
        
    Returns:
        True se o formato for válido
    """
    if len(audio_data) < 1000:  # Muito pequeno
        return False
    
    # Verificar alguns headers básicos
    # WAV header
    if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
        return True
    
    # MP3 header
    if audio_data[:3] == b'ID3' or audio_data[:2] == b'\xff\xfb':
        return True
    
    # Para outros formatos, aceitar por enquanto
    return True


# Configurações de qualidade de áudio
AUDIO_CONFIG = {
    "sample_rate": 16000,  # 16kHz é suficiente para voz
    "channels": 1,  # Mono
    "bit_depth": 16,  # 16 bits
    "format": "wav"  # Formato preferido
}