"""
Prontuário Eletrônico Hands-Free - Main FastAPI Application
Sistema de registro de atendimentos médicos com painéis de apoio clínico em tempo real
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from app.asr.client import transcribe_audio_chunk
from app.nlp.pipeline import normalize_and_extract
from app.panels.orchestrator import update_panels, PanelsState
from app.models import EncounterData

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Prontuário Eletrônico Hands-Free", version="1.0.0")

# Storage em memória para MVP
TRANSCRIPTS: Dict[str, str] = {}
PANELS: Dict[str, PanelsState] = {}
RUNNING_TASKS: Dict[str, asyncio.Task] = {}
TRANSCRIPTION_WORKERS: Dict[str, asyncio.Task] = {}
AUDIO_QUEUES: Dict[str, asyncio.Queue] = {}
ACTIVE_CONNECTIONS: Dict[str, Dict[str, Set[WebSocket]]] = {"audio": {}, "panels": {}}

# Servir arquivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_index():
    """Serve a página principal do sistema"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

@app.get("/healthz")
async def health_check():
    """Endpoint de saúde do sistema"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_encounters": len(TRANSCRIPTS),
        "running_tasks": len(RUNNING_TASKS)
    }

async def minute_tick_scheduler(encounter_id: str):
    """
    Scheduler que processa transcrição a cada 10s e painéis clínicos a cada 60s
    """
    logger.info(f"Iniciando scheduler para encounter {encounter_id}")
    transcription_counter = 0

    while encounter_id in TRANSCRIPTS:
        try:
            # Buffer não é mais usado para transcrição periódica
            # As transcrições são feitas em tempo real por chunk individual
            
            # A cada 60 segundos, processar painéis clínicos
            if transcription_counter % 60 == 0:
                text = TRANSCRIPTS.get(encounter_id, "")
                
                if text.strip():
                    # Processar com NLP
                    facts = normalize_and_extract(text)
                    
                    # Atualizar painéis
                    current_state = PANELS.get(encounter_id, PanelsState())
                    updated_state = update_panels(current_state, facts, text)
                    PANELS[encounter_id] = updated_state
                    
                    # Notificar clientes conectados nos painéis
                    panel_data = updated_state.model_dump()
                    panel_data["updated_at"] = datetime.now().timestamp()
                    panel_data["transcript"] = text  # Incluir transcrição atual
                    
                    # Enviar para WebSockets conectados aos painéis DESTE encounter específico
                    if encounter_id in ACTIVE_CONNECTIONS["panels"]:
                        disconnected = set()
                        for ws in ACTIVE_CONNECTIONS["panels"][encounter_id]:
                            try:
                                await ws.send_text(json.dumps(panel_data))
                            except Exception:
                                disconnected.add(ws)
                        
                        # Limpar conexões desconectadas
                        ACTIVE_CONNECTIONS["panels"][encounter_id] -= disconnected
                    
                    logger.info(f"Painéis atualizados para encounter {encounter_id}")
            
            # Aguardar 1 segundo e incrementar contador
            await asyncio.sleep(1)
            transcription_counter += 1
            
        except Exception as e:
            logger.error(f"Erro no scheduler para encounter {encounter_id}: {e}")
            await asyncio.sleep(5)  # Retry em caso de erro

async def process_audio_queue(encounter_id: str):
    """Processa chunks de áudio em fila garantindo ordem de chegada."""
    queue = AUDIO_QUEUES.get(encounter_id)
    if queue is None:
        logger.warning(f"Fila de áudio ausente para encounter {encounter_id}")
        return

    logger.info(f"Iniciando worker de transcrição para encounter {encounter_id}")

    try:
        while True:
            try:
                audio_chunk = await queue.get()
            except asyncio.CancelledError:
                logger.info(f"Worker de transcrição cancelado para encounter {encounter_id}")
                raise

            try:
                transcription = await transcribe_audio_chunk(audio_chunk)
                if transcription and transcription.strip():
                    cleaned = transcription.strip()
                    current_text = TRANSCRIPTS.get(encounter_id, "").strip()
                    updated_text = f"{current_text} {cleaned}".strip() if current_text else cleaned
                    TRANSCRIPTS[encounter_id] = updated_text
                    logger.info(f"Nova transcrição adicionada: '{cleaned}'")
                    await broadcast_transcript_update(encounter_id, updated_text)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Erro na transcrição do arquivo: {e}")
            finally:
                queue.task_done()
    except asyncio.CancelledError:
        logger.debug(f"Encerrando worker de transcrição para encounter {encounter_id}")
        raise

@app.websocket("/ws/audio/{encounter_id}")
async def websocket_audio_endpoint(websocket: WebSocket, encounter_id: str):
    """
    WebSocket para receber chunks de áudio e processar transcrição
    """
    await websocket.accept()
    
    # Organizar conexões por encounter
    if encounter_id not in ACTIVE_CONNECTIONS["audio"]:
        ACTIVE_CONNECTIONS["audio"][encounter_id] = set()
    ACTIVE_CONNECTIONS["audio"][encounter_id].add(websocket)
    logger.info(f"Conexão de áudio estabelecida para encounter {encounter_id}")
    
    try:
        # Inicializar dados do encounter se necessário
        if encounter_id not in TRANSCRIPTS:
            TRANSCRIPTS[encounter_id] = ""
            PANELS[encounter_id] = PanelsState()

            # Iniciar scheduler de 1 minuto
            if encounter_id not in RUNNING_TASKS:
                task = asyncio.create_task(minute_tick_scheduler(encounter_id))
                RUNNING_TASKS[encounter_id] = task

        if encounter_id not in AUDIO_QUEUES:
            AUDIO_QUEUES[encounter_id] = asyncio.Queue()

        if encounter_id not in TRANSCRIPTION_WORKERS:
            worker = asyncio.create_task(process_audio_queue(encounter_id))
            TRANSCRIPTION_WORKERS[encounter_id] = worker
        
        while True:
            # Receber dados (pode ser áudio em bytes ou comando de texto)
            data = await websocket.receive()
            
            if "bytes" in data:
                # Áudio recebido (agora são arquivos WebM completos de 6 segundos)
                audio_chunk = data["bytes"]
                logger.info(f"Recebido arquivo de áudio: {len(audio_chunk)} bytes")
                
                # Cada chunk agora é um arquivo WebM válido completo
                if len(audio_chunk) > 5000:  # Arquivo válido deve ter tamanho significativo
                    await AUDIO_QUEUES[encounter_id].put(audio_chunk)
                else:
                    logger.debug(f"Arquivo muito pequeno para transcrição: {len(audio_chunk)} bytes")
            
            elif "text" in data:
                text_data = data["text"]
                
                # Comando de finalização
                if text_data == "__finalize__":
                    logger.info(f"Finalizando encounter {encounter_id}")

                    if encounter_id in AUDIO_QUEUES:
                        await AUDIO_QUEUES[encounter_id].join()
                    
                    # Gerar relatório final
                    final_text = TRANSCRIPTS[encounter_id]
                    final_facts = normalize_and_extract(final_text)
                    final_panels = update_panels(PANELS[encounter_id], final_facts, final_text)
                    
                    # Enviar relatório estruturado
                    final_report = {
                        "encounter_id": encounter_id,
                        "timestamp": datetime.now().isoformat(),
                        "anamnese": final_text,
                        "sindromico": final_panels.sindromico,
                        "hipoteses": final_panels.hipoteses,
                        "condutas": final_panels.condutas,
                        "total_duration": "N/A"
                    }
                    
                    await websocket.send_text(json.dumps({
                        "type": "final_report",
                        "data": final_report
                    }))
                    
                    # Limpar recursos do encounter
                    await cleanup_encounter(encounter_id)
                    
                    break
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket de áudio desconectado para encounter {encounter_id}")
    except Exception as e:
        logger.error(f"Erro no WebSocket de áudio: {e}")
    finally:
        if encounter_id in ACTIVE_CONNECTIONS["audio"]:
            ACTIVE_CONNECTIONS["audio"][encounter_id].discard(websocket)

@app.websocket("/ws/panels/{encounter_id}")
async def websocket_panels_endpoint(websocket: WebSocket, encounter_id: str):
    """
    WebSocket para enviar atualizações dos painéis clínicos em tempo real
    """
    await websocket.accept()
    
    # Organizar conexões por encounter
    if encounter_id not in ACTIVE_CONNECTIONS["panels"]:
        ACTIVE_CONNECTIONS["panels"][encounter_id] = set()
    ACTIVE_CONNECTIONS["panels"][encounter_id].add(websocket)
    logger.info(f"Conexão de painéis estabelecida para encounter {encounter_id}")
    
    try:
        # Enviar estado atual dos painéis se existir
        if encounter_id in PANELS:
            current_panels = PANELS[encounter_id].model_dump()
            current_panels["updated_at"] = datetime.now().timestamp()
            # Incluir transcrição atual se disponível
            current_panels["transcript"] = TRANSCRIPTS.get(encounter_id, "")
            await websocket.send_text(json.dumps(current_panels))
        else:
            # Enviar painéis vazios
            empty_panels = PanelsState().model_dump()
            empty_panels["updated_at"] = datetime.now().timestamp()
            empty_panels["transcript"] = ""
            await websocket.send_text(json.dumps(empty_panels))
        
        # Manter conexão ativa
        while True:
            await websocket.receive_text()  # Aguardar mensagens do cliente
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket de painéis desconectado para encounter {encounter_id}")
    except Exception as e:
        logger.error(f"Erro no WebSocket de painéis: {e}")
    finally:
        if encounter_id in ACTIVE_CONNECTIONS["panels"]:
            ACTIVE_CONNECTIONS["panels"][encounter_id].discard(websocket)

async def broadcast_transcript_update(encounter_id: str, transcript_text: str):
    """Envia atualização imediata da transcrição para clientes dos painéis"""
    if encounter_id in ACTIVE_CONNECTIONS["panels"]:
        transcript_update = {
            "type": "transcript_update", 
            "transcript": transcript_text,
            "updated_at": datetime.now().timestamp()
        }
        
        disconnected = set()
        for ws in ACTIVE_CONNECTIONS["panels"][encounter_id]:
            try:
                await ws.send_text(json.dumps(transcript_update))
            except Exception:
                disconnected.add(ws)
        
        # Limpar conexões desconectadas
        ACTIVE_CONNECTIONS["panels"][encounter_id] -= disconnected

async def cleanup_encounter(encounter_id: str):
    """Limpa recursos de um encounter finalizado"""
    # Cancelar scheduler se estiver rodando
    if encounter_id in RUNNING_TASKS:
        RUNNING_TASKS[encounter_id].cancel()
        del RUNNING_TASKS[encounter_id]

    # Cancelar worker de transcrição e limpar fila
    if encounter_id in TRANSCRIPTION_WORKERS:
        worker = TRANSCRIPTION_WORKERS.pop(encounter_id)
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

    if encounter_id in AUDIO_QUEUES:
        queue = AUDIO_QUEUES.pop(encounter_id)
        while not queue.empty():
            queue.get_nowait()
            queue.task_done()

    # Limpar dados em memória
    TRANSCRIPTS.pop(encounter_id, None)
    PANELS.pop(encounter_id, None)
    
    logger.info(f"Recursos limpos para encounter {encounter_id}")

@app.on_event("shutdown")
async def shutdown_event():
    """Limpeza ao encerrar a aplicação"""
    logger.info("Encerrando aplicação...")

    # Cancelar todas as tarefas em execução
    for task in RUNNING_TASKS.values():
        task.cancel()
    for worker in TRANSCRIPTION_WORKERS.values():
        worker.cancel()

    # Aguardar cancelamento
    if RUNNING_TASKS:
        await asyncio.gather(*RUNNING_TASKS.values(), return_exceptions=True)
    if TRANSCRIPTION_WORKERS:
        await asyncio.gather(*TRANSCRIPTION_WORKERS.values(), return_exceptions=True)

def validate_startup_requirements():
    """Valida requisitos essenciais para inicialização"""
    # Verificar OpenAI API Key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY não encontrada nas variáveis de ambiente")
        raise ValueError("OpenAI API Key é obrigatória para transcrição de áudio")
    
    logger.info("Validação de requisitos concluída com sucesso")

if __name__ == "__main__":
    # Validar requisitos
    validate_startup_requirements()
    
    # Configuração do servidor
    port = int(os.getenv("PORT", 5000))
    host = "0.0.0.0"
    
    logger.info(f"Iniciando servidor em {host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )