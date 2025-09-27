"""
Modelos de dados para o sistema de prontuário eletrônico
"""

from datetime import datetime
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class DiagnosticHypothesis(BaseModel):
    """Hipótese diagnóstica com confiança e justificativa"""
    dx: str = Field(description="Nome do diagnóstico")
    conf: float = Field(ge=0.0, le=1.0, description="Nível de confiança (0-1)")
    porque: List[str] = Field(default_factory=list, description="Razões para esta hipótese")


class PanelsState(BaseModel):
    """Estado atual dos 4 painéis clínicos"""
    sindromico: str = Field(default="", description="Resumo sindrômico do quadro")
    hipoteses: List[DiagnosticHypothesis] = Field(default_factory=list, description="Hipóteses diagnósticas")
    perguntas: List[str] = Field(default_factory=list, description="Perguntas a serem feitas")
    condutas: List[str] = Field(default_factory=list, description="Sugestões de conduta")


class ClinicalFacts(BaseModel):
    """Fatos clínicos extraídos do texto"""
    sintomas: List[str] = Field(default_factory=list)
    sintomas_negados: List[str] = Field(default_factory=list)
    achados_exame: List[str] = Field(default_factory=list)
    medicamentos: List[str] = Field(default_factory=list)
    alergias: List[str] = Field(default_factory=list)
    antecedentes: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)


class EncounterData(BaseModel):
    """Dados completos de um atendimento"""
    encounter_id: str
    start_time: datetime
    transcript: str = ""
    facts: Optional[ClinicalFacts] = None
    panels: Optional[PanelsState] = None
    final_report: Optional[Dict[str, Any]] = None


class FinalReport(BaseModel):
    """Relatório final estruturado do atendimento"""
    encounter_id: str
    timestamp: str
    anamnese: str
    exame_fisico: Optional[str] = None
    sindromico: str
    hipoteses: List[DiagnosticHypothesis]
    condutas: List[str]
    observacoes: Optional[str] = None