"""
Orchestrador dos 4 painéis clínicos em tempo real
Consolida dados do NLP e aplica regras clínicas
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

from app.models import PanelsState, ClinicalFacts, DiagnosticHypothesis
from app.nlp.rules import ClinicalRuleEngine

logger = logging.getLogger(__name__)

# Instância global do motor de regras
rule_engine = ClinicalRuleEngine()


def generate_syndromic_summary(facts: ClinicalFacts, full_text: str) -> str:
    """
    Gera resumo sindrômico do quadro clínico atual
    """
    try:
        # Componentes do resumo
        components = []
        
        # Sintomas principais
        if facts.sintomas:
            main_symptoms = facts.sintomas[:5]  # Top 5 sintomas
            components.append(f"Sintomas: {', '.join(main_symptoms)}")
        
        # Achados relevantes
        if facts.achados_exame:
            components.append(f"Achados: {', '.join(facts.achados_exame[:3])}")
        
        # Red flags se houver
        if facts.red_flags:
            components.append(f"⚠️ Alertas: {', '.join(facts.red_flags)}")
        
        # Se não há dados suficientes
        if not components:
            return "Dados insuficientes para análise sindrômica. Continue a consulta."
        
        # Resumo baseado na duração estimada da consulta
        text_length = len(full_text.split())
        if text_length < 50:
            prefix = "Início da consulta - "
        elif text_length < 200:
            prefix = "Consulta em andamento - "
        else:
            prefix = "Quadro estabelecido - "
        
        return prefix + "; ".join(components)
        
    except Exception as e:
        logger.error(f"Erro ao gerar resumo sindrômico: {e}")
        return "Erro na análise sindrômica"


def generate_diagnostic_hypotheses(facts: ClinicalFacts, full_text: str) -> List[DiagnosticHypothesis]:
    """
    Gera hipóteses diagnósticas com base nos fatos clínicos
    """
    try:
        hypotheses = []
        
        # Usar motor de regras para diagnósticos
        rule_results = rule_engine.generate_diagnoses(facts)
        
        # Converter resultados em hipóteses estruturadas
        for diagnosis in rule_results:
            hypothesis = DiagnosticHypothesis(
                dx=diagnosis["name"],
                conf=diagnosis["confidence"],
                porque=diagnosis["reasons"]
            )
            hypotheses.append(hypothesis)
        
        # Se não temos hipóteses das regras, gerar algumas básicas
        if not hypotheses and facts.sintomas:
            hypotheses = _generate_basic_hypotheses(facts)
        
        # Ordenar por confiança
        hypotheses.sort(key=lambda x: x.conf, reverse=True)
        
        return hypotheses[:5]  # Top 5 hipóteses
        
    except Exception as e:
        logger.error(f"Erro ao gerar hipóteses diagnósticas: {e}")
        return []


def _generate_basic_hypotheses(facts: ClinicalFacts) -> List[DiagnosticHypothesis]:
    """
    Gera hipóteses básicas baseadas em padrões simples
    """
    hypotheses = []
    symptoms = facts.sintomas
    
    # Padrões respiratórios
    respiratory_symptoms = {"tosse", "falta de ar", "dispneia", "chiado", "catarro"}
    if any(s in " ".join(symptoms) for s in respiratory_symptoms):
        hypotheses.append(DiagnosticHypothesis(
            dx="Síndrome respiratória",
            conf=0.7,
            porque=["Sintomas respiratórios presentes"]
        ))
    
    # Padrões gastrointestinais
    gi_symptoms = {"náusea", "vômito", "diarreia", "dor abdominal", "azia"}
    if any(s in " ".join(symptoms) for s in gi_symptoms):
        hypotheses.append(DiagnosticHypothesis(
            dx="Síndrome gastrointestinal",
            conf=0.6,
            porque=["Sintomas gastrointestinais presentes"]
        ))
    
    # Febre + sintomas gerais
    if any("febre" in s for s in symptoms):
        hypotheses.append(DiagnosticHypothesis(
            dx="Síndrome febril",
            conf=0.8,
            porque=["Febre presente", "Possível processo infeccioso"]
        ))
    
    # Dor como sintoma principal
    pain_symptoms = [s for s in symptoms if "dor" in s]
    if pain_symptoms:
        hypotheses.append(DiagnosticHypothesis(
            dx="Síndrome dolorosa",
            conf=0.5,
            porque=[f"Dor localizada: {', '.join(pain_symptoms[:2])}"]
        ))
    
    return hypotheses


def generate_clinical_questions(facts: ClinicalFacts, current_state: PanelsState) -> List[str]:
    """
    Gera perguntas clínicas para preencher lacunas importantes
    """
    try:
        questions = []
        
        # Usar motor de regras para lacunas
        rule_questions = rule_engine.identify_gaps(facts, current_state)
        questions.extend(rule_questions)
        
        # Perguntas básicas de triagem
        basic_questions = _generate_basic_questions(facts)
        questions.extend(basic_questions)
        
        # Remover duplicatas e limitar
        unique_questions = list(dict.fromkeys(questions))  # Preserva ordem
        return unique_questions[:6]  # Top 6 perguntas
        
    except Exception as e:
        logger.error(f"Erro ao gerar perguntas clínicas: {e}")
        return []


def _generate_basic_questions(facts: ClinicalFacts) -> List[str]:
    """
    Gera perguntas básicas baseadas no que está faltando
    """
    questions = []
    
    # Se há dor, perguntar características
    pain_symptoms = [s for s in facts.sintomas if "dor" in s]
    if pain_symptoms:
        questions.append("Qual a intensidade da dor (0-10)?")
        questions.append("A dor piora com alguma atividade específica?")
        questions.append("Há algo que alivia a dor?")
    
    # Se há febre, investigar
    if any("febre" in s for s in facts.sintomas):
        questions.append("Há quanto tempo tem febre?")
        questions.append("A febre é contínua ou intermitente?")
    
    # Antecedentes se não mencionados
    if not facts.antecedentes:
        questions.append("Tem alguma doença crônica conhecida?")
        questions.append("Toma algum medicamento regularmente?")
    
    # Alergias se não mencionadas
    if not facts.alergias:
        questions.append("Tem alergia a algum medicamento?")
    
    # Sinais vitais se não registrados
    if not facts.achados_exame:
        questions.append("Verificar sinais vitais (PA, FC, Tax)")
    
    return questions


def generate_clinical_management(facts: ClinicalFacts, current_state: PanelsState) -> List[str]:
    """
    Gera sugestões de conduta clínica e alertas
    """
    try:
        management = []
        
        # Usar motor de regras para condutas
        rule_management = rule_engine.suggest_management(facts, current_state)
        management.extend(rule_management)
        
        # Condutas básicas
        basic_management = _generate_basic_management(facts)
        management.extend(basic_management)
        
        # Alertas de red flags
        if facts.red_flags:
            management.insert(0, f"🚨 RED FLAG: {'; '.join(facts.red_flags)}")
        
        # Remover duplicatas
        unique_management = list(dict.fromkeys(management))
        return unique_management[:8]  # Top 8 condutas
        
    except Exception as e:
        logger.error(f"Erro ao gerar condutas clínicas: {e}")
        return []


def _generate_basic_management(facts: ClinicalFacts) -> List[str]:
    """
    Gera condutas básicas baseadas nos achados
    """
    management = []
    
    # Manejo da dor
    if any("dor" in s for s in facts.sintomas):
        management.append("Considerar analgesia conforme intensidade da dor")
    
    # Manejo da febre
    if any("febre" in s for s in facts.sintomas):
        management.append("Antitérmico se febre > 37.8°C")
        management.append("Investigar foco infeccioso")
    
    # Hidratação se sintomas GI
    gi_symptoms = {"náusea", "vômito", "diarreia"}
    if any(s in " ".join(facts.sintomas) for s in gi_symptoms):
        management.append("Atenção à hidratação")
    
    # Pressão arterial elevada
    if any("pressão alta" in s or "hipertensão" in s for s in facts.achados_exame):
        management.append("Monitorar PA - considerar anti-hipertensivo")
    
    # Seguimento
    management.append("Agendar retorno conforme evolução")
    management.append("Orientar sinais de alerta para retorno imediato")
    
    return management


def update_panels(current_state: PanelsState, facts: ClinicalFacts, full_text: str) -> PanelsState:
    """
    Atualiza o estado dos 4 painéis clínicos
    
    Args:
        current_state: Estado atual dos painéis
        facts: Fatos clínicos extraídos do NLP
        full_text: Texto completo da consulta
        
    Returns:
        PanelsState: Novo estado dos painéis atualizado
    """
    try:
        logger.info("Atualizando painéis clínicos...")
        
        # Gerar cada painel
        sindromico = generate_syndromic_summary(facts, full_text)
        hipoteses = generate_diagnostic_hypotheses(facts, full_text)
        perguntas = generate_clinical_questions(facts, current_state)
        condutas = generate_clinical_management(facts, current_state)
        
        # Criar novo estado
        new_state = PanelsState(
            sindromico=sindromico,
            hipoteses=hipoteses,
            perguntas=perguntas,
            condutas=condutas
        )
        
        logger.info(f"Painéis atualizados: {len(hipoteses)} hipóteses, {len(perguntas)} perguntas, {len(condutas)} condutas")
        return new_state
        
    except Exception as e:
        logger.error(f"Erro ao atualizar painéis: {e}")
        return current_state  # Retorna estado anterior em caso de erro