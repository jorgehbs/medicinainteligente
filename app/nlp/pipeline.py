"""
Pipeline de NLP clínico para português brasileiro
Extração de sintomas, achados clínicos e processamento de negações
"""

import re
import logging
from typing import List, Set, Dict, Any
import spacy
from spacy.lang.pt import Portuguese
from negspacy.negation import Negex

from app.models import ClinicalFacts

logger = logging.getLogger(__name__)

# Inicialização do modelo spaCy para português
try:
    # Tentar carregar modelo completo
    nlp = spacy.load("pt_core_news_sm")
except OSError:
    logger.warning("Modelo pt_core_news_sm não encontrado, usando modelo básico")
    # Fallback para modelo básico
    nlp = Portuguese()

# Adicionar componente de negação
nlp.add_pipe("negex", config={"ent_types": ["SYMPTOM", "CONDITION", "FINDING"]})

# Dicionários clínicos específicos para português médico
SINTOMAS_COMUNS = {
    # Sintomas gerais
    "dor", "dores", "dolorido", "dolorosa", "doendo",
    "febre", "febril", "temperatura", "calafrios",
    "cansaço", "fadiga", "fraqueza", "astenia",
    "tontura", "vertigem", "mal estar", "indisposição",
    "náusea", "náuseas", "enjoo", "vômito", "vômitos",
    
    # Sintomas respiratórios
    "tosse", "tossindo", "catarro", "expectoração",
    "falta de ar", "dispneia", "chiado", "sibilância",
    "dor no peito", "aperto no peito",
    
    # Sintomas gastrointestinais
    "diarreia", "constipação", "prisão de ventre",
    "dor abdominal", "dor na barriga", "azia", "queimação",
    "gases", "flatulência", "inchaço",
    
    # Sintomas neurológicos
    "dor de cabeça", "cefaleia", "enxaqueca",
    "confusão", "esquecimento", "tontura",
    "formigamento", "dormência", "fraqueza muscular",
    
    # Sintomas cardiovasculares
    "palpitação", "batimento acelerado", "aperto no peito",
    "inchaço nas pernas", "edema", "falta de ar aos esforços"
}

ACHADOS_EXAME = {
    "pressão alta", "hipertensão", "pressão baixa", "hipotensão",
    "sopro", "sopro cardíaco", "arritmia", "batimentos irregulares",
    "linfonodos aumentados", "ínguas", "gânglios",
    "abdome distendido", "rigidez abdominal", "massa palpável",
    "edema", "inchaço", "cianose", "palidez", "icterícia",
    "rash", "erupção", "lesões de pele", "petéquias"
}

RED_FLAGS = {
    "dor torácica intensa", "dor no peito forte", "infarto",
    "dificuldade respiratória severa", "falta de ar intensa",
    "pressão muito alta", "pressão sistólica acima de 180",
    "febre alta", "temperatura acima de 39", "febre persistente",
    "perda de consciência", "desmaio", "síncope",
    "sangramento", "hemorragia", "sangue nas fezes", "sangue na urina",
    "dor abdominal intensa", "abdome em tábua",
    "alteração do nível de consciência", "confusão mental",
    "convulsões", "crises convulsivas"
}

MEDICAMENTOS_PATTERNS = [
    r'\b\w+cilina\b',  # Antibióticos penicilina
    r'\b\w+prazol\b',  # Inibidores bomba prótons
    r'\b\w+sartan\b',  # ARBs
    r'\b\w+pril\b',    # IECA
    r'\b\w+olol\b',    # Beta-bloqueadores
    r'\b\w+tina\b',    # Estatinas
    r'\b\w+micina\b',  # Antibióticos aminoglicosídeos
]


def normalize_text(text: str) -> str:
    """
    Normaliza texto clínico em português
    """
    if not text:
        return ""
    
    # Converter para minúsculas
    text = text.lower()
    
    # Remover caracteres especiais, manter pontuação relevante
    text = re.sub(r'[^\w\sáàãâéèêíìîóòõôúùûç.,;:!?-]', ' ', text)
    
    # Normalizar espaços múltiplos
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def extract_symptoms(text: str) -> Set[str]:
    """
    Extrai sintomas mencionados no texto
    """
    symptoms = set()
    normalized_text = normalize_text(text)
    
    # Buscar sintomas conhecidos
    for symptom in SINTOMAS_COMUNS:
        if symptom in normalized_text:
            symptoms.add(symptom)
    
    # Padrões para dor localizada
    dor_patterns = [
        r'dor\s+(?:no|na|nos|nas)\s+(\w+(?:\s+\w+)?)',
        r'dores?\s+(?:no|na|nos|nas)\s+(\w+(?:\s+\w+)?)',
        r'doendo\s+(?:o|a|os|as)\s+(\w+(?:\s+\w+)?)'
    ]
    
    for pattern in dor_patterns:
        matches = re.findall(pattern, normalized_text)
        for match in matches:
            symptoms.add(f"dor {match}")
    
    return symptoms


def extract_physical_findings(text: str) -> Set[str]:
    """
    Extrai achados de exame físico do texto
    """
    findings = set()
    normalized_text = normalize_text(text)
    
    # Buscar achados conhecidos
    for finding in ACHADOS_EXAME:
        if finding in normalized_text:
            findings.add(finding)
    
    # Padrões para medidas vitais
    vital_patterns = [
        r'pressão\s+(\d+)[x\/](\d+)',
        r'pa\s+(\d+)[x\/](\d+)',
        r'temperatura\s+(\d+(?:\.\d+)?)',
        r'fc\s+(\d+)',
        r'frequência\s+cardíaca\s+(\d+)'
    ]
    
    for pattern in vital_patterns:
        matches = re.findall(pattern, normalized_text)
        if matches:
            findings.update([f"medida vital: {match}" for match in matches])
    
    return findings


def extract_medications(text: str) -> Set[str]:
    """
    Extrai medicamentos mencionados no texto
    """
    medications = set()
    normalized_text = normalize_text(text)
    
    # Buscar padrões de medicamentos
    for pattern in MEDICAMENTOS_PATTERNS:
        matches = re.findall(pattern, normalized_text, re.IGNORECASE)
        medications.update(matches)
    
    # Medicamentos comuns por nome
    common_meds = [
        "paracetamol", "dipirona", "ibuprofeno", "aspirina",
        "omeprazol", "pantoprazol", "losartana", "atenolol",
        "metformina", "sinvastatina", "amlodipina", "enalapril"
    ]
    
    for med in common_meds:
        if med in normalized_text:
            medications.add(med)
    
    return medications


def detect_red_flags(text: str) -> Set[str]:
    """
    Detecta red flags clínicos no texto
    """
    flags = set()
    normalized_text = normalize_text(text)
    
    for flag in RED_FLAGS:
        if flag in normalized_text:
            flags.add(flag)
    
    return flags


def process_negations(doc) -> Dict[str, List[str]]:
    """
    Processa negações usando negspacy
    """
    negated = []
    affirmed = []
    
    for ent in doc.ents:
        if ent._.negex:
            negated.append(ent.text.lower())
        else:
            affirmed.append(ent.text.lower())
    
    return {"negated": negated, "affirmed": affirmed}


def normalize_and_extract(text: str) -> ClinicalFacts:
    """
    Pipeline principal de NLP clínico
    
    Args:
        text: Texto da consulta médica transcrito
        
    Returns:
        ClinicalFacts: Fatos clínicos estruturados
    """
    try:
        if not text or not text.strip():
            return ClinicalFacts()
        
        # Normalizar texto
        normalized = normalize_text(text)
        
        # Processar com spaCy
        doc = nlp(normalized)
        
        # Extrair componentes
        symptoms = extract_symptoms(normalized)
        findings = extract_physical_findings(normalized)
        medications = extract_medications(normalized)
        red_flags = detect_red_flags(normalized)
        
        # Processar negações
        negation_result = process_negations(doc)
        
        # Filtrar sintomas negados
        symptoms_negated = set()
        symptoms_affirmed = set()
        
        for symptom in symptoms:
            # Verificar se há negação próxima
            if any(neg in symptom or symptom in neg for neg in negation_result["negated"]):
                symptoms_negated.add(symptom)
            else:
                symptoms_affirmed.add(symptom)
        
        # Criar resultado estruturado
        facts = ClinicalFacts(
            sintomas=list(symptoms_affirmed),
            sintomas_negados=list(symptoms_negated),
            achados_exame=list(findings),
            medicamentos=list(medications),
            red_flags=list(red_flags),
            alergias=[],  # TODO: implementar extração de alergias
            antecedentes=[]  # TODO: implementar extração de antecedentes
        )
        
        logger.info(f"NLP processado: {len(facts.sintomas)} sintomas, {len(facts.achados_exame)} achados")
        return facts
        
    except Exception as e:
        logger.error(f"Erro no pipeline NLP: {e}")
        return ClinicalFacts()


# Inicialização do modelo - baixar se necessário
def ensure_portuguese_model():
    """
    Garante que temos um modelo português disponível
    """
    try:
        nlp = spacy.load("pt_core_news_sm")
        logger.info("Modelo português carregado com sucesso")
    except OSError:
        logger.warning("Modelo pt_core_news_sm não encontrado")
        logger.info("Para instalar: python -m spacy download pt_core_news_sm")
        # Por enquanto, usar modelo básico


if __name__ == "__main__":
    # Teste básico
    test_text = """
    Paciente refere dor de cabeça há 3 dias, com febre baixa.
    Nega náuseas ou vômitos. Pressão arterial 140x90.
    Está tomando paracetamol para a dor.
    """
    
    facts = normalize_and_extract(test_text)
    print("Teste do pipeline NLP:")
    print(f"Sintomas: {facts.sintomas}")
    print(f"Sintomas negados: {facts.sintomas_negados}")
    print(f"Achados: {facts.achados_exame}")
    print(f"Medicamentos: {facts.medicamentos}")