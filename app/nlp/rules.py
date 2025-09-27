"""
Motor de regras clínicas determinísticas - versão simplificada
Regras para diagnósticos, lacunas de informação e condutas
"""

import logging
from typing import List, Dict, Any, Set

from app.models import ClinicalFacts, PanelsState

logger = logging.getLogger(__name__)


class ClinicalRuleEngine:
    """
    Motor de regras clínicas baseado em Experta
    """
    
    def __init__(self):
        self.diagnoses = []
        self.gaps = []
        self.management = []
    
    def generate_diagnoses(self, facts: ClinicalFacts) -> List[Dict[str, Any]]:
        """
        Gera diagnósticos baseados em regras clínicas simples
        """
        try:
            diagnoses = []
            
            # Regras respiratórias
            respiratory_symptoms = {"tosse", "falta de ar", "dispneia", "chiado", "catarro"}
            if any(any(r in symptom for r in respiratory_symptoms) for symptom in facts.sintomas):
                has_fever = any("febre" in s for s in facts.sintomas)
                if has_fever:
                    diagnoses.append({
                        "name": "Infecção respiratória",
                        "confidence": 0.85,
                        "reasons": ["Febre presente", "Sintomas respiratórios"]
                    })
                else:
                    diagnoses.append({
                        "name": "Síndrome respiratória",
                        "confidence": 0.70,
                        "reasons": ["Sintomas respiratórios presentes"]
                    })
            
            # Regras gastrointestinais
            gi_symptoms = {"náusea", "vômito", "diarreia", "dor abdominal", "azia"}
            if any(any(g in symptom for g in gi_symptoms) for symptom in facts.sintomas):
                has_fever = any("febre" in s for s in facts.sintomas)
                if has_fever:
                    diagnoses.append({
                        "name": "Gastroenterite",
                        "confidence": 0.80,
                        "reasons": ["Febre presente", "Sintomas gastrointestinais"]
                    })
                else:
                    diagnoses.append({
                        "name": "Síndrome gastrointestinal",
                        "confidence": 0.65,
                        "reasons": ["Sintomas gastrointestinais presentes"]
                    })
            
            # Regras de dor
            pain_symptoms = [s for s in facts.sintomas if "dor" in s]
            if pain_symptoms:
                if any("dor de cabeça" in s or "cefaleia" in s for s in pain_symptoms):
                    diagnoses.append({
                        "name": "Cefaleia primária",
                        "confidence": 0.70,
                        "reasons": ["Dor de cabeça como sintoma principal"]
                    })
                else:
                    diagnoses.append({
                        "name": "Síndrome dolorosa",
                        "confidence": 0.60,
                        "reasons": [f"Dor localizada: {', '.join(pain_symptoms[:2])}"]
                    })
            
            # Red flags
            if facts.red_flags:
                diagnoses.append({
                    "name": "CONDIÇÃO DE EMERGÊNCIA",
                    "confidence": 0.95,
                    "reasons": [f"Red flag presente: {', '.join(facts.red_flags)}"]
                })
            
            # Hipertensão
            if any("pressão alta" in f or "hipertensão" in f for f in facts.achados_exame):
                diagnoses.append({
                    "name": "Hipertensão arterial",
                    "confidence": 0.90,
                    "reasons": ["Pressão arterial elevada documentada"]
                })
            
            return diagnoses[:5]  # Top 5
            
        except Exception as e:
            logger.error(f"Erro no motor de diagnósticos: {e}")
            return []
    
    def identify_gaps(self, facts: ClinicalFacts, current_state: PanelsState) -> List[str]:
        """
        Identifica lacunas de informação importantes
        """
        try:
            gaps = []
            
            # Lacunas para dor
            pain_symptoms = [s for s in facts.sintomas if "dor" in s]
            if pain_symptoms:
                gaps.append("Caracterizar a dor: intensidade, duração, fatores de melhora/piora")
            
            # Lacunas para febre
            if any("febre" in s for s in facts.sintomas):
                gaps.append("Investigar: há quanto tempo tem febre? Temperatura máxima?")
            
            # Sintomas respiratórios
            respiratory_symptoms = {"tosse", "falta de ar", "dispneia", "chiado", "catarro"}
            if any(any(r in symptom for r in respiratory_symptoms) for symptom in facts.sintomas):
                gaps.append("Investigar: presença de expectoração, cor do escarro")
            
            # Sintomas GI
            gi_symptoms = {"náusea", "vômito", "diarreia"}
            if any(any(g in symptom for g in gi_symptoms) for symptom in facts.sintomas):
                gaps.append("Avaliar hidratação e características das evacuações")
            
            # Sinais vitais
            if not facts.achados_exame:
                gaps.append("Verificar sinais vitais completos")
            
            # Medicamentos
            if not facts.medicamentos:
                gaps.append("Investigar medicações em uso regular")
            
            # Alergias
            if not facts.alergias:
                gaps.append("Verificar alergias medicamentosas")
            
            return gaps[:6]  # Top 6
            
        except Exception as e:
            logger.error(f"Erro na identificação de lacunas: {e}")
            return []
    
    def suggest_management(self, facts: ClinicalFacts, current_state: PanelsState) -> List[str]:
        """
        Sugere condutas clínicas baseadas em regras
        """
        try:
            management = []
            
            # Red flags - prioridade máxima
            if facts.red_flags:
                management.append("🚨 AVALIAÇÃO IMEDIATA NECESSÁRIA")
                management.append("Considerar encaminhamento para emergência")
            
            # Manejo de febre
            if any("febre" in s for s in facts.sintomas):
                management.append("Antitérmico se temperatura > 37.8°C")
                management.append("Hidratação adequada")
            
            # Manejo de dor
            if any("dor" in s for s in facts.sintomas):
                management.append("Analgesia conforme intensidade (escala 0-10)")
            
            # Sintomas respiratórios
            respiratory_symptoms = {"tosse", "falta de ar", "dispneia", "chiado", "catarro"}
            if any(any(r in symptom for r in respiratory_symptoms) for symptom in facts.sintomas):
                management.append("Repouso relativo")
                management.append("Hidratação e umidificação do ambiente")
            
            # Sintomas GI
            gi_symptoms = {"náusea", "vômito", "diarreia"}
            if any(any(g in symptom for g in gi_symptoms) for symptom in facts.sintomas):
                management.append("Dieta leve e fracionada")
                management.append("Monitorar sinais de desidratação")
            
            # Hipertensão
            if any("pressão alta" in f or "hipertensão" in f for f in facts.achados_exame):
                management.append("Controle da pressão arterial")
                management.append("Orientações dietéticas (baixo sódio)")
            
            # Condutas gerais
            if not management or len(management) == 0:
                management.append("Tratamento sintomático")
            
            management.append("Retorno se piora ou persistência dos sintomas")
            
            return management[:8]  # Top 8
            
        except Exception as e:
            logger.error(f"Erro nas sugestões de conduta: {e}")
            return []
    
# Funções auxiliares para testes

def test_rules():
    """
    Teste básico das regras clínicas
    """
    from app.models import ClinicalFacts
    
    # Criar fatos de teste
    facts = ClinicalFacts(
        sintomas=["dor de cabeça", "febre", "tosse"],
        achados_exame=["pressão alta"],
        red_flags=["febre alta"]
    )
    
    # Testar motor de regras
    engine = ClinicalRuleEngine()
    
    diagnoses = engine.generate_diagnoses(facts)
    gaps = engine.identify_gaps(facts, PanelsState())
    management = engine.suggest_management(facts, PanelsState())
    
    print("=== Teste do Motor de Regras ===")
    print(f"Diagnósticos: {diagnoses}")
    print(f"Lacunas: {gaps}")
    print(f"Condutas: {management}")


if __name__ == "__main__":
    test_rules()