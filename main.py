import os
import pandas as pd
import numpy as np
import joblib

# ==========================================
# CONFIGURAÇÃO DE AMBIENTE
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_INPUT = os.path.join(BASE_DIR, "data_consumos.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "models/DT_Ilicito_Modelo.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "previsao_ilicitos.xlsx")

def calcular_probabilidade_polinomial(x):
    """Cálculo heurístico baseado na função polinomial de 6º grau (Mestrado 2019)."""
    if x < 7:
        L = (-0.0003*x**6 + 0.0081*x**5 - 0.0814*x**4 + 0.3519*x**3 - 0.6084*x**2 + 0.4036*x) * 100
        return min(max(L, 0), 100)
    return 50

def processar_analise_ilicitos():
    if not os.path.exists(FILE_INPUT):
        print(f"[ERRO] Ficheiro de entrada não encontrado: {FILE_INPUT}")
        return

    df = pd.read_excel(FILE_INPUT).fillna(0)
    
    # --- MÉTODO A: LÓGICA HEURÍSTICA (REGRAS DE NEGÓCIO) ---
    resultados_prob = []
    
    for index, row in df.iterrows():
        # Filtro de Contratos Ativos
        if row['Estado Contrato'] in ['12 - Contrato em vigor', '13 - Reativado de inspeção']:
            prob_base = calcular_probabilidade_polinomial(row['NrIlicitos'])
            
            # Fator Consumo (Anomalias)
            consumo_medio = row['Consumo2018'] / 12
            if consumo_medio < 10 and row['Cortados'] == 'Cortado':
                prob_base = min(prob_base + 70, 100)
            
            resultados_prob.append(prob_base)
        else:
            # Caso especial: Contrato Suspenso com Consumo Ativo
            if row['Estado Contrato'] == '18 - Suspenso' and row['ConsumoAbril2019'] > 0:
                resultados_prob.append(100)
            else:
                resultados_prob.append(0)

    df['Probabilidade_Heuristica'] = resultados_prob

    # --- MÉTODO B: MACHINE LEARNING (PREVISÃO) ---
    if os.path.exists(MODEL_PATH):
        print("A carregar modelo de Machine Learning...")
        clf = joblib.load(MODEL_PATH)
        
        # Seleção de Features conforme treinado em 2019
        features = ['Estado Contrato', 'Consumo2018', 'ConsumoAbril2019', 'NrIlicitos', 'Cortados', 'EstadoHabitacao']
        X = df[features]
        
        df['Previsao_ML_Ilicito'] = clf.predict(X)
    
    # Exportação
    df.to_excel(OUTPUT_PATH, index=False)
    print(f"Análise concluída! Resultados em: {OUTPUT_PATH}")

if __name__ == "__main__":
    processar_analise_ilicitos()
