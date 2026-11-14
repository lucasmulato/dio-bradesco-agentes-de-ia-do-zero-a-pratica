# Detecção de Fraude em Cartão de Crédito 💳

> Projeto de Machine Learning para detectar transações fraudulentas - um problema real com apenas 0,17% de fraude, onde acurácia engana.

### 🔗 Dataset
Público: [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) - 284.807 transações, carregado via URL direta.

### 🎯 O Problema
Em fraude, se eu criar um modelo burro que sempre prevê "não é fraude", eu tenho 99,83% de acurácia. Por isso este projeto NÃO usa acurácia.

**Métricas que importam:**
- **Recall (Fraude):** Quantas fraudes eu consegui pegar? Prioridade máxima.
- **Precision:** Das que eu alertei como fraude, quantas eram fraude mesmo?
- **AUC-PR:** Melhor que ROC para dados desbalanceados.
- **Matriz de Confusão**

### 🧪 O Que Foi Feito (5 Passos)

**1. Preparação:** Normalização de `Amount` e `Time`. Features `V1-V28` já são PCA.
**2. Split Estratificado:** `stratify=y` para manter 0,17% em treino e teste.
**3. Modelos Comparados:**
- Logistic Regression (`class_weight='balanced'`) - Baseline
- Random Forest (`balanced_subsample`) - Bom recall, lento
- XGBoost (`scale_pos_weight=577`) - **Vencedor**

**4. Ajuste de Limiar:** O padrão 0.5 não serve para banco. Baixei o limiar de 0.5 para **0.21** para garantir Recall > 85%, aceitando mais falsos positivos. Justificativa: no banco, é melhor bloquear 10 transações boas do que deixar 1 fraude de R$ 5.000 passar.

**5. Explicabilidade com SHAP:** Mostra por que o modelo marcou como fraude. Features `V14, V17, V10, V12` foram as mais importantes.

### 📊 Resultados Finais

| Modelo | Recall Fraude | Precision | AUC-PR |
| :--- | :--- | :--- | :--- |
| Logistic | 0.89 | 0.06 | 0.71 |
| Random Forest | 0.78 | 0.85 | 0.82 |
| **XGBoost (com limiar 0.21)** | **0.88** | **0.32** | **0.85** |

### 🚀 Como Rodar

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_modelagem_fraude.ipynb
