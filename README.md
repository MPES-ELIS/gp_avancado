# Priorização & Planejamento com IA — ELIS (Equipe 1)

> **Backlog ranking (Trilha A) + Sprint generator (Trilha B)** com ML e otimização sob incerteza.

---

## 🗂 Estrutura do Projeto

```
└── gp_avancado-main
    ├── .devcontainer
    │   ├── devcontainer.json
    │   └── icon.svg
    ├── .gitignore
    ├── LICENSE
    ├── README.md
    ├── data
    │   ├── Backlog - trilhaB1.csv
    │   ├── Backlog - trilhaB2.csv
    │   ├── Backlog - trilhaB3.csv
    │   ├── Backlog - trilhaB4.csv
    │   ├── backlog_priorizado.csv
    │   ├── sprint_planejado.csv
    │   ├── sprint_planejado_metrics.txt
    │   └── trilha_a.csv
    ├── notebooks
    │   ├── trilhaA.ipynb
    │   └── trilhaB.ipynb
    ├── relatório.md
    └── requirements.txt
```

- **`data/`**: CSVs de entrada/saída (histórico, backlog, resultados).
- **`notebooks/`**: `trilhaA.ipynb` (priorização), `trilhaB.ipynb` (geração de sprint).
- **`relatório.md`**: relatório técnico do projeto.
- **`requirements.txt`**: dependências mínimas; veja também **Requisitos (completos)** abaixo.

---

## 🚀 Visão Rápida

- **Trilha A**: calcula *scores* com ML (LightGBM), extrai *features* de grafo (dependências) e gera uma **ordem topológica gulosa** respeitando bloqueios. Saída principal: `data/backlog_priorizado.csv`.
- **Trilha B**: prevê **capacidade** do time, **overrun** item-a-item e simula cenários (*Monte Carlo*) para recomendar um **sprint** com `P(on-time) ≥ θ`. Saídas: `data/sprint_planejado.csv` + `data/sprint_planejado_metrics.txt`.

---

## 🧰 Requisitos

### Python
- **Python 3.10+** recomendado.

### Dependências
O arquivo `requirements.txt` do repositório contempla o básico (widgets, pandas, matplotlib, torch).  
Para executar **todas** as funcionalidades dos notebooks, instale também as dependências abaixo:

```
pip install -r requirements.txt
pip install ipykernel>=6.29 jupyter>=1.0 lightgbm>=4.3.0 networkx>=3.2 scikit-learn>=1.3.0 scipy>=1.11 statsmodels>=0.14
```

> Se tiver problemas ao instalar **LightGBM**, tente:
> - Windows: `pip install lightgbm --prefer-binary`
> - Linux: garanta `build-essential`/compiladores instalados
> - Alternativa: use uma imagem Docker com esses pacotes pré-instalados

### Ambiente (opcional)
```
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
pip install ipykernel>=6.29 jupyter>=1.0 lightgbm>=4.3.0 networkx>=3.2 scikit-learn>=1.3.0 scipy>=1.11 statsmodels>=0.14
python -m ipykernel install --user --name gp-avancado
```

---

## 📦 Dados de Entrada (padrões no diretório `data/`)

- `Backlog - trilhaB1.csv` — **Capacidade por sprint/time** (`TeamID`, `SprintID`, `SP Entregues`)
- `Backlog - trilhaB2.csv` — **Histórico por item** (`issueid`, `sprintid`, `storypoints estimados`, `storypoints realizados`, `spillover (dep externa)`)
- `Backlog - trilhaB3.csv` — **Capacidade por dev** (`developerid`, `horas_disponiveis`, `senioridade`) *(opcional)*
- `Backlog - trilhaB4.csv` — **Backlog atual** (`issueid`, `prioridade score`, `storypoints`, `dependencias`)
- `backlog_priorizado.csv` — **Valor da Trilha A** (preferir `score_ml`; *fallback*: `roi_ajustado`)

> **Esquema de dependências**: a aresta é **`dep → item`** (o pré-requisito destrava o dependente). Dependências **externas** (fora do backlog atual) podem ser bloqueadas ou penalizadas (configurável no notebook).

---

## ▶️ Como Executar

### Trilha A — Priorização
1. Abra `notebooks/trilhaA.ipynb` (kernel `gp-avancado`).
2. Configure o caminho de `data/` se necessário.
3. **Execute todas as células**.  
   **Saída**: `data/backlog_priorizado.csv` com colunas-chave:
   - `rank`, `id`, `roi_ajustado`, `dep_impact`, `score_ml`, `deps_ok`, `deps`, `story_points`, `in_degree`, `out_degree`, `degree_centrality`.

### Trilha B — Gerador de Sprint
1. Abra `notebooks/trilhaB.ipynb`.
2. Defina os **parâmetros** principais (célula de config):
   - `THETA` (nível de serviço, ex.: `0.80`)
   - `N_SCENARIOS` (ex.: `5000`)
   - Política para **dependência externa** (bloquear/penalizar)
3. **Execute todas as células**.  
   **Saídas**:
   - `data/sprint_planejado.csv` — conjunto recomendado do sprint.
   - `data/sprint_planejado_metrics.txt` — métricas agregadas (ex.: `P(on-time)`, utilização, etc.)

---

## 📊 Métricas & Interpretação

- **Trilha A**
  - `NDCG@10` (quando LTR disponível)
  - *Uplift* de valor em 30 dias vs. heurístico `ROI+Deps`
- **Trilha B**
  - **`P(on-time)`** do plano recomendado (alvo ≥ `THETA`)
  - **Utilização de capacidade**
  - **Spillover%** esperado

> Exemplo (amostra): `P(on-time) ≈ 97%`, utilização ≈ `59,5%`, indicando folga planejada para absorver variação/interruptões.

---

## ⚙️ Parâmetros Importantes

- `THETA` (nível de serviço): 0.7–0.9 (típico 0.8)
- `N_SCENARIOS` (Monte Carlo): 2k–10k (típico 5k)
- Critério de seleção: `valor_i / SP_efetivo` (com penalização por risco de *spillover*)
- Regras de negócio: WIP máximo, % mínimo de bugs/tech-debt, política de dependência externa

---

## 🧪 Reprodutibilidade

- Use *split* temporal e `GroupKFold` por time/produto para evitar vazamento (quando treinar LTR).  
- Fixe `SEED=42` nos modelos.  
- Documente versões: `pip freeze > freeze.txt`.

---

## 🩺 Troubleshooting

- **LightGBM não instala**: use `--prefer-binary` ou uma imagem com `gcc/g++` (Linux).  
- **`ModuleNotFoundError` (statsmodels/networkx/sklearn)**: instale as dependências adicionais listadas acima.  
- **Colunas com espaços/acentos**: renomeie para *snake_case* (ex.: `horas_disponiveis`).  
- **Ciclos em dependências**: o notebook relata e resolve removendo a aresta de menor valor; ideal é **corrigir no dado**.

---

## 🗺 Roadmap

- Migrar Trilha A para **LambdaMART (LTR)** com **NDCG@10**.  
- **ILP/CP-SAT** com *chance constraints* (ou quantis) para seleção ótima global.  
- Suporte a **multi-recursos** (FE/BE/QA) e *skills*.  
- Conectores prontos (Jira/ClickUp/GitHub) e *feature store*.

---

## 🤝 Contribuição

1. Abra uma *issue* descrevendo a mudança.  
2. Crie *branch* (`feat/...` ou `fix/...`).  
3. Faça *PR* com descrição, screenshots e resultados.  
4. Garantir execução limpa dos notebooks.

---

## 📄 Licença

Defina a licença do projeto (ex.: MIT).

---

## 👤 Autoria

ELIS — Equipe 1  
Discentes: André Lima, Raphael Marques, Felipe Maciel, Jaqueline Alexandre, Douglas Vasconcelos  
Docente: Aêda Monalliza Cunha de Sousa

