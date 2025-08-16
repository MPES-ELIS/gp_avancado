# Relatório – Gestão de Projetos Avançado
**ELIS - EQUIPE 1**

---

## Discentes
- André Lima  
- Raphael Marques  
- Felipe Maciel  
- Jaqueline Alexandre  
- Douglas Vasconcelos  

**Docente:** Aêda Monalliza Cunha de Sousa

---

## Sumário
- [Visão Geral do Projeto](#visão-geral-do-projeto)
- [Dados](#dados)
- [3. Metodologia](#3-metodologia)
  - [3.1 Trilhas de IA escolhidas e justificativa](#31-trilhas-de-ia-escolhidas-e-justificativa)
  - [3.2 Baselines, modelos, bibliotecas, hiperparâmetros, seed e features](#32-baselines-modelos-bibliotecas-hiperparâmetros-seed-e-features)
  - [3.3 Métricas de avaliação e justificativas](#33-métricas-de-avaliação-e-justificativas)
- [Resultados](#resultados)
  - [1) Riscos relevantes no sprint recomendado](#1-riscos-relevantes-no-sprint-recomendado-projeto-em-execução)
  - [2) Processo atual vs. com IA](#2-processo-atual-vs-com-ia)
  - [3) Baselines vs. modelo (síntese)](#3-baselines-vs-modelo-síntese)
  - [4) Riscos & mitigação](#4-riscos--mitigação)
  - [5) Monitoramento (produção)](#5-monitoramento-produção)
  - [6) Lições aprendidas (preliminar)](#6-lições-aprendidas-preliminar)

---

## Visão Geral do Projeto

**Objetivo:** Priorizar o backlog e planejar sprints com maior previsibilidade, maximizando valor entregue e reduzindo *spillover*, usando técnicas de *machine learning* (ML) e otimização sob incerteza.

**Escopo:** Duas trilhas complementares — (A) Priorização com *Learning-to-Rank* e consciência de dependências; (B) Gerador de Sprint que recomenda escopos com probabilidade de conclusão ≥ θ.

**Tempo total de desenvolvimento:** 2 semanas

**Contexto e restrições relevantes:**  
Backlog multi-produto/time, dependências técnicas entre itens, variação real de capacidade, bugs e interrupções.

---

## Dados

Foram utilizados a sprint do JIRA.  
Para esse treinamento foram priorizadas as sprints até o final do projeto, bem como estimadas.

---

## 3. Metodologia

### 3.1 Trilhas de IA escolhidas e justificativa

**Trilha A — Priorização (ranqueamento com dependências)**  
Ordenar o backlog maximizando valor e respeitando dependências.  
**Por quê:** escolher certo o que fazer primeiro é a maior alavanca de valor. Considerar dependências evita planos inviáveis e captura “valor encadeado”.

**Implementação neste projeto (com os dados disponíveis):**  
Como não havia histórico temporal (*snapshots*) suficiente para *Learning-to-Rank*, utilizamos um **modelo supervisionado de regressão (LightGBM)** para prever um **score de prioridade** a partir de *features* do grafo (*in/out-degree*, impacto *downstream*, centralidade) e **story points**.  
Em seguida aplicamos **ordenação topológica gulosa**: entre os itens desbloqueados, escolhemos o de maior `score_ml`.

> **Observação:** quando os *snapshots* históricos forem disponibilizados, migramos para **LambdaMART (LTR)** e avaliamos com **NDCG@10**.

**Trilha B — Gerador de Sprint (previsão + otimização sob incerteza)**  
Sugerir um escopo que **maximiza o valor** (peso vindo da Trilha A) com **confiabilidade operacional**: `P(on-time) ≥ θ`.  
**Por quê:** planejar apenas por soma de SP ignora a variação real de capacidade e o *overrun*; usar **previsões + probabilidades** reduz *spillover* e melhora a confiabilidade.

**Implementação neste projeto (com os dados B1…B4):**
- **Capacidade:** série de `SP Entregues` por `SprintID/TeamID` (trilhaB1) → distribuição preditiva (ex.: P10/P50/P90).  
- **Overrun por item:** `r = SP_real/SP_estimado` (trilhaB2) → LightGBM Regressor em `log(r)` (≈ lognormal).  
- **Risco de *spillover*:** a partir do histórico (trilhaB2) por faixas de tamanho e dependência externa; opcionalmente classificador calibrado.  
- **Valor:** *score* da Trilha A (arquivo `backlog_priorizado (1).csv`) como **peso**; *fallback* para `prioridade score` (trilhaB4).  
- **Otimização:** **Monte Carlo** (amostra capacidade e *overrun*) + **guloso com trocas locais** (ou ILP aproximado) para escolher conjunto `S` tal que `P(Σ SP_efetivo ≤ capacidade) ≥ θ`, respeitando deps/WIP.

---

### 3.2 Baselines, modelos, bibliotecas, hiperparâmetros, seed e features

#### Trilha A — Priorização (com os dados atuais)

**Baselines**
- **Só ROI:** ordena por `roi_ajustado` (ou `prioridade score`).  
- **ROI+Deps (heurístico):** `0.7·norm(ROI) + 0.3·norm(dep_impact)`; `dep_impact` = soma dos ROIs dos dependentes diretos.

**Modelo (ML de regressão)**
- **Algoritmo:** LightGBM Regressor (`objective='regression_l2'`, `learning_rate≈0.05`, `num_leaves≈31–63`, *early stopping*).  
- **Bibliotecas:** `lightgbm`, `scikit-learn`, `networkx`.  
- **Seed:** 42.  
- **Pós-processo:** **ordenação topológica gulosa** para respeitar dependências.

**Principais *features***  
`story_points`, `in_degree`, `out_degree`, `dep_impact`, `degree_centrality`.  
(Quando houver: deadlines/OKR, componente/time, idade do item, *lead time*.)

> **Migração planejada** para **LTR (LambdaMART)** quando houver *snapshots* temporais; avaliação com **NDCG@10** e **calibração isotônica** do score.

#### Trilha B — Gerador de Sprint (previsão + otimização)

**Baselines**
- **Bin-packing determinístico:** enche até `0,9×capacidade`, ignora incerteza.  
- **Velocidade média:** capacidade fixa = média dos 3 últimos sprints.

**Modelos preditivos**
- **Capacidade (SP por sprint):** ETS/ARIMA (*statsmodels*) ou Prophet; *fallback* LightGBM. Saída: **P10/P50/P90**.  
- **Overrun por item (`r = SP_real/SP_estimado`):** LightGBM Regressor em **`log(r)`** (assume lognormal).  
- **Spillover por item (risco):** LightGBM Classifier com `scale_pos_weight` e **calibração** (Platt/isotônica).

**Otimização do sprint**
- **Monte Carlo** (≈ 5k cenários): amostra capacidade e `r` por item → obtém `SP_efetivo`.  
- **Seleção:** fila de desbloqueados (DAG) + **ganho marginal** `valor_i / SP_efetivo` (com penalização por risco); **trocas locais**; alternativa **ILP** com *chance constraint* (quantil).

**Principais *features* (overrun/spillover)**
- Tamanho (SP), presença de dep. externa, idade, histórico de *spillover*/erro por tamanho; (quando disponível) componente, autor/time, interrupções/bugs, tipo (bug/feature/tech-debt).

---

### 3.3 Métricas de avaliação e justificativas

**Trilha A — Priorização**
- **NDCG@10:** qualidade do topo (≈ sprint).  
- **Uplift de valor (30d) vs. baseline:** ganho % de valor realizado sobre ROI+Deps.

**Trilha B — Gerador de Sprint**
- **Hit-rate on-time** para o limiar `θ`: % de sprints que fecham no prazo.  
- **Spillover%:** % dos SP planejados que rolam; métrica sintética de superalocação/erro.

---

## Resultados

### 1) Riscos relevantes no sprint recomendado (projeto em execução)
- **Probabilidade de sucesso estimada:** **97,1%**, acima de um limiar típico `θ=0,80`, indicando **baixo risco de atraso**.  
- **Utilização prevista:** **59,5%** (capacidade média **9,2 SP** vs. ~**5,5 SP** efetivos esperados), sugerindo **folga planejada** — opção conservadora para absorver variação/interruptões.

**Interpretação:** não foram identificados **riscos significativos** para o sprint corrente; os riscos residuais concentram-se em variações de *overrun* item-a-item e eventuais dependências externas, mas a **folga** e a **probabilidade** calculada mitigam esses riscos.

### 2) Processo atual vs. com IA
- **Antes:** seleção por julgamento + soma de SP, sem modelar dependências/variação → *spillover* frequente.  
- **Com IA:**  
  - **Trilha A** prioriza com consciência de **DAG**;  
  - **Trilha B** escolhe um escopo que **maximiza valor** com **meta de confiabilidade** (`P(on-time) ≥ θ`) e **valida capacidade** via previsão.

### 3) Baselines vs. modelo (síntese)
- **Bin-packing/Velocidade média** não controlam a probabilidade de conclusão e são sensíveis a *overrun*.  
- **Modelo proposto** entrega um plano com **`P(on-time) ≈ 97%`** e **folga explícita**, reduzindo risco de *spillover* frente aos baselines.  
  *(Números comparativos detalhados podem ser expandidos com mais histórico e simulações dedicadas por time.)*

### 4) Riscos & mitigação
- **Dados esparsos por item** (*features* limitadas) → manter modelo **parsimonioso** e atualizar mensalmente.  
- **Dependências externas** → sinalizar no input, **penalizar** no ganho marginal ou **bloquear** até confirmação.  
- **Mudanças de processo** → monitorar *drift* e recalibrar `θ`.

### 5) Monitoramento (produção)
- Painéis com **Hit-rate on-time**, **Spillover%**, **calibração de `P(on-time)`** e **importância de *features***.  
- **Guardrails:** impedir planos com `P(on-time) < θ`, alertar **subutilização crônica** (<65%) ou **sobrecarga** (>95%).

### 6) Lições aprendidas (preliminar)
- **Respeitar o DAG** e **calibrar probabilidade** são alavancas simples e potentes para confiabilidade.  
- Sem histórico temporal, **regressão + topo guloso** é um bom *proxy* de LTR; a migração para **LambdaMART** deve **elevar NDCG** e **valor** no topo.  
- **Folga proposital** é útil quando a variância de capacidade é alta; com mais maturidade, pode-se aproximar da borda (maior utilização) mantendo `θ`.

