# sistema-recomendacao

## Projeto final — Sistemas de Recomendação

**Laboratório de Estatística Aplicada — 2026/2 · Professor Victor Coscrato**  
**Isabella Viana Bambirra · Estatística/UFSCar**

O projeto usa o MovieLens 100k para estudar avaliações explícitas e comparar **SVD/FunkSVD** e **User-Based KNN**, com **popularidade** como referencial de listas e **média global** como referencial de previsão de notas. Metadados servem apenas para descrição e apresentação dos filmes.

### Relatório interativo

O [relatório em HTML](relatorio_sistemas_recomendacao_movielens.html) reúne metodologia, análise descritiva, explicações dos modelos, diagnósticos, comparação e conclusão. Para usar as abas, filtros e exportações, baixe o arquivo e abra-o no navegador. O relatório funciona sem conexão à internet; somente os links para fontes externas exigem acesso à rede.

### Ordem de leitura

1. [Análise descritiva](notebooks/01_analise_descritiva.ipynb): conferência da base, notas, perfis, atividade, filmes, gêneros, anos e esparsidade.
2. [Fatorização matricial](notebooks/02_fatorizacao_matricial_svd.ipynb): fatores, vieses, validação, ajuste limitado, erros e recomendações.
3. [User-Based KNN e comparação](notebooks/03_user_based_knn_comparacao.ipynb): cosseno, suporte, vizinhos, fallback, avaliação e conclusão.

Os três notebooks estão executados, com saídas, gráficos e interpretações específicas. Cada um inicia sua própria leitura e não usa variáveis da memória de outro notebook. O terceiro lê tabelas produzidas pelo segundo; execute na ordem.

### Resultados principais

Média ± desvio-padrão amostral de **cinco folds externos**, para os modelos ajustados exclusivamente dentro dos respectivos treinos:

| Método | RMSE ↓ | Precisão@10 ↑ |
|---|---:|---:|
| SVD | 0,9223 ± 0,0061 | 0,0707 ± 0,0037 |
| User-Based KNN | 1,0139 ± 0,0062 | 0,0675 ± 0,0121 |
| Média global | 1,1257 ± 0,0073 | — |
| Popularidade | — | 0,1332 ± 0,0038 |

O **SVD previu melhor as notas** em todos os folds. A **popularidade recuperou mais relevantes no top 10** em todos os folds; a personalização não superou esse baseline neste protocolo. Entre SVD e KNN, a diferença média de precisão é pequena e não estabelece superioridade estatística. O ajuste do SVD reduziu o RMSE, mas sua Precisão@10 caiu de 0,0743 para 0,0707. Isso ilustra que as duas métricas medem objetivos distintos.

O ranking avaliou **929, 925, 921, 930 e 923 usuários** nos folds 1 a 5, sempre os mesmos entre os métodos. RMSE usou as 20 mil avaliações de cada teste. Foram encontrados filmes desconhecidos no treino, mas nenhum usuário desconhecido; IDs e contagens constam em `outputs/tables/cold_start_ids.csv` e `fold_sizes.csv`.

### Protocolo

- Cinco folds externos aleatórios de avaliações, semente **2026**, compartilhados pelos modelos.
- Uma validação interna 80/20 por fold, semente **2026 + fold**, com três configurações por algoritmo. Seleção por **RMSE interno**, nunca pelo teste externo.
- SVD inicial: 50 fatores, 20 épocas, regularização 0,02, taxa 0,005. Selecionado nos cinco folds: 100 fatores, 30 épocas, regularização 0,08, taxa 0,005.
- KNNBasic inicial: `k=40, min_k=1, min_support=1`. Selecionado nos cinco folds: `k=60, min_k=3, min_support=3`, sempre `user_based=True` e `cosine`.
- Relevância: nota real **≥ 4 no teste**. Candidatos: filmes presentes no treino e ainda não avaliados pelo usuário naquele treino. Dez itens, sem corte de nota prevista; desempate por ID.
- Elegibilidade: histórico no treino, pelo menos dez candidatos e pelo menos um relevante elegível no teste. Exclusões registradas por pessoa e motivo.
- Precisão@10 é a média por usuário; não distingue posições dentro do conjunto de dez.

Leia [o protocolo completo](docs/protocolo_avaliacao.md) e [fontes e licença](docs/fontes_e_licenca.md).

### Reprodução

Ambiente validado: **Python 3.12.14**, NumPy 1.26.4 e Surprise 1.1.5. As versões principais estão em `requirements.txt`; o ambiente completo em `requirements-lock.txt` e as versões efetivamente usadas em `outputs/versions.json`.

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name python3 --display-name "Python 3 (MovieLens)"
python src/executar_notebooks.py
python src/verificar_projeto.py
python src/exportar_entrega.py
```

Windows/PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ipykernel install --user --name python3 --display-name "Python 3 (MovieLens)"
python src/executar_notebooks.py
python src/verificar_projeto.py
python src/exportar_entrega.py
```

Se a instalação da Surprise precisar compilar em uma plataforma sem wheel compatível, é necessário um compilador C/C++; no Windows, use os Build Tools do Visual Studio. A alternativa é executar no Linux/WSL com as mesmas versões. Não migre para NumPy 2 sem verificar a compatibilidade da extensão compilada.

Também é possível abrir os notebooks no VS Code ou Jupyter e usar **Restart Kernel and Run All** em cada um, na ordem. Num ambiente que não permita sockets Jupyter:

```bash
python src/executar_notebooks.py --inprocess
```

Esse modo foi usado na entrega: **InProcessKernelManager do IPython, um novo processo e um novo kernel por notebook**, com captura das mensagens de saída. O modo padrão utiliza `nbclient` e kernels externos. Fixar `OPENBLAS_NUM_THREADS=1` e `OMP_NUM_THREADS=1` aproxima o ambiente dos tempos reportados; tempos de processamento dependem do hardware e incluem verificações do lote.

A primeira leitura baixa automaticamente o ZIP oficial do MovieLens 100k. A execução requer acesso a `files.grouplens.org`. Dados brutos e notas individuais não são redistribuídos no repositório, conforme o README original. `src/exportar_entrega.py` disponibiliza previsões e recomendações derivadas em CSV gzip **sem notas/timestamps originais**; `row_id` identifica a linha em `u.data` para recuperar a nota localmente.

### Arquivos e auditoria

- `src/projeto.py`: leitura, integridade, modelos, folds, seleção, ranking e exemplos compartilhados.
- `data/splits/`: índices externos/internos NPZ e manifesto com SHA256 de `u.data`.
- `outputs/tables/`: métricas por fold, ajustes, configurações, exclusões, erros, recomendações, previsões derivadas e comparação.
- `outputs/figures/`: figuras PNG reutilizáveis.
- `outputs/verification.json`: resultado das verificações.

A execução recria as previsões completas com notas originais e os históricos dos usuários **localmente**, em arquivos ignorados pelo Git. As previsões/listas derivadas comprimidas podem ser auditadas em conjunto com o download oficial.

As verificações cobrem: integridade e junções, pares treino/teste disjuntos, validação interna isolada, igualdade de usuários elegíveis, top 10 sem duplicatas ou itens já avaliados, recálculo das métricas, exemplos manuais e equivalência do processamento em lote com `Surprise.predict`. Os modelos são da Surprise; o lote apenas calcula os mesmos escores com menos chamadas Python, sem mudar fórmulas ou seleção de vizinhos.

### Limitações e interpretação

A base é histórica, voluntária e já filtrada para ao menos vinte avaliações por pessoa. A divisão aleatória não reproduz recomendação temporal e não avalia adequadamente novos usuários. Itens sem nota no teste não contam como acertos, embora sua relevância seja desconhecida; popularidade pode se beneficiar da maior chance de esses filmes aparecerem no teste. Isso não prova que filmes menos populares sejam irrelevantes.

Média e desvio entre folds não são teste de superioridade estatística. Diversidade, novidade e serendipidade não foram medidas. Fatores latentes não são gêneros identificados e escores não são probabilidades. O diagnóstico do KNN inicial mostra listas frequentemente dominadas por filmes com poucos avaliadores; aumentar suporte e vizinhos mínimos ajudou na configuração escolhida internamente, sem eliminar as limitações do método.

Para previsão de estrelas, o SVD é a escolha sustentada pelos resultados; para o top 10 sob o desenho adotado, popularidade é o referencial mais forte.

Referência: Harper, F. M.; Konstan, J. A. (2015). *The MovieLens Datasets: History and Context*. DOI: https://doi.org/10.1145/2827872.
