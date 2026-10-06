# Recomendação de filmes com SVD e User-Based KNN

Análise do **MovieLens 100k** e comparação de métodos para prever notas e recomendar filmes. O projeto investiga duas perguntas: qual modelo estima melhor as avaliações dos usuários e qual método recupera mais filmes relevantes em uma lista de dez recomendações?

**Autora:** Isabella Viana Bambirra — graduação em Estatística na UFSCar  
**Disciplina:** Laboratório de Estatística Aplicada — 2026/2  
**Professor:** Victor Coscrato

[Relatório interativo](relatorio_sistemas_recomendacao_movielens.html) · [Notebooks](notebooks/) · [Resultados](docs/resultados_e_interpretacoes.md) · [Protocolo de avaliação](docs/protocolo_avaliacao.md)

## O que foi desenvolvido

- Análise descritiva da base: integridade, distribuição das notas, atividade dos usuários, popularidade dos filmes, perfis e esparsidade.
- Fatorização matricial com **SVD da Surprise**, com vieses e fatores latentes aprendidos nas avaliações observadas.
- Filtragem colaborativa com **User-Based KNN**, usando `KNNBasic` e similaridade cosseno.
- Comparação com **média global**, para previsão de notas, e **popularidade**, para recomendação de listas.
- Avaliação externa em cinco folds, ajuste de parâmetros dentro do treino e diagnósticos de erros, suporte e recomendações.

Os notebooks estão executados e incluem gráficos, tabelas e interpretações. O relatório reúne essas análises em uma apresentação interativa.

## A base

| Característica | MovieLens 100k |
|---|---|
| Avaliações | 100.000 |
| Usuários | 943 |
| Filmes | 1.682 |
| Escala das notas | 1 a 5 estrelas |
| Período das avaliações | 19/09/1997 a 22/04/1998, em UTC |
| Células sem avaliação na matriz usuário × filme | 93,69% |

Uma nota ausente representa uma preferência desconhecida. Os modelos usam somente as avaliações observadas; a matriz não é preenchida com zeros. Idade, gênero, ocupação e características dos filmes são usados na análise descritiva, sem entrar como preditores.

## Principais resultados

A tabela resume a **média e o desvio-padrão amostral dos cinco folds externos**, usando as configurações ajustadas por validação interna.

| Método | RMSE (estrelas) ↓ | DP do RMSE (milésimos de estrela) | Precisão no top 10 (%) ↑ | DP da precisão (p.p.) |
|---|---:|---:|---:|---:|
| SVD ajustado | 0,92 | 6,09 | 7,06 | 0,36 |
| User-Based KNN ajustado | 1,01 | 6,15 | 6,74 | 1,21 |
| Média global | 1,12 | 7,30 | — | — |
| Popularidade | — | — | 13,32 | 0,38 |

**Como ler:** menor RMSE indica melhor previsão de notas; maior precisão indica mais relevantes observados entre os dez recomendados. O DP descreve a dispersão entre folds, não um intervalo de confiança. Apenas o DP do RMSE está em milésimos de estrela: 1.000 milésimos equivalem a 1 estrela. O traço indica uma métrica não aplicável.

Os valores de apresentação são truncados em duas casas decimais. Os cálculos mantêm a precisão original, disponível em [comparison_summary.csv](outputs/tables/comparison_summary.csv); parâmetros como a taxa de aprendizagem 0,005 são preservados exatamente.

**O SVD apresentou o menor RMSE em todos os folds. A popularidade recuperou mais relevantes observados no top 10 em todos eles.** Assim, a personalização não superou esse baseline no protocolo adotado. A precisão de 13,32% da popularidade corresponde a aproximadamente 1,33 acertos observados por lista.

O ajuste do SVD reduziu o erro de previsão, mas também reduziu a precisão das listas. No KNN, ambas as métricas melhoraram. Esses resultados mostram que prever notas e ordenar recomendações são objetivos distintos; diferenças médias, por si só, não demonstram superioridade estatística.

## Como acessar o relatório

1. Abra [relatorio_sistemas_recomendacao_movielens.html](relatorio_sistemas_recomendacao_movielens.html).
2. No GitHub, use **Download raw file** para baixar o arquivo.
3. Abra o HTML em um navegador.

O relatório funciona localmente, sem instalação e sem conexão à internet. Os links para fontes externas exigem acesso à rede. As abas apresentam metodologia, análise descritiva, SVD, User-Based KNN, avaliação e comparação e conclusão. Há seleção de fold, configuração e usuários de exemplo, pesquisa de filmes e exportação de tabelas derivadas.

## Notebooks e ordem de execução

| Etapa | Notebook | Conteúdo |
|---|---|---|
| 1 | [Análise descritiva](notebooks/01_analise_descritiva.ipynb) | Integridade, notas, usuários, filmes, gêneros, anos e esparsidade |
| 2 | [Fatorização matricial — SVD](notebooks/02_fatorizacao_matricial_svd.ipynb) | Modelo, busca interna, resultados externos, erros e recomendações |
| 3 | [User-Based KNN e comparação](notebooks/03_user_based_knn_comparacao.ipynb) | Similaridades, vizinhos, fallback, diagnósticos e comparação final |

Execute na ordem acima. Cada notebook lê a base e importa o código compartilhado, sem depender de variáveis deixadas na memória por outro notebook. O terceiro utiliza tabelas produzidas pelo segundo.

## Desenho da avaliação

Os mesmos cinco folds são usados por todos os métodos: **80.000 avaliações para treino e 20.000 para teste** em cada rodada. Dentro de cada treino externo, uma divisão 80/20 compara três configurações por algoritmo. A seleção considera o menor RMSE interno; o teste externo não participa da escolha de parâmetros.

| Modelo | Configuração inicial | Configuração selecionada nos cinco folds |
|---|---|---|
| SVD | 50 fatores, 20 épocas, regularização 0,02 | 100 fatores, 30 épocas, regularização 0,08 |
| User-Based KNN | `k=40, min_k=1, min_support=1` | `k=60, min_k=3, min_support=3` |

A taxa de aprendizagem do SVD é 0,005. O KNN usa `user_based=True` e similaridade `cosine` em ambas as configurações.

Para as listas, um filme é considerado relevante quando recebe **nota real ≥ 4 no teste**. Os candidatos são filmes presentes no treino e ainda não avaliados pelo usuário naquele treino. Cada lista contém dez itens; empates são resolvidos pelo menor ID.

Participam usuários com histórico no treino, pelo menos dez candidatos e ao menos um relevante elegível no teste. O ranking avaliou **929, 925, 921, 930 e 923 usuários** nos folds 1 a 5, respectivamente, sempre os mesmos entre SVD, KNN e popularidade. O RMSE considera todas as 20.000 notas do teste de cada fold.

As fórmulas, sementes, regras de elegibilidade, tratamento de casos desconhecidos e critérios de desempate estão no [protocolo completo](docs/protocolo_avaliacao.md).

## Como reproduzir

### Preparar o ambiente

O ambiente de execução registrado usa **Python 3.12.14**, NumPy 1.26.4 e Surprise 1.1.5. Consulte [requirements.txt](requirements.txt), [requirements-lock.txt](requirements-lock.txt) e [outputs/versions.json](outputs/versions.json).

Clone o repositório e entre na pasta:

```bash
git clone https://github.com/bellaviana4-cyber/sistema-recomendacao.git
cd sistema-recomendacao
```

**Windows / PowerShell:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ipykernel install --user --name python3 --display-name "Python 3 (MovieLens)"
```

**Linux / macOS:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name python3 --display-name "Python 3 (MovieLens)"
```

### Executar e verificar

Com o ambiente ativado:

```bash
python src/executar_notebooks.py
python src/verificar_projeto.py
python src/exportar_entrega.py
```

O primeiro comando executa os notebooks em ordem. O segundo verifica a integridade e os resultados. O terceiro exporta previsões e recomendações derivadas para auditoria.

Também é possível abrir os notebooks no VS Code ou Jupyter e executar todas as células, reiniciando o kernel entre notebooks. Para ambientes que não permitem sockets Jupyter, o executor oferece:

```bash
python src/executar_notebooks.py --inprocess
```

<details>
<summary>Notas sobre instalação e processamento</summary>

A Surprise pode exigir um compilador C/C++ quando não há wheel compatível. No Windows, uma opção é usar os Build Tools do Visual Studio ou executar em Linux/WSL. A extensão compilada foi validada com NumPy 1.x.

O modo padrão usa `nbclient` e kernels externos. O modo `--inprocess` cria um kernel IPython em um processo separado para cada notebook.

Os tempos registrados dependem do hardware e da implementação. Definir `OPENBLAS_NUM_THREADS=1` e `OMP_NUM_THREADS=1` limita o paralelismo e aproxima o ambiente dos tempos reportados. Treinamento, avaliação e busca interna têm escopos distintos, descritos no protocolo.

</details>

A primeira leitura baixa o ZIP oficial do MovieLens 100k; essa etapa exige conexão com `files.grouplens.org`. Para consultar os resultados já salvos ou usar o relatório, não é necessário treinar novamente os modelos.

## Organização do repositório

| Caminho | Finalidade |
|---|---|
| [notebooks/](notebooks/) | Análises executadas e documentadas |
| [src/projeto.py](src/projeto.py) | Leitura, integridade, modelos, seleção, ranking e exemplos |
| [src/executar_notebooks.py](src/executar_notebooks.py) | Execução ordenada dos notebooks |
| [src/verificar_projeto.py](src/verificar_projeto.py) | Verificações de dados, avaliação e resultados |
| [src/exportar_entrega.py](src/exportar_entrega.py) | Exportação dos resultados derivados |
| [data/splits/](data/splits/) | Índices de treino/teste e manifesto da base |
| [outputs/tables/](outputs/tables/) | Métricas, configurações, erros, exclusões, previsões e listas |
| [outputs/figures/](outputs/figures/) | Gráficos das análises |
| [docs/](docs/) | Protocolo, resultados, revisão metodológica e fontes |
| [relatorio_sistemas_recomendacao_movielens.html](relatorio_sistemas_recomendacao_movielens.html) | Relatório interativo standalone |

## Verificações

As verificações registradas em [outputs/verification.json](outputs/verification.json) foram aprovadas. Elas cobrem integridade da base, separação entre treino e teste, isolamento da validação interna, igualdade de usuários elegíveis, listas sem duplicatas ou itens do histórico, recálculo das métricas e equivalência dos escores em lote com a Surprise.

O processamento em lote reutiliza os mesmos fatores, vieses, similaridades e regras dos modelos. As verificações comparam seus escores com `Surprise.predict`.

## Limitações

A base é histórica e reúne usuários voluntários com pelo menos vinte avaliações. A divisão aleatória mede recuperação de avaliações ocultadas, sem reproduzir uma avaliação temporal ou representar adequadamente novos usuários.

Filmes sem nota no teste não contam como acertos, embora sua relevância seja desconhecida. A elegibilidade depende de existir um relevante observado, e filmes populares podem ter maior chance de aparecer no teste. A busca de parâmetros foi pequena e selecionada por RMSE; diversidade, novidade e serendipidade não foram avaliadas.

Essas condições delimitam as conclusões. Fatores latentes não foram interpretados como gêneros, escores não são probabilidades e os desvios entre folds não demonstram superioridade estatística.

## Dados e referências

Os dados brutos, notas originais e históricos individuais completos não são redistribuídos neste repositório. A execução os recupera localmente a partir do download oficial. As previsões exportadas não incluem notas ou timestamps originais; `row_id` permite recuperar a avaliação localmente na ordem de `u.data`. Consulte [fontes e licença](docs/fontes_e_licenca.md) para as condições de uso dos dados.

- [MovieLens 100k — GroupLens](https://grouplens.org/datasets/movielens/100k/).
- Harper, F. M.; Konstan, J. A. (2015). *The MovieLens Datasets: History and Context*. [DOI: 10.1145/2827872](https://doi.org/10.1145/2827872).
- [Surprise: fatorização matricial](https://surprise.readthedocs.io/en/stable/matrix_factorization.html), [KNN](https://surprise.readthedocs.io/en/stable/knn_inspired.html) e [similaridades](https://surprise.readthedocs.io/en/stable/similarities.html).
