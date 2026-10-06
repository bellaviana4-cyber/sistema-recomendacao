# Revisão de aderência ao material do professor

Material: Victor Coscrato, Sistemas de Recomendação, Laboratório de Estatística Aplicada — 2026/2. Revisão de 05/10/2026.

## Correspondência com a aula e a tarefa

| Páginas | Conteúdo do professor | Correspondência no projeto |
|---|---|---|
| 12–16 | Matriz usuário × item, ausências, feedback, esparsidade e seleção das avaliações | Notebook 01; ausências preservadas e limites da amostra discutidos |
| 20–21 | Cosseno e média ponderada de vizinhos que avaliaram o item | Notebook 03; KNNBasic, user_based=True, cosine; suporte e fallback explicitados |
| 23–24 | Fatores latentes, vieses, perda regularizada e SGD | Notebook 02; Surprise SVD com vieses e SGD nas notas observadas |
| 27–29 | Objetivos distintos, RMSE e Precision@k | Protocolo comum nos notebooks 02 e 03; RMSE de estrelas e listas de dez |
| 31 | Diversidade, novidade e serendipidade | Discussão de limites, sem afirmar que essas propriedades foram medidas |
| 33–35 | MovieLens 100k, exploração e bibliotecas | Notebook 01 e Surprise nos modelos |
| 36 | Exploração; SVD; User-Based KNN; RMSE com CV; Precisão@10; comparação | Atendidos pelos três notebooks executados |

A página 34 cita filtragem por conteúdo como possibilidade; a tarefa específica da página 36 exige SVD e User-Based KNN. Não há necessidade de acrescentar conteúdo, híbridos, BPR, Item-Based KNN, MAE, Recall ou nDCG para cumprir esta entrega.

## Decisões adicionais do projeto

Cinco folds, semente 2026, relevância definida como nota ≥ 4, catálogo de candidatos do treino, elegibilidade e pequena busca interna são decisões operacionais do projeto. O material não fixa esses detalhes; eles concretizam uma avaliação reproduzível. A Precisão@10 segue a fórmula da página 29 e não depende da posição dos acertos dentro do top 10. A busca escolhe por RMSE; os resultados de ranking avaliam esse procedimento, não o melhor ranking possível de cada algoritmo.

## Ajustes aplicados

1. O tempo de ajuste dos baselines anteriormente media apenas a ausência de um modelo a ajustar. Agora inclui o cálculo da média ou das contagens. Leitura dos dados, construção comum do Trainset e busca interna ficam fora desse tempo. Os resultados de acurácia não dependem dessa correção.
2. A função de perda da página 24 foi explicitada no notebook SVD. A Surprise regulariza nas atualizações por avaliação; o coeficiente não deve ser transferido automaticamente entre implementações com convenções diferentes.
3. Substituída a expressão “avaliação conservadora” por uma descrição da avaliação incompleta e da seleção de preferências observadas. Não se estabelece um limite inferior universal para a precisão real.
4. Esclarecido que cossenos negativos não surgem com as notas positivas usadas aqui. A exclusão de similaridades não positivas é uma regra da biblioteca.
5. A auditoria verifica também que todos os filmes recomendados pertencem ao catálogo de treino e que os usuários das listas coincidem com os elegíveis.
6. Corrigido o título do README.

## Conclusão metodológica

Não foi encontrado motivo para mudar algoritmos, folds, candidatos ou seleção de parâmetros. A conclusão depende do objetivo: SVD prevê melhor as notas observadas; popularidade recupera mais relevantes observados no top 10 neste protocolo. Isso não prova superioridade universal, causal ou estatística. A divisão temporal e métricas adicionais podem ser extensões posteriores, não correções necessárias da atividade atual.

## Validação após a revisão

Os três notebooks foram reexecutados integralmente, na ordem, com um novo kernel e processo por notebook. A auditoria ampliada passou. RMSE e Precisão@10, médias e desvios entre folds, foram reproduzidos sem alteração em relação à entrega anterior. Tempos foram medidos novamente; previsões e listas derivadas foram reexportadas.
