# Resultados e interpretações

## 1. Base e análise descritiva

Foram conferidas 100.000 avaliações, 943 usuários e 1.682 filmes, sem duplicidade usuário–filme. A leitura manteve notas desconhecidas como ausentes, conferiu IDs e validou junções muitos-para-um. Metadados ausentes não foram imputados; datas de vídeo não foram usadas. A coleta vai de 19/09/1997 a 22/04/1998 (UTC).

A média das notas foi **3,5299**, a mediana **4** e o desvio-padrão **1,1257**. Notas 4 e 5 somam **55,375%**. Isso descreve avaliações voluntárias de filmes selecionados pelo público, não todas as possíveis preferências.

A matriz tem **1.586.126** pares possíveis, densidade **6,3047%** e **93,6953%** de células ausentes. Os modelos trabalham somente com notas observadas. A atividade por usuário e a popularidade dos filmes são desiguais, afetando o suporte disponível para estimar preferências.

**Star Wars (1977)** foi o filme mais avaliado, com **583** registros. Entre filmes com ao menos 50 notas — corte fixado antes do ranking — **Close Shave, A (1995)** liderou a média, **4,4911**, com **112** avaliações. Popularidade e média são conceitos diferentes. Gêneros são multirrótulo; idade, gênero e ocupação foram descritos contando pessoas distintas e comparados ao perfil ponderado pelas avaliações.

## 2. Desenho da avaliação

Cinco folds externos de 80.000/20.000 avaliações, semente 2026, compartilhados por todos os modelos. A busca interna usou 80/20 do treino externo e três configurações por algoritmo, selecionadas por RMSE interno. Toda média, popularidade e similaridade veio do treino correspondente.

RMSE compara previsão de estrelas com a nota observada. Precisão@10 conta filmes com nota real ≥ 4 no teste entre dez recomendados. Candidatos são filmes presentes no treino que o usuário não avaliou naquele treino. Elegibilidade exige histórico, dez candidatos e relevante elegível no teste; não há corte de nota prevista.

Os folds avaliaram **929, 925, 921, 930 e 923** usuários para ranking. As **14, 18, 22, 13 e 20** exclusões, respectivamente, decorrem de ausência de relevante elegível; nenhuma pessoa ficou sem histórico e ninguém teve menos de dez candidatos. Todas as 20.000 notas externas de cada fold entraram no RMSE, incluindo filmes desconhecidos no treino. IDs e suporte de cold start estão nas tabelas exportadas.

## 3. SVD

A implementação aprende média global, vieses de usuário e filme e fatores latentes por SGD regularizado nas notas observadas. Não preenche a matriz com zeros nem atribui gêneros aos fatores.

A busca interna escolheu **100 fatores, 30 épocas, regularização 0,08 e taxa 0,005** nos cinco folds. A configuração inicial usava 50 fatores, 20 épocas e regularização 0,02. O RMSE médio caiu de **0,9338** para **0,9223**, mas a Precisão@10 passou de **0,0743** para **0,0707**. Como a seleção foi por RMSE, esse comportamento não é contraditório: erro de notas e recuperação de relevantes diferem.

As notas 1 foram as mais difíceis, com RMSE **1,8842** e previsão média **2,7532**. O erro em notas 4 foi **0,5502**. Essa tendência de aproximar extremos do centro é compatível com o compartilhamento de informação e a regularização. Por atividade do usuário, o RMSE foi **0,9990** até 30 notas de treino, **0,9462** entre 31 e 100 e **0,9027** acima de 100. É uma associação descritiva, sem interpretação causal.

## 4. User-Based KNN

`KNNBasic`, cosseno e `user_based=True`; somente filmes comuns ao par sustentam a similaridade. Vizinhos precisam ter avaliado o filme previsto, e somente pesos positivos entram na média. `min_k` limita insuficiência de vizinhos; fallback é a média global.

A busca interna escolheu **k=60, min_k=3, min_support=3** nos cinco folds. Inicialmente os parâmetros eram (40,1,1). O RMSE caiu de **1,0167** para **1,0139**; a Precisão@10 subiu de **0,0035** para **0,0675**.

O diagnóstico das listas iniciais mostrou que **82,98% a 99,98%** dos itens recomendados por fold tinham até cinco avaliações no treino. Filmes com nota estimada extrema, sustentada por poucos avaliadores, dominavam o topo e raramente apareciam como relevante no teste do usuário. No modelo ajustado, essa proporção caiu para **8,30% a 29,79%**. Como três parâmetros mudaram conjuntamente, não atribuimos todo o ganho a um só. O diagnóstico não foi usado para mudar os parâmetros após olhar o teste.

## 5. Comparação final

Média ± desvio-padrão entre folds:

| Método | RMSE ↓ | Precisão@10 ↑ |
|---|---:|---:|
| SVD ajustado | 0,9223 ± 0,0061 | 0,0707 ± 0,0037 |
| KNN ajustado | 1,0139 ± 0,0062 | 0,0675 ± 0,0121 |
| Média global | 1,1257 ± 0,0073 | — |
| Popularidade | — | 0,1332 ± 0,0038 |

**SVD foi o melhor para prever notas, em todos os folds. Popularidade recuperou mais relevantes, em todos os folds. A personalização não superou o baseline de popularidade.** Os personalizados ficaram próximos na média de precisão, com maior variação entre folds no KNN e empate no fold 4. Nenhuma dessas diferenças é apresentada como superioridade estatística.

Precisão 0,1332 significa cerca de **1,332 acertos observados por lista de dez**; SVD e KNN recuperaram cerca de **0,707** e **0,675**, respectivamente. A métrica não distingue a posição dos acertos na lista. Muitos filmes recomendados podem ser relevantes sem terem nota no teste, então não contam como acertos observados.

O SVD treinou em média em aproximadamente **1,20 s** e avaliou em **0,56 s** por fold; o KNN treinou em **0,31 s** e avaliou em **5,34 s**. Esses tempos são específicos do ambiente e incluem processamento em lote e verificações; a busca interna tem custo separado nas tabelas. KNN treina rapidamente, mas consultar vizinhos para todo o catálogo custa mais. SVD utiliza fatores compactos e previsões por produto de vetores. Popularidade é barata e transparente.

## 6. Escolha e limitações

Para prever notas, preferimos **SVD**; para recuperar relevantes no top 10 sob este protocolo, **popularidade** é o ponto de partida mais forte. KNN oferece explicação por vizinhos, mas teve maior erro e listas menos estáveis.

Essa conclusão está condicionada à amostra histórica, ao catálogo e à divisão aleatória. O desenho não representa plenamente recomendação futura ou cold start real. Popularidade pode se beneficiar da maior probabilidade de filmes conhecidos terem nota no teste; ausência de nota não prova irrelevância. Treinos de folds se sobrepõem, e a busca foi pequena. Diversidade, novidade e serendipidade não foram medidas. Metadados não foram usados como preditores.

## 7. Verificações e entrega

Os notebooks foram executados em kernels IPython novos, um processo por notebook, com saídas preservadas. A alternativa em processo foi usada porque o ambiente bloqueou sockets TCP/IPC; o executor também oferece modo padrão `nbclient`.

A auditoria confirmou integridade, pares disjuntos, validação interna isolada, mesmos usuários elegíveis, listas de dez sem itens repetidos ou do histórico, métricas recalculadas das previsões externas e equivalência de escores em lote com Surprise. Exemplos manuais: RMSE=√2,5 e Precisão@10=2/10. Dados brutos e notas originais permanecem fora da redistribuição; previsões derivadas e índices salvos permitem reconstrução com o download oficial.

Não foi gerado relatório HTML nesta etapa. Tabelas e figuras estão organizadas para a sequência descritiva → SVD → KNN → avaliação/comparação → conclusão.
