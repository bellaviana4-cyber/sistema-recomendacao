# Protocolo de avaliação — definido antes da comparação

## Protocolo comum: prever notas e recomendar listas
Usamos **5 folds externos** de avaliações, embaralhados com semente 2026: 80 mil notas no treino e 20 mil no teste. Os mesmos índices são gravados em `data/splits/`. Todos os modelos recebem exatamente esses conjuntos. A divisão aleatória mede recuperação de avaliações ocultadas de usuários existentes; não simula integralmente recomendar filmes futuros.

O ajuste usa uma única divisão interna 80/20 dentro de cada treino externo, semente 2026 + fold. Cada algoritmo testa somente três configurações. Selecionamos o **menor RMSE interno**, com desempate pela ordem da grade. Portanto, o ajuste não foi otimizado para Precisão@10. O teste externo nunca escolhe parâmetros. A configuração inicial também é avaliada, mas as conclusões usam a ajustada.

**RMSE**: $\sqrt{\frac{1}{n}\sum_{(u,i)\in teste}(r_{ui}-\hat r_{ui})^2}$. Penaliza mais erros grandes e está na escala das notas. O baseline prevê a média global do treino; contagens de popularidade não são notas.

**Precisão@10**: $|Top10_u\cap Relevantes_u|/10$, com nota real **≥ 4 no teste** como decisão operacional. Usuário elegível precisa de histórico no treino, pelo menos dez filmes candidatos e pelo menos um relevante elegível. Candidatos são todos os filmes presentes no treino, exceto os avaliados por aquela pessoa no treino. Empates de escores são resolvidos pelo menor ID. Nenhum corte de nota prevista encurta a lista.

Os três métodos usam os mesmos candidatos e usuários. A popularidade ordena contagens de treino. Filmes sem avaliação no teste não são acertos, embora sua relevância seja desconhecida: esta é uma avaliação offline conservadora e condicionada a preferências observadas. Exclusões por usuário e motivo são salvas. RMSE inclui todo o teste, inclusive casos de filme desconhecido; Precisão@10 exclui filmes fora do catálogo de treino.

Médias e desvios-padrão amostrais resumem os cinco folds. Esses folds compartilham treinos; diferenças médias e barras de desvio não estabelecem superioridade estatística.

## Configurações predefinidas

SVD: (50 fatores, 20 épocas, reg=0,02); (50,30,0,08); (100,30,0,08). Taxa 0,005 e semente 2026. KNN: (k,min_k,min_support) = (40,1,1), (20,3,3), (60,3,3), sempre user_based=True e cosine. A primeira configuração de cada grade é a inicial.

## Implementação e auditoria

As previsões de notas usam Surprise.predict/test. As matrizes de escores para ranking são calculadas em lote com os fatores/vieses aprendidos ou as mesmas similaridades/vizinhos do KNNBasic. Ordenação estável de vizinhos preserva empates da biblioteca; 200 pares por matriz são confrontados com predict, tolerância 1e-10. As listas desempatam por ID do filme.

Integridade e disjunção são verificadas por assert. Médias globais, contagens e similaridades usam exclusivamente treino. O código contém exemplos manuais RMSE=sqrt(2,5) e P@10=2/10. Os arquivos de elegibilidade registram todas as exclusões.

## Persistência

Índices zero-based referem-se à ordem original de u.data, cujo SHA256 está em data/splits/manifest.json. Divisões internas e externas ficam em NPZ; métricas/configurações, previsões e listas ficam em outputs/tables. Dados brutos e previsões individuais contendo notas originais não são redistribuídos no GitHub por restrição do README do MovieLens 100k. São recriados localmente na execução.

## Execução sem estado compartilhado

Cada notebook lê a base e importa src/projeto.py. O notebook 03 usa as tabelas produzidas no 02, não variáveis de outro kernel. Execute na ordem. O modo --inprocess executa um kernel IPython novo em um processo separado por notebook e captura mensagens Jupyter, sem sockets. É a alternativa usada no ambiente de entrega, que não permitiu abrir sockets TCP/IPC. O modo padrão usa nbclient com kernel separado.

## Exemplos

Usuários selecionados pelos quantis 10%,50%,90% de atividade do treino externo 1, menor ID em empate. Históricos completos ficam localmente em CSV. As listas são desse modelo de avaliação; não há treinamento final com toda a base.

## Exportação para auditoria

Após a execução, src/exportar_entrega.py cria CSV gzip de previsões externas sem notas ou timestamps originais, com row_id para recuperação local da nota, e exporta listas e auditoria de IDs desconhecidos. Isso preserva previsões reutilizáveis sem redistribuir a base original.
