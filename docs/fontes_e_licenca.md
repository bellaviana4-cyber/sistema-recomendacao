# Fontes e licença

O projeto compara popularidade, média global, SVD e User-Based KNN usando o MovieLens 100k. Metadados são usados para descrição e apresentação dos títulos, sem entrar como preditores dos modelos.

Referência de sistemas de recomendação: Victor Coscrato, *Sistemas de Recomendação*, Laboratório de Estatística Aplicada — 2026/2. https://teaching.vcoscrato.com/sisrec/?view=print#/title-slide

Dados: https://grouplens.org/datasets/movielens/100k/

Download: https://files.grouplens.org/datasets/movielens/ml-100k.zip

O README original do **MovieLens 100k** informa que a redistribuição exige permissão separada. Por isso, os dados brutos não integram o repositório. São baixados automaticamente e preservados localmente com a documentação original. As previsões externas completas e históricos em CSV também permanecem locais quando contêm notas originais. Estatísticas agregadas, configurações e exemplos de recomendação são disponibilizados. Os demais outputs são reconstituídos pela execução.

Citação: F. Maxwell Harper e Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems, 5(4), artigo 19. https://doi.org/10.1145/2827872

Documentação da implementação:
- https://surprise.readthedocs.io/en/stable/matrix_factorization.html
- https://surprise.readthedocs.io/en/stable/knn_inspired.html
- https://surprise.readthedocs.io/en/stable/similarities.html

KNNBasic foi conferido também no código instalado da Surprise 1.1.5: seleção com `heapq.nlargest`, agregação apenas de pesos positivos, fallback por `PredictionImpossible` e média global. SVD usa SGD sobre notas observadas, com vieses e regularização.
