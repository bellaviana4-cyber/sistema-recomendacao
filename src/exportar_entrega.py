"""Exporta previsões/listas derivadas, sem redistribuir notas/timestamps originais."""
from projeto import *
ratings,_,_=load_data()
keys=ratings[['user_id','item_id']].reset_index(names='row_id')
for kind in ['Baselines','SVD','KNN']:
    pred=pd.read_csv(TABLES/f'{kind}_predictions.csv')
    pred=pred.drop(columns=['rating','timestamp']).merge(keys,on=['user_id','item_id'],validate='many_to_one')
    pred.to_csv(TABLES/f'{kind}_external_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    top=pd.read_csv(TABLES/f'{kind}_top10.csv')
    top.to_csv(TABLES/f'{kind}_recommendations.csv.gz',index=False,compression={'method':'gzip','mtime':0})
print('Previsões externas e top 10 exportados sem notas originais.')
cold=[]; sizes=[]
for fold in range(1,6):
    indices=np.load(ROOT/f'data/splits/fold_{fold}.npz')
    train=ratings.iloc[indices['train']]; test=ratings.iloc[indices['test']]
    missing_users=test.loc[~test.user_id.isin(train.user_id),'user_id'].value_counts()
    missing_items=test.loc[~test.item_id.isin(train.item_id),'item_id'].value_counts()
    for column,missing in [('user_id',missing_users),('item_id',missing_items)]:
        for identifier,count in missing.items(): cold.append(dict(fold=fold,entity=column,identifier=identifier,n_test_ratings=count))
    sizes.append(dict(fold=fold,n_train=len(train),n_test=len(test),n_train_users=train.user_id.nunique(),n_train_items=train.item_id.nunique(),n_unknown_users=len(missing_users),n_unknown_items=len(missing_items),n_test_ratings_unknown_item=int(missing_items.sum())))
save(pd.DataFrame(cold,columns=['fold','entity','identifier','n_test_ratings']),'cold_start_ids')
save(pd.DataFrame(sizes),'fold_sizes')
