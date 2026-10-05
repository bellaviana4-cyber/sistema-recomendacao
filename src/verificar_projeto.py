"""Verificações de integridade e do protocolo, sem selecionar modelos."""
from projeto import *
import nbformat
r,u,i=load_data()
print(manual_checks())
for fold,ti,vi in folds(r):
    inner=np.load(ROOT/f'data/splits/inner_{fold}.npz')
    assert set(inner['train'])|set(inner['validation'])==set(ti)
    assert not set(inner['train'])&set(inner['validation'])
    assert not (set(inner['train'])|set(inner['validation']))&set(vi)
# Previsões em lote, inclusive pesos empatados e insuficiência de suporte.
small=pd.DataFrame([(1,1,5),(1,2,3),(2,1,5),(2,3,4),(3,1,5),(3,2,2),(3,3,1),(4,2,1),(4,3,5)],columns=['user_id','item_id','rating'])
for kind in ['SVD','KNN']:
    for config in CONFIGS[kind]:
        a=model(kind,config).fit(trainset(small)); us,it,scores,nk=score_matrix(a)
        for j,user in enumerate(us):
            for k,item in enumerate(it): assert np.isclose(scores[j,k],a.predict(int(user),int(item)).est,atol=1e-10)
summary=comparison()
for kind in ['SVD','KNN','Baselines']:
    p=pd.read_csv(TABLES/f'{kind}_predictions.csv')
    m=pd.read_csv(TABLES/f'{kind}_metrics.csv')
    for (fold,label,variant),g in p.groupby(['fold','model','variant']):
        ids=np.load(ROOT/f'data/splits/fold_{fold}.npz')['test']
        assert len(g)==len(ids)==20000
        assert set(map(tuple,g[['user_id','item_id']].values))==set(map(tuple,r.iloc[ids][['user_id','item_id']].values))
        mean=r.iloc[np.load(ROOT/f'data/splits/fold_{fold}.npz')['train']].rating.mean()
        if kind=='Baselines': assert np.allclose(g.prediction,mean)
        actual=m[(m.fold==fold)&(m.model==label)&(m.variant==variant)].iloc[0].rmse
        assert np.isclose(rmse(g.rating,g.prediction),actual)
    if kind!='Baselines':
        ad=pd.read_csv(TABLES/f'{kind}_eligibility.csv'); top=pd.read_csv(TABLES/f'{kind}_top10.csv')
    else:
        ad=pd.read_csv(TABLES/'Baselines_eligibility.csv'); top=pd.read_csv(TABLES/'Baselines_top10.csv')
    for (fold,variant),g in top.groupby(['fold','variant']):
        train=r.iloc[np.load(ROOT/f'data/splits/fold_{fold}.npz')['train']]
        test=r.iloc[np.load(ROOT/f'data/splits/fold_{fold}.npz')['test']]
        assert g.groupby('user_id').size().eq(10).all()
        assert not g.duplicated(['user_id','item_id']).any()
        seen=set(map(tuple,train[['user_id','item_id']].values))
        assert not seen & set(map(tuple,g[['user_id','item_id']].values))
        assert set(g.item_id)<=set(train.item_id)
        relevant=set(map(tuple,test[test.rating>=4][['user_id','item_id']].values))
        hits=np.array([int((user,item) in relevant) for user,item in g[['user_id','item_id']].itertuples(index=False,name=None)])
        assert np.array_equal(hits,g.hit)
        precision_recomputed=g.groupby('user_id').hit.sum()/10
        a=ad[(ad.fold==fold)&(ad.variant==variant)&(ad.reason=='incluido')].set_index('user_id')
        assert np.allclose(precision_recomputed.sort_index(),a.precision10.sort_index())
        assert set(precision_recomputed.index)==set(a.index)
for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
    nb=nbformat.read(path,4); nbformat.validate(nb)
    cells=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in cells)
    assert not [o for c in cells for o in c.outputs if o.output_type=='error']
report={'integridade':'aprovada','folds_disjuntos':'aprovados','validacao_interna_isolada':'aprovada','usuarios_elegiveis_identicos':'aprovados','top10_sem_duplicatas_ou_historico':'aprovado','metricas_recalculadas':'aprovadas','lote_equivalente_surprise':'aprovado','notebooks_executados_sem_erros':'aprovados','kernel_delivery':'IPython InProcessKernelManager, processo novo por notebook'}
(ROOT/'outputs/verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(summary.to_string(index=False));print(report)
