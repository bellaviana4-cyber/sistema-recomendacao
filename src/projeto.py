"""Dados, divisões e avaliação compartilhados. Metadados nunca entram nos modelos."""
from pathlib import Path
import json, hashlib, time, urllib.request, zipfile, importlib.metadata
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from surprise import Dataset, Reader, SVD, KNNBasic
ROOT=Path(__file__).resolve().parents[1]
SEED=2026
GENRES=['unknown','Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
TABLES=ROOT/'outputs/tables'; FIGURES=ROOT/'outputs/figures'
for p in [TABLES,FIGURES,ROOT/'data/splits']: p.mkdir(parents=True,exist_ok=True)
def save(df,name):
    df.to_csv(TABLES/f'{name}.csv',index=False)
    return df

def load_data():
    folder=ROOT/'data/raw/ml-100k'
    if not (folder/'u.data').exists():
        folder.parent.mkdir(parents=True,exist_ok=True)
        archive=folder.parent/'ml-100k.zip'
        urllib.request.urlretrieve('https://files.grouplens.org/datasets/movielens/ml-100k.zip',archive)
        with zipfile.ZipFile(archive) as z:
            for name in ['u.data','u.user','u.item','README','u.genre','u.occupation','u.info']:
                z.extract('ml-100k/'+name,folder.parent)
    ratings=pd.read_csv(folder/'u.data',sep='\t',names=['user_id','item_id','rating','timestamp'])
    users=pd.read_csv(folder/'u.user',sep='|',names=['user_id','age','gender','occupation','zip_code'],dtype={'zip_code':str})
    items=pd.read_csv(folder/'u.item',sep='|',encoding='latin-1',names=['item_id','title','release_date','video_release_date','imdb_url']+GENRES)
    items['year']=pd.to_datetime(items.release_date,format='%d-%b-%Y',errors='coerce').dt.year
    assert len(ratings)==100000 and len(users)==943 and len(items)==1682
    assert ratings.rating.between(1,5).all() and ratings.isna().sum().sum()==0
    assert not ratings.duplicated(['user_id','item_id']).any()
    assert users.user_id.is_unique and items.item_id.is_unique
    assert set(ratings.user_id)==set(users.user_id) and set(ratings.item_id)==set(items.item_id)
    merged=ratings.merge(users,on='user_id',validate='many_to_one').merge(items,on='item_id',validate='many_to_one')
    assert len(merged)==len(ratings)
    return ratings,users,items

def folds(ratings):
    assignment=np.empty(len(ratings),dtype=int)
    result=[]
    for fold,(train,test) in enumerate(KFold(5,shuffle=True,random_state=SEED).split(ratings),1):
        assert not set(train)&set(test)
        assert not set(map(tuple,ratings.iloc[train][['user_id','item_id']].values))&set(map(tuple,ratings.iloc[test][['user_id','item_id']].values))
        assignment[test]=fold
        np.savez_compressed(ROOT/f'data/splits/fold_{fold}.npz',train=train,test=test)
        result.append((fold,train,test))
    save(pd.DataFrame({'row_id':ratings.index,'test_fold':assignment}),'fold_assignment')
    digest=hashlib.sha256((ROOT/'data/raw/ml-100k/u.data').read_bytes()).hexdigest()
    (ROOT/'data/splits/manifest.json').write_text(json.dumps({'seed':SEED,'sha256_u_data':digest,'folds':5,'inner_test_size':0.2},indent=2))
    return result

def trainset(df):
    return Dataset.load_from_df(df[['user_id','item_id','rating']],Reader(rating_scale=(1,5))).build_full_trainset()

CONFIGS={
 'SVD':[dict(n_factors=50,n_epochs=20,lr_all=.005,reg_all=.02),dict(n_factors=50,n_epochs=30,lr_all=.005,reg_all=.08),dict(n_factors=100,n_epochs=30,lr_all=.005,reg_all=.08)],
 'KNN':[dict(k=40,min_k=1,min_support=1),dict(k=20,min_k=3,min_support=3),dict(k=60,min_k=3,min_support=3)]}
def model(kind,config):
    if kind=='SVD': return SVD(**config,random_state=SEED)
    c=config.copy(); support=c.pop('min_support')
    return KNNBasic(**c,sim_options={'name':'cosine','user_based':True,'min_support':support},verbose=False)

def rmse(actual,predicted): return float(np.sqrt(np.mean((np.asarray(actual)-np.asarray(predicted))**2)))
def precision(top,relevant): return len(set(top)&set(relevant))/10

def score_matrix(algo):
    """Lote exato das previsões em usuários/itens conhecidos, conferido contra predict."""
    t=algo.trainset
    us=np.array([t.to_raw_uid(u) for u in t.all_users()]); it=np.array([t.to_raw_iid(i) for i in t.all_items()])
    if isinstance(algo,SVD):
        scores=t.global_mean+algo.bu[:,None]+algo.bi[None,:]+algo.pu@algo.qi.T
        nk=np.zeros(scores.shape,dtype=int)
    else:
        scores=np.full((t.n_users,t.n_items),t.global_mean); nk=np.zeros(scores.shape,dtype=int)
        for i in t.all_items():
            neighbors=np.array([u for u,r in t.ir[i]]); ratings=np.array([r for u,r in t.ir[i]])
            weights=algo.sim[:,neighbors]
            # Ordenação estável preserva empates como heapq.nlargest da Surprise.
            order=np.argsort(-weights,axis=1,kind='stable')[:,:algo.k]
            selected=np.take_along_axis(weights,order,axis=1)
            positive=selected>0; selected=np.where(positive,selected,0)
            counts=positive.sum(axis=1); denom=selected.sum(axis=1)
            numer=(selected*ratings[order]).sum(axis=1)
            ok=counts>=algo.min_k
            scores[ok,i]=numer[ok]/denom[ok]; nk[:,i]=counts
    scores=np.clip(scores,1,5)
    rng=np.random.default_rng(SEED)
    for _ in range(200):
        u=int(rng.integers(len(us))); i=int(rng.integers(len(it)))
        assert np.isclose(scores[u,i],algo.predict(int(us[u]),int(it[i])).est,atol=1e-10)
    return us,it,scores,nk

def ranking(algo,train,test,kind,fold,variant,popularity_counts=None):
    if kind=='Popularidade':
        us=np.sort(train.user_id.unique()); counts=train.item_id.value_counts() if popularity_counts is None else popularity_counts
        it=np.sort(counts.index.values); scores=np.broadcast_to(counts.reindex(it).values,(len(us),len(it)))
        nk=None
    else: us,it,scores,nk=score_matrix(algo)
    histories=train.groupby('user_id').item_id.apply(set).to_dict()
    relevant=test[test.rating>=4].groupby('user_id').item_id.apply(set).to_dict()
    lookup={int(u):j for j,u in enumerate(us)}; universe=set(it)
    audit=[]; tops=[]; candidates_fallback=0; candidates_total=0
    for u in sorted(set(train.user_id)|set(test.user_id)):
        seen=histories.get(u,set()); candidates=universe-seen; rel=relevant.get(u,set())&candidates
        reason='sem_historico' if u not in lookup else 'menos_10_candidatos' if len(candidates)<10 else 'sem_relevante_elegivel' if not rel else 'incluido'
        row=dict(fold=fold,model=kind,variant=variant,user_id=u,n_history=len(seen),n_candidates=len(candidates),n_relevant=len(rel),reason=reason,precision10=np.nan)
        if reason=='incluido':
            j=lookup[u]; mask=np.array([i not in seen for i in it]); inds=np.flatnonzero(mask)
            order=np.lexsort((it[inds],-scores[j,inds]))[:10]; chosen=inds[order]
            top=it[chosen]; assert len(top)==10 and len(set(top))==10 and not set(top)&seen
            row['precision10']=precision(top,rel)
            for rank,c in enumerate(chosen,1): tops.append(dict(fold=fold,model=kind,variant=variant,user_id=u,rank=rank,item_id=int(it[c]),score=float(scores[j,c]),hit=int(it[c] in rel)))
            if kind=='KNN': candidates_fallback+=int((nk[j,inds]<algo.min_k).sum()); candidates_total+=len(inds)
        audit.append(row)
    ad=pd.DataFrame(audit)
    return ad,pd.DataFrame(tops),candidates_fallback/max(candidates_total,1)

def evaluate(kind,ratings):
    metrics=[]; preds=[]; audits=[]; tops=[]; tuning=[]
    for fold,ti,vi in folds(ratings):
        train=ratings.iloc[ti]; test=ratings.iloc[vi]
        print(kind,'fold',fold,flush=True)
        missing_users=~test.user_id.isin(train.user_id); missing_items=~test.item_id.isin(train.item_id)
        if kind=='Baselines': variants=[('principal','Média global',None),('principal','Popularidade',None)]
        else:
            inner_t,inner_v=train_test_split(ti,test_size=.2,random_state=SEED+fold)
            np.savez_compressed(ROOT/f'data/splits/inner_{fold}.npz',train=inner_t,validation=inner_v)
            inner=trainset(ratings.iloc[inner_t]); values=[]
            for idx,c in enumerate(CONFIGS[kind]):
                start=time.perf_counter(); a=model(kind,c).fit(inner)
                pp=a.test(list(ratings.iloc[inner_v][['user_id','item_id','rating']].itertuples(index=False,name=None)))
                loss=rmse([p.r_ui for p in pp],[p.est for p in pp]); values.append(loss)
                tuning.append(dict(fold=fold,model=kind,configuration=idx,params=json.dumps(c),inner_rmse=loss,seconds=time.perf_counter()-start))
            best=int(np.argmin(values)); variants=[('inicial',kind,CONFIGS[kind][0]),('ajustado',kind,CONFIGS[kind][best])]
        tr=trainset(train)
        for variant,label,c in variants:
            start=time.perf_counter()
            a=None if c is None else model(kind,c).fit(tr)
            global_mean=train.rating.mean() if label=='Média global' else None
            popularity_counts=train.item_id.value_counts() if label=='Popularidade' else None
            fit=time.perf_counter()-start
            start=time.perf_counter(); loss=np.nan; p10=np.nan; fallback=np.nan; rankfallback=np.nan; n=0
            if label!='Popularidade':
                pp=None if a is None else a.test(list(test[['user_id','item_id','rating']].itertuples(index=False,name=None)))
                estimates=np.repeat(global_mean,len(test)) if a is None else np.array([p.est for p in pp])
                fallback=0 if pp is None else np.mean([p.details.get('was_impossible',False) for p in pp])
                loss=rmse(test.rating,estimates)
                pr=test.copy(); pr['prediction']=estimates; pr['fold']=fold; pr['model']=label; pr['variant']=variant
                pr['n_history']=pr.user_id.map(train.user_id.value_counts()).fillna(0).astype(int)
                pr['unknown_user']=missing_users.values; pr['unknown_item']=missing_items.values
                pr['fallback']=False if pp is None else [p.details.get('was_impossible',False) for p in pp]
                pr['actual_k']=np.nan if pp is None else [p.details.get('actual_k',np.nan) for p in pp]
                preds.append(pr)
            if label!='Média global':
                ad,tp,rankfallback=ranking(a,train,test,label,fold,variant,popularity_counts); audits.append(ad); tops.append(tp)
                p10=ad.precision10.mean(); n=int(ad.precision10.notna().sum())
            metrics.append(dict(fold=fold,model=label,variant=variant,rmse=loss,precision10=p10,n_users=n,fit_seconds=fit,evaluation_seconds=time.perf_counter()-start,fallback_test=fallback,fallback_candidates=rankfallback,unknown_users=int(missing_users.sum()),unknown_items=int(missing_items.sum()),params=json.dumps(c)))
        save(pd.DataFrame(metrics),kind+'_metrics')
    for data,suffix in [(preds,'predictions'),(audits,'eligibility'),(tops,'top10')]:
        if data: save(pd.concat(data,ignore_index=True),kind+'_'+suffix)
    if tuning: save(pd.DataFrame(tuning),kind+'_tuning')
    return pd.DataFrame(metrics)

def comparison():
    tables=[pd.read_csv(TABLES/f'{k}_metrics.csv') for k in ['Baselines','SVD','KNN']]
    allmetrics=pd.concat(tables,ignore_index=True)
    chosen=allmetrics[allmetrics.variant.isin(['principal','ajustado'])]
    summary=chosen.groupby('model').agg(rmse_mean=('rmse','mean'),rmse_sd=('rmse','std'),precision10_mean=('precision10','mean'),precision10_sd=('precision10','std'),n_users_min=('n_users','min'),n_users_max=('n_users','max'),fit_seconds_mean=('fit_seconds','mean'),evaluation_seconds_mean=('evaluation_seconds','mean')).reset_index()
    audits=[pd.read_csv(TABLES/f'{k}_eligibility.csv') for k in ['Baselines','SVD','KNN']]
    audit=pd.concat(audits); audit=audit[audit.variant.isin(['principal','ajustado'])]
    for fold,g in audit.groupby('fold'):
        included=[set(x.loc[x.reason=='incluido','user_id']) for _,x in g.groupby('model')]
        assert included[0]==included[1]==included[2]
    save(chosen,'comparison_folds'); return save(summary,'comparison_summary')

def manual_checks():
    assert np.isclose(rmse([1,5],[2,3]),np.sqrt(2.5))
    assert precision(list(range(1,11)),{2,5,20})==.2
    assert precision(list(range(1,11)),set())==0
    # Cold start conforme as duas implementações.
    small=pd.DataFrame({'user_id':[1,1,2,2],'item_id':[1,2,1,2],'rating':[3,5,2,4]})
    for kind in ['SVD','KNN']:
        a=model(kind,CONFIGS[kind][0]).fit(trainset(small))
        assert np.isfinite(a.predict(999,999).est)
        if kind=='KNN': assert a.predict(999,999).details['was_impossible']
    return 'RMSE manual = √2,5; Precisão@10 manual = 2/10; cold start conferido.'

def versions():
    import platform
    d={'python':platform.python_version()}
    for p in ['numpy','pandas','matplotlib','scikit-learn','scikit-surprise','nbformat','nbclient','ipykernel']: d[p]=importlib.metadata.version(p)
    (ROOT/'outputs/versions.json').write_text(json.dumps(d,indent=2)); return d

def examples(kind,ratings,items):
    split=np.load(ROOT/'data/splits/fold_1.npz'); train=ratings.iloc[split['train']]
    counts=train.user_id.value_counts().sort_index(); chosen=[]
    for label,q in [('baixa',.1),('intermediária',.5),('alta',.9)]:
        target=counts.quantile(q); u=int((counts-target).abs().sort_values(kind='stable').index[0]); chosen.append((label,u))
    tuning=pd.read_csv(TABLES/f'{kind}_tuning.csv'); best=tuning[tuning.fold==1].sort_values(['inner_rmse','configuration']).iloc[0]
    algo=model(kind,json.loads(best.params)).fit(trainset(train))
    us,it,scores,nk=score_matrix(algo); rows=[]; history=[]
    for label,u in chosen:
        j=int(np.where(us==u)[0][0]); seen=set(train.loc[train.user_id==u,'item_id']); inds=np.array([i for i in range(len(it)) if it[i] not in seen])
        top=inds[np.lexsort((it[inds],-scores[j,inds]))[:10]]
        for rank,c in enumerate(top,1): rows.append(dict(activity=label,user_id=u,n_history=len(seen),rank=rank,item_id=int(it[c]),score=float(scores[j,c])))
        h=train[train.user_id==u].copy(); h['activity']=label; history.append(h)
    rec=pd.DataFrame(rows).merge(items[['item_id','title']],on='item_id',validate='many_to_one')
    hist=pd.concat(history).merge(items[['item_id','title']],on='item_id',validate='many_to_one')
    save(hist,kind+'_example_history'); return save(rec,kind+'_examples'),hist
