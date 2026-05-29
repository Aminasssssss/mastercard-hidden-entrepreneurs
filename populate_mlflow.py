# -*- coding: utf-8 -*-
"""Populate mlruns/ with experiments comparing all methods.
Run once; afterwards `mlflow ui` shows the comparison."""
import os, sys
os.chdir('/home/claude/work/mdq_app')

import mlflow
import pandas as pd, numpy as np, warnings, gc
warnings.filterwarnings('ignore')
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from scipy.stats import rankdata

SEED=42; np.random.seed(SEED)
mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")
mlflow.set_experiment("hidden_entrepreneur_detection")

# load already-prepared scored cards to get features fast
df = pd.read_parquet('data/scored_cards.parquet')

# We need to actually run methods to log them. Reload full data quickly.
USECOLS = ['card_number','merchant_id','mcc','transaction_amount_kzt',
           'transaction_timestamp','transaction_date','channel','country',
           'tokenized','is_recurring']
print("loading raw data...")
biz = pd.read_parquet('/home/claude/work/business_cards_MDQ.parquet', columns=USECOLS)
con = pd.read_parquet('/home/claude/work/consumer_cards_MDQ.parquet', columns=USECOLS)
for d in (biz,con):
    d['transaction_timestamp']=pd.to_datetime(d['transaction_timestamp'])
    d['transaction_date']=pd.to_datetime(d['transaction_date'])

biz['_isb']=1; con['_isb']=0
pair = pd.concat([biz[['merchant_id','card_number','_isb']], con[['merchant_id','card_number','_isb']]]).drop_duplicates()
ms = pair.groupby('merchant_id').agg(n_cards=('card_number','nunique'), n_biz=('_isb','sum'))
ms['biz_intensity'] = ms['n_biz']/ms['n_cards']
MINT = ms['biz_intensity'].to_dict()
biz.drop(columns='_isb',inplace=True); con.drop(columns='_isb',inplace=True)
del pair; gc.collect()

EXP_B2B = {
    'business_serv':['7399','7389'],'computers':['5045','5044'],'stationery':['5943','5111'],
    'computer_repair':['7379'],'accounting':['8931'],'telecom':['4812','4816'],
    'office_furniture':['5021'],'professional_serv':['8911','8111'],'consulting':['7392'],
    'employment':['7361'],'software':['7372'],'subscriptions':['5968'],
    'logistics':['4214','4215'],'wholesale':['5199','5065'],'advertising':['7311']}
ALL_B2B=sum(EXP_B2B.values(),[])

def build_features(df):
    df=df.copy(); df['mcc']=df.mcc.astype(str)
    ts=df.transaction_timestamp
    df['hour']=ts.dt.hour; df['weekday']=ts.dt.weekday
    df['is_biz_hour']=((df.hour.between(9,18))&(df.weekday<5)).astype('int8')
    df['is_weekday']=(df.weekday<5).astype('int8'); df['is_weekend']=(df.weekday>=5).astype('int8')
    df['is_night']=(df.hour.between(0,6)).astype('int8'); df['is_kz']=(df.country=='Kazakhstan').astype('int8')
    df['is_online']=(df.channel=='online').astype('int8'); df['is_b2b']=df.mcc.isin(ALL_B2B).astype('int8')
    for g,codes in EXP_B2B.items(): df[f'{g}_ratio']=df.mcc.isin(codes).astype('int8')
    df['is_large']=(df.transaction_amount_kzt>=200_000).astype('int8')
    df['is_round']=(df.transaction_amount_kzt%1000==0).astype('int8')
    df['mint']=df.merchant_id.map(MINT).fillna(0.0)
    df['is_bizdom_merch']=(df.mint>0.5).astype('int8')
    grp=df.groupby('card_number',sort=False)
    agg={'txn_count':('transaction_amount_kzt','count'),'total_spend':('transaction_amount_kzt','sum'),
        'avg_amount':('transaction_amount_kzt','mean'),'median_amount':('transaction_amount_kzt','median'),
        'std_amount':('transaction_amount_kzt','std'),'max_amount':('transaction_amount_kzt','max'),
        'min_amount':('transaction_amount_kzt','min'),'b2b_ratio':('is_b2b','mean'),
        'online_ratio':('is_online','mean'),'tokenized_ratio':('tokenized','mean'),
        'recurring_ratio':('is_recurring','mean'),'biz_hour_ratio':('is_biz_hour','mean'),
        'weekday_ratio':('is_weekday','mean'),'unique_merchants':('merchant_id','nunique'),
        'unique_mcc':('mcc','nunique'),'kz_ratio':('is_kz','mean'),
        'large_amount_ratio':('is_large','mean'),
        'graph_merch_biz_affinity':('mint','mean'),'graph_max_merch_biz':('mint','max')}
    for g in EXP_B2B: agg[f'{g}_ratio']=(f'{g}_ratio','mean')
    return grp.agg(**agg).reset_index()

print("features...")
biz_feats = build_features(biz); con_feats = build_features(con)
del biz, con; gc.collect()
NUMERIC = [c for c in biz_feats.columns if c != 'card_number']
LOGF=['total_spend','avg_amount','median_amount','std_amount','max_amount','min_amount']
bl,cl = biz_feats.copy(), con_feats.copy()
for c in LOGF:
    bl[c]=np.log1p(bl[c].clip(lower=0)); cl[c]=np.log1p(cl[c].clip(lower=0))
X_biz = bl[NUMERIC].fillna(0).values.astype(float)
X_con = cl[NUMERIC].fillna(0).values.astype(float)
scl = RobustScaler().fit(X_biz)
Xb, Xc = scl.transform(X_biz), scl.transform(X_con)
y_all = np.concatenate([np.ones(len(Xb)), np.zeros(len(Xc))])

def evaluate(name, params, biz_score, con_score):
    """Log experiment to mlflow."""
    pseudo_auc = roc_auc_score(y_all, np.concatenate([biz_score, con_score]))
    top100_b2b = con_feats.iloc[np.argsort(-con_score)[:100]]['b2b_ratio'].mean()
    rest_b2b = con_feats.iloc[np.argsort(-con_score)[100:]]['b2b_ratio'].mean()
    lift = top100_b2b / max(rest_b2b, 1e-9)
    with mlflow.start_run(run_name=name):
        for k,v in params.items(): mlflow.log_param(k, v)
        mlflow.log_metric("pseudo_auc", float(pseudo_auc))
        mlflow.log_metric("top100_b2b_ratio", float(top100_b2b))
        mlflow.log_metric("top100_lift_vs_rest", float(lift))
        mlflow.log_metric("n_features", len(NUMERIC))
        mlflow.set_tag("method_family", params.get("family","one_class"))
        mlflow.set_tag("status", "trained")
    print(f"  {name}: pseudo_auc={pseudo_auc:.4f} lift={lift:.1f}")

print("running experiments...")

# 1. Naive rule
print("1/10 naive rule")
bs = biz_feats.b2b_ratio.values; cs = con_feats.b2b_ratio.values
evaluate("naive_rule_b2b_ratio", {"family":"baseline","feature":"b2b_ratio"}, bs, cs)

# 2. Isolation Forest
print("2/10 isolation forest")
m = IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xb)
evaluate("isolation_forest", {"family":"one_class","n_estimators":300}, m.decision_function(Xb), m.decision_function(Xc))

# 3. One-Class SVM
print("3/10 one-class svm")
sub = np.random.RandomState(SEED).choice(len(Xb), 5000, replace=False)
m = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1).fit(Xb[sub])
evaluate("one_class_svm", {"family":"one_class","kernel":"rbf","nu":0.1,"n_train":5000}, m.decision_function(Xb), m.decision_function(Xc))

# 4. GMM
print("4/10 gmm")
m = GaussianMixture(n_components=5, covariance_type='full', random_state=SEED, max_iter=200).fit(Xb)
evaluate("gaussian_mixture", {"family":"one_class","n_components":5}, m.score_samples(Xb), m.score_samples(Xc))

# 5. Mahalanobis
print("5/10 mahalanobis")
mu = Xb.mean(0); cov = np.cov(Xb.T) + np.eye(Xb.shape[1])*1e-4; inv = np.linalg.pinv(cov)
def mah(X):
    d = X-mu; return -np.sqrt(np.einsum('ij,jk,ik->i', d, inv, d))
evaluate("mahalanobis", {"family":"one_class","regularization":1e-4}, mah(Xb), mah(Xc))

# 6. Autoencoder
print("6/10 autoencoder")
ae = MLPRegressor(hidden_layer_sizes=(16,8,16), max_iter=60, random_state=SEED); ae.fit(Xb, Xb)
def ae_score(X): return -((ae.predict(X)-X)**2).mean(1)
evaluate("autoencoder_mlp", {"family":"one_class","layers":"16-8-16","max_iter":60}, ae_score(Xb), ae_score(Xc))

# 7. PU Learning
print("7/10 pu learning")
Ptr, Phd = train_test_split(X_biz, test_size=0.2, random_state=SEED)
Ptr, Phd = scl.transform(Ptr), scl.transform(Phd)
Xpu = np.vstack([Ptr, Xc]); spu = np.concatenate([np.ones(len(Ptr)), np.zeros(len(Xc))])
idx = np.random.RandomState(SEED).permutation(len(Xpu))
mpu = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                          class_weight='balanced', random_state=SEED, verbose=-1).fit(Xpu[idx], spu[idx])
c_hat = mpu.predict_proba(Phd)[:,1].mean()
pu_b = mpu.predict_proba(scl.transform(X_biz))[:,1]
pu_c = np.clip(mpu.predict_proba(Xc)[:,1]/c_hat, 0, 1)
evaluate("pu_learning", {"family":"semi_supervised","c_hat":round(c_hat,4),"estimator":"lightgbm"}, pu_b, pu_c)

# 8-10. binary classifiers (для comparison)
print("8/10 logistic regression")
Xall = np.vstack([Xb,Xc])
m = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED).fit(Xall, y_all)
evaluate("binary_logreg", {"family":"binary","class_weight":"balanced"}, m.predict_proba(Xb)[:,1], m.predict_proba(Xc)[:,1])

print("9/10 random forest binary")
m = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', n_jobs=-1, random_state=SEED).fit(Xall, y_all)
evaluate("binary_random_forest", {"family":"binary","n_estimators":200,"max_depth":10}, m.predict_proba(Xb)[:,1], m.predict_proba(Xc)[:,1])

print("10/10 lightgbm binary")
m = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, class_weight='balanced', random_state=SEED, verbose=-1).fit(Xall, y_all)
evaluate("binary_lightgbm", {"family":"binary","n_estimators":300}, m.predict_proba(Xb)[:,1], m.predict_proba(Xc)[:,1])

# 11 — финальный ансамбль
print("11/11 ensemble")
def rk(b,c):
    a=np.concatenate([b,c]); r=rankdata(a)/len(a); return r[:len(b)], r[len(b):]
methods_b = [IsolationForest(n_estimators=300,random_state=SEED,n_jobs=-1).fit(Xb).decision_function(Xb)]
# уже посчитанные — берём напрямую:
br=[]; cr=[]
br_, cr_ = rk(IsolationForest(n_estimators=300,random_state=SEED,n_jobs=-1).fit(Xb).decision_function(Xb),
              IsolationForest(n_estimators=300,random_state=SEED,n_jobs=-1).fit(Xb).decision_function(Xc))
br.append(br_); cr.append(cr_)
# remaining: reuse models above — для простоты пересчитаем рангировано в комплексе
# чтобы было быстро — используем уже посчитанные scores
br_full = np.column_stack([
    rk(IsolationForest(n_estimators=200,random_state=SEED,n_jobs=-1).fit(Xb).decision_function(Xb),
       IsolationForest(n_estimators=200,random_state=SEED,n_jobs=-1).fit(Xb).decision_function(Xc))[0]
    for _ in [1]
])
# proper ensemble: just use what we logged above isn't straightforward — recompute
# To keep code simple and the run fast, log the ensemble as separate run using existing per-method outputs
# Re-run methods quickly to collect both biz and con scores together
m_if = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(Xb)
s_if_b, s_if_c = m_if.decision_function(Xb), m_if.decision_function(Xc)
s_oc_b = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1).fit(Xb[sub]).decision_function(Xb)
s_oc_c = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1).fit(Xb[sub]).decision_function(Xc)
gm = GaussianMixture(5, covariance_type='full', random_state=SEED, max_iter=200).fit(Xb)
s_gm_b, s_gm_c = gm.score_samples(Xb), gm.score_samples(Xc)
s_md_b, s_md_c = mah(Xb), mah(Xc)
ae2 = MLPRegressor(hidden_layer_sizes=(16,8,16), max_iter=60, random_state=SEED); ae2.fit(Xb,Xb)
s_ae_b, s_ae_c = ae_score(Xb), ae_score(Xc)

R_b = np.column_stack([rk(s,t)[0] for s,t in [(s_if_b,s_if_c),(s_oc_b,s_oc_c),(s_gm_b,s_gm_c),(s_md_b,s_md_c),(s_ae_b,s_ae_c),(pu_b,pu_c)]])
R_c = np.column_stack([rk(s,t)[1] for s,t in [(s_if_b,s_if_c),(s_oc_b,s_oc_c),(s_gm_b,s_gm_c),(s_md_b,s_md_c),(s_ae_b,s_ae_c),(pu_b,pu_c)]])
ens_b, ens_c = R_b.mean(1), R_c.mean(1)
evaluate("ensemble_6_uniform_FINAL", {"family":"ensemble","n_methods":6,"weights":"uniform","strategy":"rank_average"}, ens_b, ens_c)

print("\nALL DONE. mlruns/ ready. To view:")
print("  cd /home/claude/work/mdq_app && mlflow ui --backend-store-uri file:./mlruns")
