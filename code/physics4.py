import os, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_selection import mutual_info_classif
pd.set_option('display.width',230)
WIN=r"C:\Users\user\Desktop\VIP 2\rebuilt_cache"
CITIES=['jeddah','kaust','kz','mekkah']
d=[]
for c in CITIES:
    t=pd.read_csv(os.path.join(WIN,c+"_60s.csv"),engine='python'); t['city']=c; d.append(t)
D=pd.concat(d,ignore_index=True)
D['label']=D['label'].astype(str); D['city']=D['city'].astype(str)
FAM={'walk':'pedestrian','jog':'pedestrian','bike':'micro','scooter':'micro',
     'car':'vehicular','bus':'vehicular','motorcycle':'vehicular','train':'rail'}
D['family']=D.label.map(FAM).astype(str)
FEAT=[c for c in D.columns if c not in ('label','city','family')]
GPS=['longitude_mean','latitude_mean']; KIN=['velocity_mean','velocity_max','velocity_var']
RADIO=[c for c in FEAT if c not in GPS+KIN]
yl=np.asarray(D.label.tolist()); yf=np.asarray(D.family.tolist()); cty=np.asarray(D.city.tolist())
LB=sorted(set(yl.tolist()))

print("=== E. LABEL INFORMATION IN MISSINGNESS ALONE (no measured values at all) ===")
M=pd.DataFrame({('miss_'+c):(D[c].isna()|(D[c]==0)).astype(int) for c in FEAT})
M=M.loc[:,M.nunique()>1]
Xm=M.to_numpy(dtype=np.int8)
a,b,c1,d1=train_test_split(Xm,yl,test_size=.2,random_state=42,stratify=yl)
rf=RandomForestClassifier(n_estimators=200,random_state=0,n_jobs=2).fit(a,c1); p=rf.predict(b)
print("8-class from MISSINGNESS PATTERN ONLY: acc %.3f  macroF1 %.3f"%(accuracy_score(d1,p),f1_score(d1,p,average='macro',zero_division=0)))
for cl,v in zip(LB,f1_score(d1,p,average=None,labels=LB,zero_division=0)): print("   %-11s %.3f"%(cl,v))
print("\n   same, but WITHIN KAUST only (one city, so geography cannot help):")
K=D.city=='kaust'
Xk=M[K.values].to_numpy(dtype=np.int8); yk=yl[K.values]
a,b,c1,d1=train_test_split(Xk,yk,test_size=.2,random_state=42,stratify=yk)
rf=RandomForestClassifier(n_estimators=200,random_state=0,n_jobs=2).fit(a,c1); p=rf.predict(b)
print("   acc %.3f  macroF1 %.3f"%(accuracy_score(d1,p),f1_score(d1,p,average='macro',zero_division=0)))

print("\n=== F. MUTUAL INFORMATION by group ===")
X=D[FEAT].fillna(0).to_numpy(dtype=np.float64)
S=pd.Series(mutual_info_classif(X,yl,random_state=0),index=FEAT).sort_values(ascending=False)
print(S.head(12).round(3).to_string())
for nm,grp in [('GPS',GPS),('KINEMATIC',KIN),('RADIO',RADIO)]:
    print("  %-10s sumMI %.3f  share %.1f%%"%(nm,S[grp].sum(),100*S[grp].sum()/S.sum()))

LEAK=['timing_advance_mean','pathloss_mean','pathloss_max','pathloss_var',
      'rsrq_mean','rsrq_var','rsrq_max','rsrq_min','pdsch_mean']
SETS={'A all features':FEAT,'B no GPS':KIN+RADIO,'C radio only':RADIO,
      'D radio minus campaign-leak':[c for c in RADIO if c not in LEAK]}
print("\n=== G. DISCLOSURE LADDER: random split vs LEAVE-ONE-CITY-OUT ===")
for tgt,y in [('label',yl),('family',yf)]:
    print("\n--- target: %s ---"%tgt)
    for nm,fs in SETS.items():
        Xa=D[fs].fillna(0).to_numpy(dtype=np.float64)
        a,b,c1,d1=train_test_split(Xa,y,test_size=.2,random_state=42,stratify=y)
        rnd=f1_score(d1,RandomForestClassifier(n_estimators=200,random_state=0,n_jobs=2).fit(a,c1).predict(b),average='macro',zero_division=0)
        lo=[]
        for ho in CITIES:
            tr=cty!=ho; te=cty==ho
            rf2=RandomForestClassifier(n_estimators=200,random_state=0,n_jobs=2).fit(Xa[tr],y[tr])
            lo.append(f1_score(y[te],rf2.predict(Xa[te]),average='macro',zero_division=0))
        print("  %-28s random %.3f | LOCO mean %.3f  [%s]"%(nm,rnd,np.mean(lo)," ".join("%s %.2f"%(c[:4],v) for c,v in zip(CITIES,lo))))

print("\n=== H. THE PHYSICAL ESTIMATOR:  v_hat = 3.6 * handover_rate * D_city   (D = cell diameter, m) ===")
E=D[(D.velocity_mean>0)&(D.handover_rate>0)].copy()
print("windows with both velocity>0 and handover_rate>0: %d of %d (%.1f%%)"%(len(E),len(D),100*len(E)/len(D)))
rows=[]
for c in CITIES:
    s=E[(E.city==c)&(E.family=='vehicular')]
    if len(s)<10: rows.append([c,len(s),np.nan]); continue
    Dm=np.median(s.velocity_mean/(3.6*s.handover_rate))
    rows.append([c,len(s),round(Dm,0)])
CD=dict((r[0],r[2]) for r in rows)
print(pd.DataFrame(rows,columns=['city','vehicular_windows','implied_cell_diameter_m']).to_string(index=False))
E['v_hat']=3.6*E.handover_rate*E.city.map(CD)
print("\ncorrelation of v_hat with true velocity_mean:")
for c in CITIES:
    s=E[E.city==c]
    print("  %-8s n=%4d  pearson r=%.3f  spearman=%.3f"%(c,len(s),s.v_hat.corr(s.velocity_mean),s.v_hat.corr(s.velocity_mean,method='spearman')))
s=E; print("  %-8s n=%4d  pearson r=%.3f  spearman=%.3f"%('POOLED',len(s),s.v_hat.corr(s.velocity_mean),s.v_hat.corr(s.velocity_mean,method='spearman')))
print("\nmedian v_hat vs median true velocity, per mode:")
print(E.groupby('label').agg(n=('v_hat','size'),v_true=('velocity_mean','median'),v_hat=('v_hat','median')).round(1).to_string())
print("\n=== I. FAMILY from v_hat ALONE via kinematic thresholds (ped<15, micro<40, veh<160, else rail) ===")
def fam_of(v):
    return 'pedestrian' if v<15 else ('micro' if v<40 else ('vehicular' if v<160 else 'rail'))
E['fam_hat']=E.v_hat.map(fam_of); E['fam_true_v']=E.velocity_mean.map(fam_of)
yt=np.asarray(E.family.tolist())
print("  from v_hat (radio only):      macroF1 %.3f  acc %.3f"%(
    f1_score(yt,np.asarray(E.fam_hat.tolist()),average='macro',zero_division=0),accuracy_score(yt,np.asarray(E.fam_hat.tolist()))))
print("  from TRUE velocity (oracle):  macroF1 %.3f  acc %.3f"%(
    f1_score(yt,np.asarray(E.fam_true_v.tolist()),average='macro',zero_division=0),accuracy_score(yt,np.asarray(E.fam_true_v.tolist()))))
