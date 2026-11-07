import os, numpy as np, pandas as pd
pd.set_option('display.width',220)
WIN=r"C:\Users\user\Desktop\VIP 2\rebuilt_cache"
CITIES=['jeddah','kaust','kz','mekkah']
d=[]
for c in CITIES:
    t=pd.read_csv(os.path.join(WIN,c+"_60s.csv")); t['city']=c; d.append(t)
D=pd.concat(d,ignore_index=True)
ORDER=['walk','jog','bike','scooter','bus','motorcycle','car','train']

print("=== A. IS pathloss/timing_advance A DEVICE FINGERPRINT? missingness within KAUST ONLY ===")
K=D[D.city=='kaust']
rows=[]
for m in ORDER:
    s=K[K.label==m]
    if len(s)==0: continue
    r=[m,len(s)]
    for col in ['timing_advance_mean','pathloss_mean','rsrq_mean','pdsch_mean','rank_mean','cqi1_mean']:
        r.append(round(100*(s[col].isna()|(s[col]==0)).mean(),1))
    rows.append(r)
print(pd.DataFrame(rows,columns=['mode','n_kaust','TA','pathloss','rsrq','pdsch','rank','cqi1']).to_string(index=False))

print("\n=== B. VELOCITY_MAX vs VELOCITY_MEAN quantiles (reconciling the ceilings) ===")
rows=[]
for m in ORDER:
    s=D[D.label==m]
    vx=s.velocity_max.dropna(); vx=vx[vx>0]
    vm=s.velocity_mean.dropna(); vm=vm[vm>0]
    rows.append([m,len(vx),round(vx.quantile(.95),1),round(vx.quantile(.99),1),round(vx.max(),1),
                 round(vm.quantile(.99),1),round(vm.max(),1)])
print(pd.DataFrame(rows,columns=['mode','n','vmax_p95','vmax_p99','vmax_MAX','vmean_p99','vmean_MAX']).to_string(index=False))

print("\n=== C. PHYSICALLY IMPLAUSIBLE WINDOWS (velocity_mean above generous ceiling) ===")
CEIL={'walk':10,'jog':20,'bike':40,'scooter':40,'bus':110,'motorcycle':140,'car':180,'train':350}
rows=[]; tot=0
for m in ORDER:
    s=D[(D.label==m)&D.velocity_mean.notna()&(D.velocity_mean>0)]
    bad=s[s.velocity_mean>CEIL[m]]; tot+=len(bad)
    rows.append([m,CEIL[m],len(s),len(bad),round(100*len(bad)/max(len(s),1),2),
                 round(bad.velocity_mean.max(),1) if len(bad) else np.nan])
print(pd.DataFrame(rows,columns=['mode','ceiling','n_obs','violating','pct','worst']).to_string(index=False))
print("total violating windows:",tot)

print("\n=== D. handover-per-distance: MEDIANS (means swamped by heavy tails) ===")
E=D[(D.velocity_mean.notna())&(D.velocity_mean>0)].copy()
E['hpd']=E.handover_rate/E.velocity_mean
FAM={'walk':'pedestrian','jog':'pedestrian','bike':'micro','scooter':'micro',
     'car':'vehicular','bus':'vehicular','motorcycle':'vehicular','train':'rail'}
E['family']=E.label.map(FAM)
pv=E.pivot_table(index='family',columns='city',values='hpd',aggfunc='median')
print(pv.round(5).to_string())
print("\nped/veh MEDIAN ratio per city:")
for c in CITIES:
    try: print("  %-8s %.2fx"%(c,pv.loc['pedestrian',c]/pv.loc['vehicular',c]))
    except Exception as e: print("  %-8s n/a"%c)
print("\nIQR of hpd by family (pooled):")
print(E.groupby('family').hpd.describe(percentiles=[.25,.5,.75])[['count','25%','50%','75%']].round(5).to_string())
