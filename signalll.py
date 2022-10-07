import uproot3 as ur
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

signal = ur.open('/cephfs/user/s6flkirf/master_thesis/codemasterthesis/finalScripts/classifier/plots/NarrowStrip/sgn_converted_NN_results.root')["SinglePhoton"].pandas.df()
signal["y_predTight"] = ur.open('/cephfs/user/s6flkirf/master_thesis/codemasterthesis/finalScripts/classifier/plots/RelaxedTight/sgn_converted_NN_results.root')["SinglePhoton"]["y_pred"].array()
signal=signal.loc[signal["y_IsLoose"]==1]
print("rows:",len(signal))

def dictgenerator(df, region, IsTrack):
    if IsTrack == True :
        df_neww = df.loc[(df["y_topoetcone20"]< (6.5*0.01* df["y_pt"])) & (df["y_ptcone20"]< (0.05 * df["y_pt"]))]
        print("track")
    if IsTrack == False :
        df_neww = df
    if region == 1:
        df_new = df_neww.loc[(df_neww["y_pred"]>0.6) & (df_neww["y_predTight"]>0.55)]
    if region == 2:
        df_new = df_neww.loc[(df_neww["y_pred"]>0.6) & (df_neww["y_predTight"]<0.55)]
    if region == 3:
        df_new = df_neww.loc[(df_neww["y_pred"]<0.6) & (df_neww["y_predTight"]>0.55)]
    if region == 4:
        df_new = df_neww.loc[(df_neww["y_pred"]<0.6) & (df_neww["y_predTight"]<0.55)]
    ptcuts=[25,30,35,40,45,50,60,80,100,125,150,175,500,1500]
    dictt={key:np.array([]) for  key in ["eta1","eta2","eta3","eta4"]}
    for i in range(len(ptcuts)-1):
        dictt["eta1"]= np.append(dictt["eta1"],df_new.loc[(df_new['y_pt'] >= ptcuts[i]) & (df_new['y_pt'] < ptcuts[i+1]) & (abs(df_new['y_eta']) >= 0) & (abs(df_new['y_eta']) < 0.60)]["y_pred"].count())
        dictt["eta2"]= np.append(dictt["eta2"],df_new.loc[(df_new['y_pt'] >= ptcuts[i]) & (df_new['y_pt'] < ptcuts[i+1]) & (abs(df_new['y_eta']) >= 0.60) & (abs(df_new['y_eta']) < 1.37)]["y_pred"].count())
        dictt["eta3"]= np.append(dictt["eta3"],df_new.loc[(df_new['y_pt'] >= ptcuts[i]) & (df_new['y_pt'] < ptcuts[i+1]) & (abs(df_new['y_eta']) > 1.52) & (abs(df_new['y_eta']) < 1.81)]["y_pred"].count())
        dictt["eta4"]= np.append(dictt["eta4"],df_new.loc[(df_new['y_pt'] >= ptcuts[i]) & (df_new['y_pt'] < ptcuts[i+1]) & (abs(df_new['y_eta']) >= 1.81) & (abs(df_new['y_eta']) < 2.37)]["y_pred"].count())
    return dictt

N1prompt=dictgenerator(signal,1,False)
N2prompt=dictgenerator(signal,2,False)
N3prompt=dictgenerator(signal,3,False)
N4prompt=dictgenerator(signal,4,False)
N1prompt_iso=dictgenerator(signal,1,True)
N2prompt_iso=dictgenerator(signal,2,True)
N3prompt_iso=dictgenerator(signal,3,True)
N4prompt_iso=dictgenerator(signal,4,True)

N234prompt={"eta1": N2prompt["eta1"]+N3prompt["eta1"]+N4prompt["eta1"],"eta2": N2prompt["eta2"]+N3prompt["eta2"]+N4prompt["eta2"],"eta3": N2prompt["eta3"]+N3prompt["eta3"]+N4prompt["eta3"],"eta4": N2prompt["eta4"]+N3prompt["eta4"]+N4prompt["eta4"]}
N1234prompt={"eta1": N1prompt["eta1"]+N2prompt["eta1"]+N3prompt["eta1"]+N4prompt["eta1"],"eta2": N1prompt["eta2"]+N2prompt["eta2"]+N3prompt["eta2"]+N4prompt["eta2"],"eta3": N1prompt["eta3"]+N2prompt["eta3"]+N3prompt["eta3"]+N4prompt["eta3"],"eta4": N1prompt["eta4"]+N2prompt["eta4"]+N3prompt["eta4"]+N4prompt["eta4"]}
N234prompt_iso={"eta1": N2prompt_iso["eta1"]+N3prompt_iso["eta1"]+N4prompt_iso["eta1"],"eta2": N2prompt_iso["eta2"]+N3prompt_iso["eta2"]+N4prompt_iso["eta2"],"eta3": N2prompt_iso["eta3"]+N3prompt_iso["eta3"]+N4prompt_iso["eta3"],"eta4": N2prompt_iso["eta4"]+N3prompt_iso["eta4"]+N4prompt_iso["eta4"]}
N1234prompt_iso={"eta1": N1prompt_iso["eta1"]+N2prompt_iso["eta1"]+N3prompt_iso["eta1"]+N4prompt_iso["eta1"],"eta2": N1prompt_iso["eta2"]+N2prompt_iso["eta2"]+N3prompt_iso["eta2"]+N4prompt_iso["eta2"],"eta3": N1prompt_iso["eta3"]+N2prompt_iso["eta3"]+N3prompt_iso["eta3"]+N4prompt_iso["eta3"],"eta4": N1prompt_iso["eta4"]+N2prompt_iso["eta4"]+N3prompt_iso["eta4"]+N4prompt_iso["eta4"]}

f_p={"eta1": (N3prompt["eta1"]/N1234prompt["eta1"]),"eta2": (N3prompt["eta2"]/N1234prompt["eta2"]),"eta3": (N3prompt["eta3"]/N1234prompt["eta3"]),"eta4": (N3prompt["eta4"]/N1234prompt["eta4"])}
f_a={"eta1": (N234prompt["eta1"]/N1234prompt["eta1"]),"eta2": (N234prompt["eta2"]/N1234prompt["eta2"]),"eta3": (N234prompt["eta3"]/N1234prompt["eta3"]),"eta4": (N234prompt["eta4"]/N1234prompt["eta4"])}
sEpsHatID={"eta1": N1prompt_iso["eta1"]/N1prompt["eta1"], "eta2": N1prompt_iso["eta2"]/N1prompt["eta2"], "eta3": N1prompt_iso["eta3"]/N1prompt["eta3"], "eta4": N1prompt_iso["eta4"]/N1prompt["eta4"]}
sEpsHat={"eta1": (N1234prompt_iso["eta1"]/N1234prompt["eta1"]), "eta2": (N1234prompt_iso["eta2"]/N1234prompt["eta2"]), "eta3": (N1234prompt_iso["eta3"]/N1234prompt["eta3"]), "eta4": (N1234prompt_iso["eta4"]/ N1234prompt["eta4"])}
sEpsHat3={"eta1": (N3prompt_iso["eta1"]/N3prompt["eta1"]), "eta2": (N3prompt_iso["eta2"]/N3prompt["eta2"]), "eta3": (N3prompt_iso["eta3"]/N3prompt["eta3"]), "eta4": (N3prompt_iso["eta4"]/N3prompt["eta4"])}
sEpsHat234={"eta1": (N234prompt_iso["eta1"]/N234prompt["eta1"]), "eta2": (N234prompt_iso["eta2"]/N234prompt["eta2"]), "eta3": (N234prompt_iso["eta3"]/N234prompt["eta3"]), "eta4": (N234prompt_iso["eta4"]/N234prompt["eta4"])}


dicts=[f_p, f_a, sEpsHat, sEpsHat234, sEpsHat3, sEpsHatID, N1234prompt]

for i in range(len(dicts)):
    dicttt=dicts[i]
    dicttt["eta1"]=dicttt["eta1"].tolist()
    dicttt["eta2"]=dicttt["eta2"].tolist()
    dicttt["eta3"]=dicttt["eta3"].tolist()
    dicttt["eta4"]=dicttt["eta4"].tolist()
print(type(f_p))
with open('/cephfs/user/suslu/sg_jsonfiles/fp.json', 'w') as fp:
     json.dump(f_p, fp)
with open('/cephfs/user/suslu/sg_jsonfiles/fa.json', 'w') as fp:
     json.dump(f_a, fp)
with open('/cephfs/user/suslu/sg_jsonfiles/sEpsHatID.json', 'w') as fp:
     json.dump(sEpsHatID, fp)
with open('/cephfs/user/suslu/sg_jsonfiles/sEpsHat.json', 'w') as fp:
     json.dump(sEpsHat, fp)
with open('/cephfs/user/suslu/sg_jsonfiles/sEpsHat3.json', 'w') as fp:
     json.dump(sEpsHat3, fp)
with open('/cephfs/user/suslu/sg_jsonfiles/sEpsHat234.json', 'w') as fp:
     json.dump(sEpsHat234, fp)
with open('/cephfs/user/suslu/sg_jsonfiles/Ns.json', 'w') as fp:
     json.dump(N1234prompt, fp)