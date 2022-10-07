import uproot3 as ur
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

backg = ur.open('/cephfs/user/s6flkirf/master_thesis/codemasterthesis/finalScripts/classifier/plots/NarrowStrip/bkg_converted_NN_results.root')["SinglePhoton"].pandas.df()
backg["y_predTight"] = ur.open('/cephfs/user/s6flkirf/master_thesis/codemasterthesis/finalScripts/classifier/plots/RelaxedTight/bkg_converted_NN_results.root')["SinglePhoton"]["y_pred"].array()

backg = backg.loc[backg["y_IsLoose"]==1]

def dictgenerator(df, region, IsTrack):
    if IsTrack == True :
        df_neww = df.loc[(df["y_topoetcone20"]< (6.5*0.01* df["y_pt"])) & (df["y_ptcone20"]< (0.05 * df["y_pt"]))]
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
    ptcuts=[25,1500]
    dictt={key:np.array([]) for  key in ["eta1","eta2","eta3","eta4"]}
    for i in range(len(ptcuts)-1):
        dictt["eta1"]= np.append(dictt["eta1"],df_new.loc[(df_new['y_pt'] >= ptcuts[i]) & (df_new['y_pt'] < ptcuts[i+1]) & (abs(df_new['y_eta']) >= 0) & (abs(df_new['y_eta']) < 0.60)]['mcTotWeight'].sum())
        dictt["eta2"]= np.append(dictt["eta2"],df_new.loc[(df_new['y_pt'] >= ptcuts[i]) & (df_new['y_pt'] < ptcuts[i+1]) & (abs(df_new['y_eta']) >= 0.60) & (abs(df_new['y_eta']) < 1.37)]['mcTotWeight'].sum())
        dictt["eta3"]= np.append(dictt["eta3"],df_new.loc[(df_new['y_pt'] >= ptcuts[i]) & (df_new['y_pt'] < ptcuts[i+1]) & (abs(df_new['y_eta']) > 1.52) & (abs(df_new['y_eta']) < 1.81)]['mcTotWeight'].sum())
        dictt["eta4"]= np.append(dictt["eta4"],df_new.loc[(df_new['y_pt'] >= ptcuts[i]) & (df_new['y_pt'] < ptcuts[i+1]) & (abs(df_new['y_eta']) >= 1.81) & (abs(df_new['y_eta']) < 2.37)]['mcTotWeight'].sum())
    return dictt

N1b=dictgenerator(backg,1,False)
N2b=dictgenerator(backg,2,False)
N3b=dictgenerator(backg,3,False)
N4b=dictgenerator(backg,4,False)
N1b_iso=dictgenerator(backg,1,True)
N2b_iso=dictgenerator(backg,2,True)
N3b_iso=dictgenerator(backg,3,True)
N4b_iso=dictgenerator(backg,4,True)

N234b={"eta1": N2b["eta1"]+N3b["eta1"]+N4b["eta1"],"eta2": N2b["eta2"]+N3b["eta2"]+N4b["eta2"],"eta3": N2b["eta3"]+N3b["eta3"]+N4b["eta3"],"eta4": N2b["eta4"]+N3b["eta4"]+N4b["eta4"]}
N1234b={"eta1": N1b["eta1"]+N2b["eta1"]+N3b["eta1"]+N4b["eta1"],"eta2": N1b["eta2"]+N2b["eta2"]+N3b["eta2"]+N4b["eta2"],"eta3": N1b["eta3"]+N2b["eta3"]+N3b["eta3"]+N4b["eta3"],"eta4": N1b["eta4"]+N2b["eta4"]+N3b["eta4"]+N4b["eta4"]}
N234b_iso={"eta1": N2b_iso["eta1"]+N3b_iso["eta1"]+N4b_iso["eta1"],"eta2": N2b_iso["eta2"]+N3b_iso["eta2"]+N4b_iso["eta2"],"eta3": N2b_iso["eta3"]+N3b_iso["eta3"]+N4b_iso["eta3"],"eta4": N2b_iso["eta4"]+N3b_iso["eta4"]+N4b_iso["eta4"]}
N1234b_iso={"eta1": N1b_iso["eta1"]+N2b_iso["eta1"]+N3b_iso["eta1"]+N4b_iso["eta1"],"eta2": N1b_iso["eta2"]+N2b_iso["eta2"]+N3b_iso["eta2"]+N4b_iso["eta2"],"eta3": N1b_iso["eta3"]+N2b_iso["eta3"]+N3b_iso["eta3"]+N4b_iso["eta3"],"eta4": N1b_iso["eta4"]+N2b_iso["eta4"]+N3b_iso["eta4"]+N4b_iso["eta4"]}

bEpsHatID={"eta1": N1b_iso["eta1"]/N1b["eta1"], "eta2": N1b_iso["eta2"]/N1b["eta2"], "eta3": N1b_iso["eta3"]/N1b["eta3"], "eta4": N1b_iso["eta4"]/N1b["eta4"]}
bEpsHat1234={"eta1": (N1234b_iso["eta1"]/N1234b["eta1"]), "eta2": (N1234b_iso["eta2"]/N1234b["eta2"]), "eta3": (N1234b_iso["eta3"]/N1234b["eta3"]), "eta4": (N1234b_iso["eta4"]/ N1234b["eta4"])}
bEpsHat3={"eta1": (N3b_iso["eta1"]/N3b["eta1"]), "eta2": (N3b_iso["eta2"]/N3b["eta2"]), "eta3": (N3b_iso["eta3"]/N3b["eta3"]), "eta4": (N3b_iso["eta4"]/N3b["eta4"])}
bEpsHat234={"eta1": (N234b_iso["eta1"]/N234b["eta1"]), "eta2": (N234b_iso["eta2"]/N234b["eta2"]), "eta3": (N234b_iso["eta3"]/N234b["eta3"]), "eta4": (N234b_iso["eta4"]/N234b["eta4"])}

DeltabEpsHatID={"eta1": abs(bEpsHatID["eta1"]-bEpsHat3["eta1"])/(bEpsHatID["eta1"]), "eta2": abs(bEpsHatID["eta2"]-bEpsHat3["eta2"])/bEpsHatID["eta2"], "eta3": abs(bEpsHatID["eta3"]-bEpsHat3["eta3"])/bEpsHatID["eta3"], "eta4": abs(bEpsHatID["eta4"]-bEpsHat3["eta4"])/bEpsHatID["eta4"]}
DeltabEpsHat1234={"eta1": abs(bEpsHat1234["eta1"]-bEpsHat234["eta1"])/bEpsHat1234["eta1"], "eta2": abs(bEpsHat1234["eta2"]-bEpsHat234["eta2"])/bEpsHat1234["eta2"], "eta3": abs(bEpsHat1234["eta3"]-bEpsHat234["eta3"])/bEpsHat1234["eta3"], "eta4": abs(bEpsHat1234["eta4"]-bEpsHat234["eta4"])/bEpsHat1234["eta4"]}

print(DeltabEpsHat1234)
print(DeltabEpsHatID)

dicts=[DeltabEpsHat1234,DeltabEpsHatID]

for i in range(len(dicts)):
    dicttt=dicts[i]
    dicttt["eta1"]=dicttt["eta1"].tolist()
    dicttt["eta2"]=dicttt["eta2"].tolist()
    dicttt["eta3"]=dicttt["eta3"].tolist()
    dicttt["eta4"]=dicttt["eta4"].tolist()

with open('/cephfs/user/suslu/jsonfiles/DeltabEpsHat1234.json', 'w') as fp:
     json.dump(DeltabEpsHat1234, fp)
with open('/cephfs/user/suslu/jsonfiles/DeltabEpsHatID.json', 'w') as fp:
     json.dump(DeltabEpsHatID, fp)