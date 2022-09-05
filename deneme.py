import uproot3 as ur
import matplotlib.pyplot as plt
import pandas 
import numpy as np
import json
from collections import Counter

df15 = ur.open('/cephfs/user/s6flkirf/master_thesis/codemasterthesis/finalScripts/classifier/plots/NarrowStrip/data15_converted_NN_results.root')["SinglePhoton"].pandas.df()
df15["y_predTight"] = ur.open('/cephfs/user/s6flkirf/master_thesis/codemasterthesis/finalScripts/classifier/plots/RelaxedTight/data15_converted_NN_results.root')["SinglePhoton"]["y_pred"].array()
df16 = ur.open('/cephfs/user/s6flkirf/master_thesis/codemasterthesis/finalScripts/classifier/plots/NarrowStrip/data16_converted_NN_results.root')["SinglePhoton"].pandas.df()
df16["y_predTight"] = ur.open('/cephfs/user/s6flkirf/master_thesis/codemasterthesis/finalScripts/classifier/plots/RelaxedTight/data16_converted_NN_results.root')["SinglePhoton"]["y_pred"].array()
df17 = ur.open('/cephfs/user/s6flkirf/master_thesis/codemasterthesis/finalScripts/classifier/plots/NarrowStrip/data17_converted_NN_results.root')["SinglePhoton"].pandas.df()
df17["y_predTight"] = ur.open('/cephfs/user/s6flkirf/master_thesis/codemasterthesis/finalScripts/classifier/plots/RelaxedTight/data17_converted_NN_results.root')["SinglePhoton"]["y_pred"].array()
df18 = ur.open('/cephfs/user/s6flkirf/master_thesis/codemasterthesis/finalScripts/classifier/plots/NarrowStrip/data18_converted_NN_results.root')["SinglePhoton"].pandas.df()
df18["y_predTight"] = ur.open('/cephfs/user/s6flkirf/master_thesis/codemasterthesis/finalScripts/classifier/plots/RelaxedTight/data18_converted_NN_results.root')["SinglePhoton"]["y_pred"].array()

df=pandas.concat([df15,df16,df17,df18])

def dictgenerator(df, region, IsTrack):
    df=df.iloc[:10000]
    if IsTrack == True :
        df_new = df.loc[(df["y_topoetcone20"]< (6.5*0.01* df["y_pt"])) & (df["y_ptcone20"]< (0.05 * df["y_pt"]))]
    if IsTrack == False :
        df_new = df
    if region == 1:
        df_new = df_new.loc[(df["y_pred"]>0.5) & (df["y_predTight"]>0.5)]
    if region == 2:
        df_new = df_new.loc[(df["y_pred"]>0.5) & (df["y_predTight"]<0.5)]
    if region == 3:
        df_new = df_new.loc[(df["y_pred"]<0.5) & (df["y_predTight"]>0.5)]
    if region == 4:
        df_new = df_new.loc[(df["y_pred"]<0.5) & (df["y_predTight"]<0.5)]
    ptcuts=[25,30,35,40,45,50,60,80,100,125,150,175,500,1500]
    dictt={key:np.array([]) for  key in ["eta1","eta2","eta3","eta4"]}
    for i in range(len(ptcuts)-1):
        dictt["eta1"]= np.append(dictt["eta1"],df_new.loc[(df_new['y_pt'] >= ptcuts[i]) & (df_new['y_pt'] < ptcuts[i+1]) & (abs(df_new['y_eta']) >= 0) & (abs(df_new['y_eta']) < 0.60)]["y_pred"].count())
        dictt["eta2"]= np.append(dictt["eta2"],df_new.loc[(df_new['y_pt'] >= ptcuts[i]) & (df_new['y_pt'] < ptcuts[i+1]) & (abs(df_new['y_eta']) >= 0.60) & (abs(df_new['y_eta']) < 1.37)]["y_pred"].count())
        dictt["eta3"]= np.append(dictt["eta3"],df_new.loc[(df_new['y_pt'] >= ptcuts[i]) & (df_new['y_pt'] < ptcuts[i+1]) & (abs(df_new['y_eta']) > 1.52) & (abs(df_new['y_eta']) < 1.81)]["y_pred"].count())
        dictt["eta4"]= np.append(dictt["eta4"],df_new.loc[(df_new['y_pt'] >= ptcuts[i]) & (df_new['y_pt'] < ptcuts[i+1]) & (abs(df_new['y_eta']) >= 1.81) & (abs(df_new['y_eta']) < 2.37)]["y_pred"].count())
    return dictt

N1=dictgenerator(df,1,False)
N2=dictgenerator(df,2,False)
N3=dictgenerator(df,3,False)
N4=dictgenerator(df,4,False)
N1_iso=dictgenerator(df,1,True)
N2_iso=dictgenerator(df,2,True)
N3_iso=dictgenerator(df,3,True)
N4_iso=dictgenerator(df,4,True)

N234={"eta1": N2["eta1"]+N3["eta1"]+N4["eta1"],"eta2": N2["eta2"]+N3["eta2"]+N4["eta2"],"eta3": N2["eta3"]+N3["eta3"]+N4["eta3"],"eta4": N2["eta4"]+N3["eta4"]+N4["eta4"]}
N1234={"eta1": N1["eta1"]+N2["eta1"]+N3["eta1"]+N4["eta1"],"eta2": N1["eta2"]+N2["eta2"]+N3["eta2"]+N4["eta2"],"eta3": N1["eta3"]+N2["eta3"]+N3["eta3"]+N4["eta3"],"eta4": N1["eta4"]+N2["eta4"]+N3["eta4"]+N4["eta4"]}
N234_iso={"eta1": N2_iso["eta1"]+N3_iso["eta1"]+N4_iso["eta1"],"eta2": N2_iso["eta2"]+N3_iso["eta2"]+N4_iso["eta2"],"eta3": N2_iso["eta3"]+N3_iso["eta3"]+N4_iso["eta3"],"eta4": N2_iso["eta4"]+N3_iso["eta4"]+N4_iso["eta4"]}
N1234_iso={"eta1": N1_iso["eta1"]+N2_iso["eta1"]+N3_iso["eta1"]+N4_iso["eta1"],"eta2": N1_iso["eta2"]+N2_iso["eta2"]+N3_iso["eta2"]+N4_iso["eta2"],"eta3": N1_iso["eta3"]+N2_iso["eta3"]+N3_iso["eta3"]+N4_iso["eta3"],"eta4": N1_iso["eta4"]+N2_iso["eta4"]+N3_iso["eta4"]+N4_iso["eta4"]}

R_p={"eta1": (N3["eta1"]/N1234["eta1"]),"eta2": (N3["eta2"]/N1234["eta2"]),"eta3": (N3["eta3"]/N1234["eta3"]),"eta4": (N3["eta4"]/N1234["eta4"])}
R_a={"eta1": (N234["eta1"]/N1234["eta1"]),"eta2": (N234["eta2"]/N1234["eta2"]),"eta3": (N234["eta3"]/N1234["eta3"]),"eta4": (N234["eta4"]/N1234["eta4"])}
EpsHatID={"eta1": N1_iso["eta1"]/N1["eta1"], "eta2": N1_iso["eta2"]/N1["eta2"], "eta3": N1_iso["eta3"]/N1["eta3"], "eta4": N1_iso["eta4"]/N1["eta4"]}
EpsHat={"eta1": (N1234_iso["eta1"]/N1234["eta1"]), "eta2": (N1234_iso["eta2"]/N1234["eta2"]), "eta3": (N1234_iso["eta3"]/N1234["eta3"]), "eta4": (N1234_iso["eta4"]/ N1234["eta4"])}
EpsHat3={"eta1": (N3_iso["eta1"]/N3["eta1"]), "eta2": (N3_iso["eta2"]/N3["eta2"]), "eta3": (N3_iso["eta3"]/N3["eta3"]), "eta4": (N3_iso["eta4"]/N3["eta4"])}
EpsHat234={"eta1": (N234_iso["eta1"]/N234["eta1"]), "eta2": (N234_iso["eta2"]/N234["eta2"]), "eta3": (N234_iso["eta3"]/N234["eta3"]), "eta4": (N234_iso["eta4"]/N234["eta4"])}

dumped=json.dumps(R_a,cls=NumpyEncoder)

with open('/cephfs/user/suslu/data.json', 'w') as fp:
    json.dump(dumped, fp)
#    json.dump(R_a, fp)
#    json.dump(EpsHatID, fp)
#    json.dump(EpsHat, fp)
#    json.dump(EpsHat3, fp)
#    json.dump(EpsHat234, fp)

#JSON save:
#import json
#with open('data.json', 'w') as fp:
#    json.dump(data, fp)
#Supply extra arguments, like sort_keys or indent, to get a pretty result. The argument sort_keys will sort the keys alphabetically and indent will indent your data structure with indent=N spaces.
#json.dump(data, fp, sort_keys=True, indent=4)
#JSON load:
#with open('data.json', 'r') as fp:
#    data = json.load(fp)
