import uproot3 as ur
import matplotlib.pyplot as plt
import pandas 
import numpy as np


df = ur.open('/cephfs/user/s6flkirf/master_thesis/codemasterthesis/finalScripts/classifier/plots/NarrowStrip/data15_converted_NN_results.root')["SinglePhoton"].pandas.df()
df["y_predTight"] = ur.open('/cephfs/user/s6flkirf/master_thesis/codemasterthesis/finalScripts/classifier/plots/RelaxedTight/data15_converted_NN_results.root')["SinglePhoton"]["y_pred"].array()
df=df.iloc[:1000]

df15 = ur.open('/cephfs/user/s6flkirf/master_thesis/codemasterthesis/finalScripts/classifier/plots/NarrowStrip/data15_converted_NN_results.root')["SinglePhoton"].pandas.df()
df16 = ur.open('/cephfs/user/s6flkirf/master_thesis/codemasterthesis/finalScripts/classifier/plots/NarrowStrip/data16_converted_NN_results.root')["SinglePhoton"].pandas.df()

dfs=dict(df15,df16)
print(dfs)



#The code 2 blocks below basicallly counts the number of events that passes the event selection in the .loc[] function. It counts the number ["y_pred"] occurances (it could also be any other variable.)

N_ID_1=df.loc[(df["y_pred"]>0.5) & (df["y_predTight"]>0.5)]["y_pred"].count()
N_ID_2=df.loc[(df["y_pred"]>0.5) & (df["y_predTight"]<0.5)]["y_pred"].count()
N_ID_3=df.loc[(df["y_pred"]<0.5) & (df["y_predTight"]>0.5)]["y_pred"].count()
N_ID_4=df.loc[(df["y_pred"]<0.5) & (df["y_predTight"]<0.5)]["y_pred"].count()

N1isolated=df.loc[(df["y_pred"]>0.5) & (df["y_predTight"]>0.5) & (df["y_topoetcone20"]< (6.5*0.01* df["y_pt"])) & (df["y_ptcone20"]< (0.05 * df["y_pt"]))]["y_pred"].count()
N2isolated=df.loc[(df["y_pred"]>0.5) & (df["y_predTight"]<0.5) & (df["y_topoetcone20"]< (6.5*0.01* df["y_pt"])) & (df["y_ptcone20"]< (0.05 * df["y_pt"]))]["y_pred"].count()
N3isolated=df.loc[(df["y_pred"]<0.5) & (df["y_predTight"]>0.5) & (df["y_topoetcone20"]< (6.5*0.01* df["y_pt"])) & (df["y_ptcone20"]< (0.05 * df["y_pt"]))]["y_pred"].count()
N4isolated=df.loc[(df["y_pred"]<0.5) & (df["y_predTight"]<0.5) & (df["y_topoetcone20"]< (6.5*0.01* df["y_pt"])) & (df["y_ptcone20"]< (0.05 * df["y_pt"]))]["y_pred"].count()

NT= N_ID_1 + N_ID_2 +N_ID_3 + N_ID_4 # N^T


#plt.scatter(tightarray[:1000],narrowarray[:1000])
#plt.show()

Rp= N_ID_3/NT
Ra= (N_ID_2 +N_ID_3 + N_ID_4)/NT
hatepsID= N1isolated / N_ID_1
hateps= (N1isolated+N2isolated+N3isolated+N4isolated)/ NT
hateps3= N3isolated / N_ID_3
hateps234= (N2isolated+N3isolated+N4isolated)/(N_ID_2+N_ID_3+N_ID_4)


print("N1isolated: ", N1isolated)

print("Rp: ", Rp, "Ra: ",Ra)

print(N_ID_1, N_ID_2, N_ID_3, N_ID_4)

