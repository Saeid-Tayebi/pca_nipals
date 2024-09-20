#%%
import numpy as np
import pca_module as pca_m
from MyPcaClass import MyPca as pca_c
# Model further settings
np.set_printoptions(precision=4)

# Generating data
Num_observation=30
Number_variables=3
X =np.random.rand(Num_observation,Number_variables)

Beta=np.array([3,2,1])
Y=(X@Beta.T).reshape(-1,1)
color_map_data =  Y.reshape(-1) #color map should contain 1D data
# Set parameters
Num_com = 2            # Number of PLS components (=Number of X Variables)
alpha = 0.95            # Confidence limit (=0.95)
X_test=np.array([[0.9,0.1,0.2],[0.5 , 0.4 , 0.9]])
scores_pca=np.array([1,2])
# Model implementation as a Module

pca_model=pca_m.pca_nipals(X,Num_com,alpha)

x_hat,T_score,Hotelin_T2,SPE_X=pca_m.pca_evaluation(pca_model,X_test)
print(f'x_hat={x_hat}\n',f'T_score={T_score}\n',f'Hotelin_T2={Hotelin_T2}\n',f'SPE_X={SPE_X}\n')

pca_m.visual_plot(pca_model,scores_pca,X_test,color_map_data) # 

#%% 
# Model implementation as a Class
MyPcaModel=pca_c()
MyPcaModel.train(X,Num_com,alpha)

x_hat,T_score,Hotelin_T2,SPE_X=MyPcaModel.evaluation(X_test)
print(f'x_hat={x_hat}\n',f'T_score={T_score}\n',f'Hotelin_T2={Hotelin_T2}\n',f'SPE_X={SPE_X}\n')

MyPcaModel.visual_plot(scores_pca,X_test,color_map_data,data_labeling=True)
input('')


# %%
