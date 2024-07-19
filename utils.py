import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random

L2018=np.load('/home/malo/Stage/Data/data modifiées 11 classes/l2018_modif.npz',allow_pickle=True)
L2019=np.load('/home/malo/Stage/Data/data modifiées 11 classes/l2019_modif.npz',allow_pickle=True)
L2020=np.load('/home/malo/Stage/Data/data modifiées 11 classes/l2020_modif.npz',allow_pickle=True)
R2018=np.load('/home/malo/Stage/Data/data modifiées 11 classes/r2018_modif.npz',allow_pickle=True)
R2019=np.load('/home/malo/Stage/Data/data modifiées 11 classes/r2019_modif.npz',allow_pickle=True)
R2020=np.load('/home/malo/Stage/Data/data modifiées 11 classes/r2020_modif.npz',allow_pickle=True)
T2018=np.load('/home/malo/Stage/Data/data modifiées 11 classes/t2018_modif.npz',allow_pickle=True)
T2019=np.load('/home/malo/Stage/Data/data modifiées 11 classes/t2019_modif.npz',allow_pickle=True)
T2020=np.load('/home/malo/Stage/Data/data modifiées 11 classes/t2020_modif.npz',allow_pickle=True)

class dropout:
    def __init__(self, p): # p est la probabilité de conservation des donées
        self.p = p
    def augment(self,x,mask):
        
        
        size = [x.shape[0],x.shape[1]]
        
        
       
        suppr = torch.bernoulli(self.p * torch.ones(size)).cuda() # on va conserver les données là où il y a un 1 et supprimer celles où où il y a un 0
        
        
        
        mask = mask.masked_fill(suppr==0,0)
        suppr = suppr.unsqueeze(2)
        suppr = suppr.repeat(1,1,2)
        
        x = x.masked_fill(suppr==0,0)
        
        return x,mask
class identité:
        
        def augment(x,mask):
          return(x,mask)
def get_day_count(dates,ref_day='09-01'):
    # Days elapsed from 'ref_day' of the year in dates[0]
    ref = np.datetime64(f'{dates.astype("datetime64[Y]")[0]}-'+ref_day)
    days_elapsed = (dates - ref).astype('timedelta64[D]').astype(int) #(dates - ref_day).astype('timedelta64[D]').astype(int)#
    return torch.tensor(days_elapsed,dtype=torch.long)

def add_mask(values,mask): # permet d'attacher les mask aux données pour pouvoir faire les batchs sans perdre le mask
    mask=mask.unsqueeze(0).unsqueeze(-1)
    shape=values.shape
    mask=mask.expand(shape[0],-1,-1)
    values=torch.tensor(values,dtype=torch.float32)

    valuesWmask=torch.cat((values,mask),dim=-1)
    return valuesWmask

def comp (data,msk) : #permet de formater les données avec 365 points d'acquisitions
  data_r={'X_SAR':data['X_SAR'],'y':data['y'],'dates_SAR':data['dates_SAR']}
  ref=data['dates_SAR'][0]
  j_p=(data['dates_SAR']-ref).astype('timedelta64[D]').astype(int)
  année=list(range(365))

  année = [ref + np.timedelta64(j, 'D') for j in année ]
  mask = []

  for i,jour in enumerate(année):
    if jour not in data['dates_SAR']:

      mask+=[0]
      msk=np.insert(msk,i,0)
      data_r['dates_SAR']=np.insert(data_r['dates_SAR'],i,jour)
      data_r['X_SAR']=np.insert(data_r['X_SAR'],i,[0,0],axis=1)
    else:
      mask+=[1]


  mask=torch.tensor(mask,dtype=torch.float32)
  msk=torch.tensor(msk,dtype=torch.float32)
  return data_r,mask,msk


def suppr (data,ratio):
  data_r={'X_SAR':data['X_SAR'],'y':data['y'],'dates_SAR':data['dates_SAR']}
  ref=data['dates_SAR'][0]
  nbr,seq_len,channels=data['X_SAR'].shape #(nbr,seq_len,channels)
  
  nbr_indice=int(seq_len*ratio)
  indice=list(range(seq_len))
  indice=random.sample(indice,nbr_indice)
  mask=[0 if i in indice else 1 for i in range(seq_len)]
  mask=torch.tensor(mask)

  data_r['X_SAR']=torch.tensor(data_r['X_SAR'])
  data_r['X_SAR']=data_r['X_SAR'].permute(0,2,1)
  data_r['X_SAR']=data_r['X_SAR'].masked_fill(mask==0,0)
  data_r['X_SAR']=data_r['X_SAR'].permute(0,2,1)
  data_r['X_SAR']=data_r['X_SAR'].numpy()
  mask=mask.numpy()
  return data_r,mask




    
    
# preparation train-val-test pas encore de dataloader, # mise au format 365 jours + masque des données
def tvt_split(data): 
  mapping={1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9,11:10}


  data,msk=suppr(data,0) # peut être utiliser si l'on souhaite diminuer la quantité de points d'acquisition dans les données
  data,_,mask=comp(data,msk) # rempli les données pour les mettre au fromat 365 j et donne le mask correspondant aux jours où on a mit un 0
  values=data['X_SAR']
  data_shape=data['X_SAR'].shape
  dates=data['dates_SAR']




  labels=data['y']
  labels=[mapping[v] if v in mapping else v for v in labels ]

  max_values = np.percentile(values,99)
  min_values = np.percentile(values,1)
  values_norm=(values-min_values)/(max_values-min_values)
  values_norm[values_norm>1] = 1
  values_norm[values_norm<0] = 0
  values = values_norm                                      # les données sont normalisées
  values=add_mask(values,mask)   
  sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
  indice = sss.split(values,labels)

  tv_index, test_index = next(indice)

  values_tv=[]
  values_test=[]
  labels_tv=[]
  labels_test=[]
  for i in tv_index :
    values_tv+=[values[i]]
    labels_tv+=[labels[i]]
  for j in test_index :
    values_test+=[values[j]]
    labels_test+=[labels[j]]


  sss2=StratifiedShuffleSplit(n_splits=1,test_size=0.25,random_state=0)
  indice2=sss2.split(values_tv,labels_tv)
  train_index,validation_index = next(indice2)

  values_train=[]
  values_validation=[]
  labels_train=[]
  labels_validation=[]

  for i in train_index :
    values_train+=[values_tv[i]]
    labels_train+=[labels_tv[i]]
  for j in validation_index :
    values_validation += [values_tv[j]]
    labels_validation += [labels_tv[j]]


  values_train=np.array(values_train)
  values_validation=np.array(values_validation)
  values_test=np.array(values_test)
  labels_train=np.array(labels_train)
  labels_validation=np.array(labels_validation)
  labels_test=np.array(labels_test)
  
  data_train = {'X_SAR':values_train, 'y':labels_train, 'dates_SAR':dates}
  data_validation = {'X_SAR': values_validation, 'y':labels_validation, 'dates_SAR':dates}
  data_test = {'X_SAR': values_test,'y':labels_test, 'dates_SAR':dates}



  



  return data_train,data_validation,data_test,dates,data_shape

# sélection des données pour le test et régularisatino et masking
def selection(data,nbr,role,nbr_ssl=200): # role peut être "source" ou "target"
    
   
    
    
    values = data['X_SAR']
    labels = data['y']
    dates = data['dates_SAR']
    
    if role == "source":
        selected_data = []
        selected_labels = []
        # Pour chaque label de 0 à 10
        for label in range(11):
            
            # Sélection des indices correspondant à ce label
            indices = np.where(labels == label)[0]
            
            # Vérification qu'il y a au moins 100 éléments pour ce label
            if len(indices) >= nbr:
                
                # Sélection aléatoire de 100 indices parmi ceux disponibles
                indices = np.random.choice(indices, nbr, replace=False)
                
                
                
                
                # Ajout des données et labels sélectionnés aux tableaux de résultats
                selected_data.append(values[indices])
                selected_labels.append(labels[indices])
                
                
            elif len(indices) == 0: 
                print(f'il n\'y a pas {label} dans les data')
            else:
                
            
                
                
                selected_data.append(values[indices])
                selected_labels.append(labels[indices])
                
                
                
            
        selected_data = np.vstack(selected_data)
        selected_labels = np.hstack(selected_labels)

            
        selection_finale = {'X_SAR':selected_data,'y':[[a,a] for a in selected_labels],'dates_SAR':dates}
        
    
        
        return selection_finale # attention ici les données sont triées par classe
    elif role == "target":
        selected_data_ul1 = []
        selected_data_ul2 = []
        selected_labels_ul1 = []
        selected_labels_ul2 = []
        selected_data_tl = []
        selected_labels_tl = []
        #pour chaque classe
        for label in range(11):
            print(label)
        
            # Sélection des indices correspondant à ce label
            indices = np.where(labels == label)[0]
            if len(indices)>nbr+nbr_ssl : 
                print(f'il y a assez de {label}')
                indices = np.random.choice(indices, nbr+nbr_ssl, replace=False)
                indices_tl = indices[:nbr_ssl]
                indices_ul1 = indices[nbr_ssl:nbr_ssl+nbr//2]
                indices_ul2 = indices[nbr_ssl:]

                selected_data_tl.append(values[indices_tl])
                selected_labels_tl.append(labels[indices_tl])
                selected_data_ul1.append(values[indices_ul1])
                selected_labels_ul1.append(labels[indices_ul1])
                selected_data_ul2.append(values[indices])
                selected_labels_ul2.append(labels[indices])
            elif len(indices) == 0:
                print (f"il n\'y a pas de {label} dans le set")
            else:
                print(f'il n\'y a pas assez de {label} il y en a {len(indices)}')
                indices = np.random.choice(indices, len(indices), replace=False)
                if len(indices) >nbr_ssl :
                    indices_tl = indices[:nbr_ssl]
                    indices_ul1 = indices[nbr_ssl:(len(indices)+nbr_ssl)//2]
                    indices_ul2 = indices[nbr_ssl:]
                else:
                    indices_tl = indices[:len(indices)//4]
                    indices_ul1 = indices[len(indices)//4:len(indices)//2]
                    indices_ul2 = indices[len(indices)//4:]
                
                selected_data_tl.append(values[indices_tl])
                selected_labels_tl.append(labels[indices_tl])
                selected_data_ul1.append(values[indices_ul1])
                selected_labels_ul1.append(labels[indices_ul1])
                selected_data_ul2.append(values[indices_ul2])
                selected_labels_ul2.append(labels[indices_ul2])
        
        selected_data_tl = np.vstack(selected_data_tl)
        selected_labels_tl = np.hstack(selected_labels_tl)
        
        selected_data_ul1 = np.vstack(selected_data_ul1)
        selected_labels_ul1 = np.hstack(selected_labels_ul1)
        
        selected_data_ul2 = np.vstack(selected_data_ul2)
        selected_labels_ul2 = np.hstack(selected_labels_ul2)
        
        
        selection_finale_tl = {'X_SAR':selected_data_tl,'y':[[a,a] for a in selected_labels_tl], 'dates_SAR':dates}
        selection_finale_ul1 = {'X_SAR':selected_data_ul1,'y':[[-1,a] for a in selected_labels_ul1],'dates_SAR':dates}
        selection_finale_ul2 = {'X_SAR':selected_data_ul2,'y':[[-1,a] for a in selected_labels_ul2],'dates_SAR':dates}
        
        return selection_finale_ul1, selection_finale_ul2, selection_finale_tl # attention ici les données sont triées par classe
                
                
        
  
  
  

  

# data_laoding on va faire 3 data loader distincts pour pouvoir ajouter des données en cours de route

def data_loading(datas ,nbr_s=100000,nbr_t=100000): #data est de la forme [[jeux_sources],[jeux_targets]]
        list_values_train = []
        list_labels_train = []
        list_domain_train = []
        list_values_test = []
        list_labels_test = []
        list_values_train1 = []
        list_labels_train1 = []
        list_domain_train1 = []
        list_values_train2 = []
        list_labels_train2 = []
        list_domain_train2 =[]
        
        for i,data in enumerate (datas[0]):
            print(i)
            k=i
            data_train,_,_,dates,data_shape = tvt_split(data)
            data_train = selection(data_train, nbr_s,role="source")
            value_train,labels_train = data_train['X_SAR'],data_train['y']
            s= np.array(labels_train).shape
            
            list_values_train.append(value_train)
            list_labels_train.append(labels_train)
            list_domain_train.append(np.ones(s[0])*i)
        
    
        
        list_labels_train1 = list_labels_train.copy()
        list_labels_train2 = list_labels_train.copy()
        list_values_train1 = list_values_train.copy()
        list_values_train2 = list_values_train.copy()
        list_domain_train1 = list_domain_train.copy()
        list_domain_train2 = list_domain_train.copy()
        
        for data in datas[1] : # les données target sont ajoutées
            
            data_train,_,data_test,dates,data_shape = tvt_split(data)
            data_train1, data_train2, data_train_ssl = selection(data_train, nbr_t, role="target")
            value_train_ssl, labels_train_ssl = data_train_ssl['X_SAR'], data_train_ssl['y']
            value_train1,labels_train1 = data_train1['X_SAR'],data_train1['y']
            value_train2,labels_train2 = data_train2['X_SAR'],data_train2['y']
            value_test, labels_test = data_test['X_SAR'], data_test['y']
            
            
            s0 = np.array(labels_train_ssl).shape
            s1 = np.array(labels_train1).shape
            s2 = np.array(labels_train2).shape            
            
            
            list_values_train.append(value_train_ssl)
            list_labels_train.append(labels_train_ssl)
            list_domain_train.append(np.ones(s0[0])*(k+1))
            list_values_train1.append(value_train1)
            list_labels_train1.append(labels_train1)
            list_domain_train1.append(np.ones(s1[0])*(k+1))
            list_values_train2.append(value_train2)
            list_labels_train2.append(labels_train2)
            list_domain_train2.append(np.ones(s2[0])*(k+1))
            
            list_values_test.append(value_test)
            list_labels_test.append(labels_test)
        
               
        list_labels_train = np.concatenate(list_labels_train,axis=0)
        list_values_train = np.concatenate(list_values_train,axis=0)
        list_domain_train = np.concatenate(list_domain_train,axis=0)
        
        
        
        list_labels_train1 = np.concatenate(list_labels_train1,axis=0)
        list_values_train1 = np.concatenate(list_values_train1,axis=0)
        list_domain_train1 = np.concatenate(list_domain_train1,axis=0)
        
        list_labels_train2 = np.concatenate(list_labels_train2,axis=0)
        list_values_train2 = np.concatenate(list_values_train2,axis=0)
        list_domain_train2 = np.concatenate(list_domain_train2,axis=0)
        
        list_values_train = torch.tensor(list_values_train,dtype=torch.float32)
        list_labels_train = torch.tensor(list_labels_train, dtype=torch.int64)
        list_domain_train = torch.tensor(list_domain_train,dtype=torch.int64)
       
        
        
        list_values_train1 = torch.tensor(list_values_train1,dtype=torch.float32)
        list_labels_train1 = torch.tensor(list_labels_train1, dtype=torch.int64)
        list_domain_train1 = torch.tensor(list_domain_train1,dtype=torch.int64)
        
        
        
        
        list_values_train2 = torch.tensor(list_values_train2,dtype=torch.float32)
        list_labels_train2 = torch.tensor(list_labels_train2, dtype=torch.int64)
        list_domain_train2 = torch.tensor(list_domain_train2,dtype=torch.int64)
        
        list_values_test = np.concatenate(list_values_test, axis=0)
        list_labels_test = np.concatenate(list_labels_test, axis=0)
        
        list_values_test = torch.tensor(list_values_test, dtype=torch.float32)
        list_labels_test = torch.tensor(list_labels_test,dtype=torch.int64)
        
        print(list_values_train.shape)
        print(list_values_test.shape)
        
        

        train_dataset = TensorDataset(list_values_train, list_labels_train, list_domain_train)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64,drop_last=True)
        
        train_dataset1 = TensorDataset(list_values_train1, list_labels_train1, list_domain_train1)
        train_dataloader1 = DataLoader(train_dataset1, shuffle=True, batch_size=64,drop_last=True)
        
        train_dataset2 = TensorDataset(list_values_train2, list_labels_train2, list_domain_train2)
        train_dataloader2 = DataLoader(train_dataset2, shuffle=True, batch_size=64,drop_last=True)
        
        
        test_dataset = TensorDataset(list_values_test, list_labels_test)
        test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=64,drop_last=True)
        
        return train_dataloader, train_dataloader1, train_dataloader2, test_dataloader, dates, data_shape


def sup_contra_Cplus2_classes(emb, ohe_label, ohe_dom, scl, epoch):
    norm_emb = nn.functional.normalize(emb)
    C = ohe_label.max() + 1
    new_combined_label = [v1 if v2==6 else C+v2 for v1, v2 in zip(ohe_label, ohe_dom)]
    new_combined_label = torch.tensor(np.array(new_combined_label), dtype=torch.int64)
    return scl(norm_emb, new_combined_label, epoch=epoch)
# early stopping

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, checkpoint_path='best_model'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_path = checkpoint_path

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif -val_loss > -(self.best_score - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
            self.save_checkpoint(model)

        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.checkpoint_path)
        print("Saved new best model.")
    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class my_transformation:
    def __init__(self,p,liste_transformation,device): # liste_transformation contient 
        self.p=p
        self.l_transformation = liste_transformation
        self.device = device 
    def augment(self,x,mask):
        for transformation in self.l_transformation : 
            if torch.rand(1)<self.p :                 # on prend la transformation avec une proba p
                if isinstance(transformation, type(dropout(p=0.8))):
                    
                    x, mask = transformation.augment(x, mask)
                else :
                    
                    x = np.array(x.cpu())
                    x = transformation.augment(x)
                    x = torch.tensor(x).to(self.device)
            else : 
                pass
        return x, mask
                
                    
            
            
            
def all_elements_same(tensor):
    # Compare each element with the first element
    comparison_tensor = tensor == tensor[0]
    # Check if all values in the comparison_tensor are True
    return torch.all(comparison_tensor).item()       
        

        




