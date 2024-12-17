from data_handler import BiradsDataSet
import numpy as np
import optuna
import os
from optuna.trial import TrialState
import torch
import sys
sys.path.insert(0,"/kaggle/working/Multimodal_Cancer_Detection_Model/Transformer-Explainability")
from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from baselines.ViT.ViT_explanation_generator import LRP
from torchmetrics import F1Score,Recall,Precision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from loss_functions import FocalLoss
import torch 

from torchvision.io import ImageReadMode
from torch.utils.data import DataLoader 
from torchvision import models

data_dir = "/kaggle/working/content/Dataset"
modality="Mammogram"
num_classes=7
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
Train_Ds =  BiradsDataSet(data_dir,modality)
Val_Ds =  BiradsDataSet(data_dir,modality)
batch_sz = 8
dataloaders = {}
datasetsizes = {}
criterion = FocalLoss()
dataloaders["train"] = DataLoader(Train_Ds,batch_sz,shuffle=True,num_workers=12)
dataloaders["val"] = DataLoader(Val_Ds,batch_sz,num_workers=12)
datasetsizes["train"] = len(Train_Ds)
datasetsizes["val"] = len(Val_Ds)
num_epochs=5
best_acc=0
def objective(trial):
    global best_acc
    optimizer_name = "RMSprop"
    lr = trial.suggest_float("lr", 1e-7, 1e-5, log=True)
    lr2 = trial.suggest_float("lr2", 1e-6, 1e-5, log=True)
    # model_name = trial.suggest_categorical("model_name",["vit","swinb","max_vit"])
    model_name = "vit"
    if(model_name=="vgg16"):
        model = models.vgg16(pretrained=True)
        model2 = models.vgg16(pretrained=True)
    elif(model_name=="swinb"):
        model = models.swin_v2_b(pretrained=True)
        model2 = models.swin_v2_b(pretrained=True)
    elif(model_name=="vit"):
        #model = models.vit_b_16(pretrained=True)
        #model2 = models.vit_b_16(pretrained=True)
        model = vit_LRP(pretrained=True)
        if(modality=="Multimodal"):
            model2 = vit_LRP(pretrained=True)
    elif(model_name=="convnext"):
        model = models.convnext.convnext_base(pretrained=True)
        model2 = models.convnext.convnext_base(pretrained=True)
    else:
        model = models.maxvit_t(pretrained=True)
        model2 = models.maxvit_t(pretrained=True)
    dropout_rate = 0.18
    model.to(device)
    if(modality=="Multimodal"):
        model2.to(device)

    if(modality=="Multimodal"):
        classification_layer = nn.Sequential(nn.Dropout(dropout_rate),nn.Linear(2000,2000),nn.GELU(),nn.Dropout(dropout_rate),nn.Linear(2000,num_classes))
    else:
        classification_layer = nn.Sequential(nn.Dropout(dropout_rate),nn.Linear(1000,1000),nn.GELU(),nn.Dropout(dropout_rate),nn.Linear(1000,num_classes))
    classification_layer.to(device)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    if(modality=="Multimodal"):
        optimizer2 = getattr(optim, optimizer_name)(model2.parameters(), lr=lr)
    optimizer_class =  getattr(optim, optimizer_name)(classification_layer.parameters(), lr=lr2)
    path = "/kaggle/working/checkpoint_path"
    best_model_params_path = os.path.join(path,"best_model_params_"+model_name+"_Mammo" + modality + ".pt")
    best_model_params_path3 = os.path.join(path,"best_model_params_"+model_name+"_Ultra" + modality + ".pt")
    best_model_params_path2 = os.path.join(path,"best_model_params_"+model_name+"_classifier_" + modality + ".pt")
    with open("logs"+"_"+"model_name","a") as f:

            f.write("model name:" + model_name + "\n")
            print("model name:" + model_name + "\n")
            f.write("Modality:" + modality + "\n")
            print("Modality:" + modality + "\n")

            Calc_F1 = F1Score(task="multiclass",num_classes=num_classes,average="macro")
            Calc_Prec = Precision(task="multiclass",num_classes=num_classes,average="macro")
            Calc_Recall = Recall(task="multiclass",num_classes=num_classes,average="macro")
            for epoch in range(num_epochs):
                f.write(f'Epoch {epoch}/{num_epochs - 1}\n')
                print(f'Epoch {epoch}/{num_epochs - 1}\n')
                f.write('-' * 10 + "\n")
                print('-' * 10 + "\n")
                     # Each epoch has a training and validation phase
                for phase in ['train', 'val']: 
                    val_preds = []
                    val_labels = []
                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.

                    for inputs, inputs2, labels in dataloaders[phase]:

                                inputs = inputs.to(device)
                                inputs2 = inputs2.to(device)
                                labels = labels.to(device)
                                # zero the parameter gradients
                                optimizer.zero_grad()
                                optimizer_class.zero_grad()
                                if(modality=="Multimodal"):
                                    optimizer2.zero_grad()

                                # forward
                                # track history if only in train

                                features = model(inputs)
                                if(modality=="Multimodal"):
                                    features2 = model2(inputs2)
                                    features = torch.cat([features,features2],dim=1)
                                with torch.set_grad_enabled(phase == 'train'):
                                    outputs = classification_layer(features)
                                    _, preds = torch.max(outputs, 1)
                                    loss = criterion(outputs, labels)

                                    # backward + optimize only if in training phase
                                    if phase == 'train':
                                        loss.backward()
                                        optimizer.step()
                                        if(modality=="Multimodal"):
                                            optimizer2.step()
                                        optimizer_class.step()
                                    val_preds.append(preds.cpu())
                                    val_labels.append(labels.cpu())
                    # statistics
                                running_loss += loss.item() * inputs.size(0)
                                running_corrects += torch.sum(preds == labels.data)
                    val_preds = np.concatenate(val_preds)
                    val_labels = np.concatenate(val_labels)
                    val_preds = torch.from_numpy(val_preds)
                    val_labels = torch.from_numpy(val_labels)
                    f1 = Calc_F1(val_preds,val_labels)
                    precision = Calc_Prec(val_preds,val_labels)
                    recall = Calc_Recall(val_preds,val_labels)
                    epoch_loss = running_loss / datasetsizes[phase]
                    epoch_acc = running_corrects.double() / datasetsizes[phase]

                    f.write(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')
                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

                    f.write(f'{phase} F1_Score: {f1} Precision : {precision} Recall : {recall}\n')
                    print(f'{phase} F1_Score: {f1} Precision : {precision} Recall : {recall}\n')
                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                    
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), best_model_params_path3)
                        print("saved ultra model")
                        torch.save(model2.state_dict(),best_model_params_path)
                        print("saved mammo model")
                        torch.save(classification_layer,best_model_params_path2)
                        print("saved classifier")
                    if phase=='val':
                        trial.report(epoch_acc, epoch)

        # Handle pruning based on the intermediate value.
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                f.write('----------\n')
                print('----------\n')
            f.write(f'Best val Acc: {best_acc:4f}\n')
            print(f'Best val Acc: {best_acc:4f}\n')
                # load best model weights
            return epoch_acc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
