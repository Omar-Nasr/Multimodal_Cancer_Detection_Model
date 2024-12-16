from torchvision import models
import torch
import optuna
from optuna.trial import TrialState
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader 
from data_handler import BiradsDataSet
import torch.optim as optim
from loss_functions import FocalLoss
import torch.nn as nn
num_epochs = 20
data_dir = "/home/omar/Programming/Grad_Project_Again/UltraSoundWithMammo/DataSet/Dataset_Extracted/"
modality="UltraSound"
Train_Ds =  BiradsDataSet(data_dir,modality,category="Train")
Val_Ds =  BiradsDataSet(data_dir,modality,category="Val")
batch_sz = 1
dataloaders = {}
datasetsizes = {}
num_classes=7
dataloaders["train"] = DataLoader(Train_Ds,batch_sz,shuffle=True)
dataloaders["val"] = DataLoader(Val_Ds,batch_sz)
datasetsizes["train"] = len(Train_Ds)
datasetsizes["val"] = len(Val_Ds)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = FocalLoss()
def objective(trial):

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    model_name = trial.suggest_categorical("model", ["swinb", "vit", "convnext"])

    classification_layer = nn.Sequential(nn.Dropout(0.4),nn.Linear(1000,1000),nn.GELU(),nn.Dropout(0.4),nn.Linear(1000,num_classes))
    lr = trial.suggest_float("lr",1e-5,1e-1)
    lr2 = trial.suggest_float("lr",1e-9,1e-4)
    if(model_name=="vgg16"):
        model = models.vgg16(pretrained=True)
    elif(model_name=="swinb"):
        model = models.swin_v2_b(pretrained=True)
    elif(model_name=="vit"):
        model = models.vit_b_16(pretrained=True)
    elif(model_name=="convnext"):
        model = models.convnext.convnext_base(pretrained=True)
    else:
        model = models.maxvit_t(pretrained=True)


    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    optimizer_class = getattr(optim, optimizer_name)(classification_layer.parameters(), lr=lr2)
    scheduler = lr_scheduler.ConstantLR(optimizer)#,step_size=1400,gamma=0.1)
    scheduler2 = lr_scheduler.ConstantLR(optimizer_class)
    for epoch in range(num_epochs):

   # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                optimizer_class.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    features = model(inputs)
                    outputs = classification_layer(features)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        optimizer_class.step()

                # statistics

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
                scheduler2.step()


            epoch_loss = running_loss / datasetsizes[phase]
            epoch_acc = running_corrects.double() / datasetsizes[phase]

            if(phase=='val'):
                trial.report(epoch_acc, epoch)

            # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            # # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #
        # load best model weights
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, timeout=600)

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
