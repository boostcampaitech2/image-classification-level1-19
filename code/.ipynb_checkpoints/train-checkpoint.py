import torch
import numpy as np
from trainer import Trainer
from dataloaders import MaskDataLoader
from model import MaskModel
import torch.optim as optim

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)

def main():
    

    # setup data_loader instances
    loader = MaskDataLoader()
    data_loader = loader.data_loader 
    test_data_loader = loader.test_data_loader

    # build model architecture, then print to console
    model = MaskModel()
 
    # prepare for (multi-device) GPU training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 학습 때 GPU 사용여부 결정. Colab에서는 "런타임"->"런타임 유형 변경"에서 "GPU"를 선택할 수 있음

    model = model.to(device)
 

    # get function handles of loss and metrics
    criterion = torch.nn.CrossEntropyLoss()
  

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # weight 업데이트를 위한 optimizer를 Adam으로 사용함
    LEARNING_RATE = 0.0005
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    
    scheduler =  optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    trainer = Trainer(model, criterion, optimizer, device,
                 data_loader, LEARNING_RATE, test_data_loader, scheduler)
    trainer.train(NUM_EPOCH=30)


if __name__ == '__main__':
   
    main()