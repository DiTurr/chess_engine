"""
Preprocess PGN database.

@author: DiTurr

"""
import torch
from torch.utils.data import DataLoader  # NOQA
from torchsummary import summary  # NOQA
import torch.optim as optim

from src.chess_engine.database import PGNDatabase
from src.chess_engine.model import AlphaZeroNet, AlphaLoss, train_model

if __name__ == "__main__":
    print("[INFO] generating dataloader")
    # training dataloader
    chess_train_dataset = PGNDatabase(path_csv_database="/home/ditu/Documents/03_Projects/chess_engine/database"
                                                        "/chess_train_database.csv")
    dataloader_train = DataLoader(chess_train_dataset,
                                  batch_size=300, shuffle=True,
                                  num_workers=1, drop_last=True)
    # validation dataloader
    chess_val_dataset = PGNDatabase(path_csv_database="/home/ditu/Documents/03_Projects/chess_engine/database"
                                                      "/chess_val_database.csv")
    dataloader_val = DataLoader(chess_val_dataset,
                                batch_size=300, shuffle=False,
                                num_workers=1, drop_last=True)

    print("[INFO] generating model")
    model = AlphaZeroNet()
    if torch.cuda.is_available():
        print("[INFO] model moved to the GPU")
        model.cuda()
    # summary(model, (19, 8, 8))  # input to the model is batch_size, channels, height and weight

    print("[INFO] setting up function loss and optimizer")
    loss_function = AlphaLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print("[INFO] training the model ...")
    train_model(model=model,
                max_epochs=100,
                training_generator=dataloader_train,
                validation_generator=dataloader_val,
                loss_function=loss_function,
                optimizer=optimizer)
