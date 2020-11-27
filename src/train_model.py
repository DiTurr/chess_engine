"""
Preprocess PGN database.

@author: DiTurr

"""
from torch.utils.data import DataLoader  # NOQA
import torch.optim as optim

from src.chess_engine.database import PGNDatabase
from src.chess_engine.model import AlphaZeroModel, AlphaLoss

if __name__ == "__main__":
    print("[INFO] generating dataloaders ...")
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

    print("[INFO] generating model ...")
    alphazero_model = AlphaZeroModel()

    print("[INFO] training model ...")
    alphazero_model.train(epochs=2,
                          training_generator=dataloader_train,
                          validation_generator=dataloader_val,
                          loss_function=AlphaLoss(),
                          optimizer=optim.SGD(alphazero_model.model.parameters(), lr=0.001, momentum=0.9))

    print("[INFO] saving model ...")
    alphazero_model.save_model("/home/ditu/Documents/03_Projects/chess_engine/models/model.pth")

    print("[INFO] ploting training history ...")
    alphazero_model.plot_history()
