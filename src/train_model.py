"""
Preprocess PGN database.

@author: DiTurr

"""
from torch.utils.data import DataLoader  # NOQA
import torch.optim as optim

from chess_engine.database import PGNDatabase  # NOQA
from chess_engine.model import AlphaZeroModel, AlphaLoss  # NOQA

PATH_LOAD_MODEL = "/home/ditu/Documents/03_Projects/chess_engine/models/model.pth"

if __name__ == "__main__":
    print("[INFO] Generating dataloaders ...")
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

    print("[INFO] Generating model ...")
    alphazero_model = AlphaZeroModel()

    if PATH_LOAD_MODEL == "":
        print("[INFO] Training model ...")
        alphazero_model.train(epochs=2,
                              training_generator=dataloader_train,
                              validation_generator=dataloader_val,
                              loss_function=AlphaLoss(),
                              optimizer=optim.SGD(alphazero_model.model.parameters(), lr=0.001, momentum=0.9))
    else:
        print("[INFO] Retraining model ...")
        alphazero_model.retrain(PATH_LOAD_MODEL, epochs=2)

    print("[INFO] saving model ...")
    alphazero_model.save_model("/home/ditu/Documents/03_Projects/chess_engine/models/model.pth", save_generator=True)

    print("[INFO] ploting training history ...")
    alphazero_model.plot_history()
