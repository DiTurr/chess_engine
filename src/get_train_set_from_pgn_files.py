"""
Preprocess PGN database.

@author: DiTurr

"""
from .chess_engine import PGNDatabase

if __name__ == "__main__":
    PGNDatabase.preprocess_pgn_files(path_pgn_files="/home/ditu/Documents/03_Projects/01_ChessEngine/database",
                                     num_moves_database=10000,
                                     path_save_csv_database="/home/ditu/Documents/03_Projects/01_ChessEngine/database",
                                     train_val_split=0.8)
