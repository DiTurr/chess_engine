"""
Preprocess PGN database.

@author: DiTurr

"""
from src.chess_engine import PGNDatabase

if __name__ == "__main__":
    PGNDatabase.preprocess_pgn_files(path_pgn_files="/home/ditu/Documents/03_Projects/chess_engine/database",
                                     num_moves_database=1000,
                                     path_save_csv_database="/home/ditu/Documents/03_Projects/chess_engine/database",
                                     train_val_split=0.8)
