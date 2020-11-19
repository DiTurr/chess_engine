"""
Preprocess PGN database.

@author: DiTurr

"""
import chess.pgn
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset  # NOQA
import random
import os

from .board import ChessBoard


class PGNDatabase(Dataset):
    def __init__(self, path_csv_database=None):
        """
        Class to preprocessed PGN files.

        Parameters
        ----------
        path_csv_database: str
            Path to the csv file containing the trainning data.

        Returns
        -------

        """
        self.df = pd.read_csv(path_csv_database)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx, channel_first=True):
        """
        Preprocess PGN files to get preprocessed board stated, mve and game winner.

        Parameters
        ----------
        idx: list or tensor
            Index inside the batch to picked out of self.df and preprocess it.
        channel_first: bool
            If true channel first numpy array are returned, otherwise channel last.

        Returns
        -------
        board_serialized: numpy array
            Serialized board.
        move_serialized: numpy array
            Serialized move.
        game_winner: numpy array
            Winner of the game.

        """
        # read row from pandas dataframe
        board_state = self.df.iloc[idx]["board_state"]
        move_uci = self.df.iloc[idx]["move"]
        game_winner = self.df.iloc[idx]["game_winner"]

        # create chess board with the information
        board = ChessBoard(chess.Board(board_state))

        # serialized the board
        board_serialized = board.serialize_board()

        # serialized the move
        move_serialized = board.serialize_move(move_uci)

        # if needed change the shape of the numpy array to channel first
        if channel_first:
            board_serialized = np.rollaxis(board_serialized, 2, 0)

        # reshape and check dtype before return
        board_serialized = board_serialized.astype(np.float32)
        move_serialized = move_serialized.reshape((-1,)).astype(np.float32)
        game_winner = game_winner.reshape((-1,)).astype(np.float32)

        # return serialized board, serialized move and winner as numpy arrays
        return board_serialized, move_serialized, game_winner

    @staticmethod
    def preprocess_pgn_files(path_pgn_files, num_moves_database, train_val_split, path_save_csv_database):
        """
        Preprocess PGN files to get board stated, move and game winner as a pandas dataframe, and save it
        as csv file.

        Parameters
        ----------
        path_pgn_files: str
            Path (folder), where the PGN files can be found.
        num_moves_database: int
            Number of moves to preprocessed before returning the pandas dataframe.
        train_val_split: float
            Split between training and validation dataset
        path_save_csv_database: str
            Path where csv file should be saved.

        Returns
        -------
        retun: bool
            "1" is loop has finished due to limit give by the user or "0" if all available data have been preprocessed.

        """
        # create empty pandas dataframe to save the information
        df_train = pd.DataFrame({"board_state": pd.Series([], dtype='str'),
                                 "move": pd.Series([], dtype='str'),
                                 "game_winner": pd.Series([], dtype='int')})
        df_val = pd.DataFrame({"board_state": pd.Series([], dtype='str'),
                               "move": pd.Series([], dtype='str'),
                               "game_winner": pd.Series([], dtype='int')})

        # create counter for total number of moves
        counter_samples = 0
        pbar = tqdm(total=num_moves_database, ascii=True)

        # find and iterate over all PGN files
        pgn_files = glob.glob(path_pgn_files + "/*.pgn")
        for path_pgn_file in pgn_files:
            pgn_file = open(path_pgn_file, encoding="ISO-8859-1")
            while True:
                game = chess.pgn.read_game(pgn_file)
                # no more games in the PGN file
                if game is None:
                    break

                # iterate through all moves and play them on a board.
                game_winner = {"0-1": -1, "1-0": 1, "1/2-1/2": 0}[game.headers["Result"]]
                board = game.board()
                for move in game.main_line():
                    # get board state
                    board_state = board.board_fen()

                    # get move corresponding to this state as UCI standard
                    move_uci = move.uci()

                    # update board state
                    board.push(move)

                    # append information to pandas dataframe
                    if random.uniform(0, 1) < train_val_split:
                        df_train = df_train.append({"board_state": board_state,
                                                    "move": move_uci,
                                                    "game_winner": game_winner}, ignore_index=True)
                    else:
                        df_val = df_val.append({"board_state": board_state,
                                                "move": move_uci,
                                                "game_winner": game_winner}, ignore_index=True)

                    # update move counter and progress bar
                    counter_samples += 1
                    pbar.update()
                    if num_moves_database is not None and counter_samples >= num_moves_database:
                        # save pandas dataframe as dataframe
                        df_train = df_train.sample(frac=1).reset_index(drop=True)
                        df_val = df_val.sample(frac=1).reset_index(drop=True)
                        df_train.to_csv(os.path.join(path_save_csv_database, "chess_train_database.csv"), index=False)
                        df_val.to_csv(os.path.join(path_save_csv_database, "chess_val_database.csv"), index=False)
                        return 1

        # save pandas dataframe as dataframe
        df_train = df_train.sample(frac=1).reset_index(drop=True)
        df_val = df_val.sample(frac=1).reset_index(drop=True)
        df_train.to_csv(os.path.join(path_save_csv_database, "chess_train_database.csv"), index=False)
        df_val.to_csv(os.path.join(path_save_csv_database, "chess_val_database.csv"), index=False)
        return 0
