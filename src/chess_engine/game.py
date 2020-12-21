"""
Implementation of chess game.

@author: DiTurr

"""
import chess

from .model import AlphaZeroModel
from .board import ChessBoard
import torch.optim as optim


class ChessGame:
    def __init__(self, path_load_model, func_emit=None):
        """


        """
        print("[INFO] Generating and loading model ...")
        self.alphazero_model = AlphaZeroModel()
        self.alphazero_model.load_model(path_load_model)
        self.game = ChessBoard()
        if func_emit is None:
            self.func_emit = None
        else:
            self.func_emit = func_emit

    def play_game(self):
        """


        """
        # loop through the game
        print("===========================================================================================")
        self.game.print(print_board_state=True)
        while True:
            # press input to continue
            print("[INFO] Press Enter to use ChessEngine or introduce move ...")
            user_move = input()

            # move calculated be engine chess
            if user_move == "":
                move = self.play_move()
                if move is not None:
                    print("===========================================================================================")
                    self.game.print(print_board_state=True)
                    self.func_emit(move)
                else:
                    print("[ERROR] No move calculated by ChessEngine is valid ... ")
                    print("[INFO] Do you want to continue manually? [N/[Y]]")
                    if input() == "N":
                        print("[INFO] GAME-OVER!!")
                        break
                    else:
                        continue

            # user move
            else:
                if self.game.chess_board.is_legal(chess.Move.from_uci(user_move)):
                    self.game.make_move(user_move)
                    print("===========================================================================================")
                    self.game.print(print_board_state=True)
                    self.func_emit(user_move)
                else:
                    print("[ERROR] Requested move not legal")

            # check status of the match
            if self.game.chess_board.is_check():
                print("[INFO] Check!!")
            if self.game.chess_board.is_checkmate():
                print("[INFO] Checkmate!! GAME-OVER!!")
                break
            if self.game.chess_board.is_stalemate():
                print("[INFO] Stalemate!! GAME-OVER!!")
                break
            if self.game.chess_board.is_insufficient_material():
                print("[INFO] No sufficient material to do checkmate!! DRAW!!")
                break
            if self.game.chess_board.is_seventyfive_moves():
                print("[INFO] Seventyfive-moves rule!! DRAW!!")
                break

    def play_move(self):
        """


        """
        # predict best moves
        serialized_board = self.game.serialize_board()
        y_hat_winner, y_hat_policy = self.alphazero_model.predict(serialized_board)
        serialized_moves, probability_moves = self.alphazero_model.get_best_moves(y_hat_policy, num_moves=1000)

        # are calculated moves legal?
        moves, flag_validity_moves = self.game.is_serialized_move_legal(serialized_moves)

        # choose move from the different possibilities
        # serialized_moves: list of moves (serialized)
        # moves: list of moves accoding to UCI standard
        # flag_validity_moves: are moves posible with the actual board state
        # probability_moves: move's probability
        move_selected, probability_move_selected = self.game.select_best_move(moves,
                                                                              flag_validity_moves,
                                                                              probability_moves)

        # make move if possible
        if move_selected is not None:
            self.game.make_move(move_selected)
            self.game.print(moves=moves, flag_validity_moves=flag_validity_moves,
                            probability_moves=probability_moves, num_moves_print=5,
                            probability_winner=y_hat_winner,
                            move_selected=move_selected, probability_move_selected=probability_move_selected)
            return move_selected
        else:
            self.game.print(moves=moves, flag_validity_moves=flag_validity_moves,
                            probability_moves=probability_moves, num_moves_print=5,
                            probability_winner=y_hat_winner)
            return None
