"""
Implementation of chess game.

@author: DiTurr

"""
import chess

from src.chess_engine.model import AlphaZeroModel
from src.chess_engine.board import ChessBoard


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
        print("============================================================")
        self.game.print(print_board_state=True)
        while True:
            # press input to continue
            print("[INFO] Press Enter to automatically calculate move or introduce move (UCI standard) to continue...")
            user_move = input()

            if user_move == "":
                print("[INFO] Calculating move ...")
                move = self.play_move()
                if move is None:
                    print("[ERROR] None of the by the chess engine calculated moves is legal ... ")
                    print("[INFO] Do you want to continue manually? [N/-]")
                    if input() == "N":
                        print("[INFO] GAME OVER !!")
                        break
                    else:
                        continue
                else:
                    self.func_emit(move)
            else:
                if self.game.chess_board.is_legal(chess.Move.from_uci(user_move)):
                    self.game.make_move(user_move)
                    print("============================================================")
                    self.game.print(print_board_state=True)
                    self.func_emit(user_move)
                else:
                    print("[ERROR] Requested move not legal")

    def play_move(self):
        """


        """
        # predict best moves
        serialized_board = self.game.serialize_board()
        y_hat_winner, y_hat_policy = self.alphazero_model.predict(serialized_board)
        serialized_moves, probability_moves = self.alphazero_model.get_best_moves(y_hat_policy, num_moves=5)

        # are calculated moves legal?
        moves, flag_validity_moves = self.game.is_serialized_move_legal(serialized_moves)
        self.game.print(moves=moves, flag_validity_moves=flag_validity_moves, probability_moves=probability_moves)

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
            print("============================================================")
            self.game.print(print_board_state=True)
            return move_selected
        else:
            return None