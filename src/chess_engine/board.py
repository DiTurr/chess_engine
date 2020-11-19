"""
Implementation of chess board needed for chess engine

@author: DiTurr

"""
import chess
import chess.pgn
import numpy as np


class ChessBoard:
    def __init__(self, chess_board=None):
        """
        Class chess board.

        Parameters
        ----------
        chess_board: chess.Board or None
            Chess board to initialized the game.

        Returns
        -------

        """
        if chess_board is None:
            self.chess_board = chess.Board()
        else:
            self.chess_board = chess_board

    def make_move(self, move):
        """
        Update the board according to the given move.

        Parameters
        ----------
        move: Move.from_uci
            Move to me made.

        Returns
        -------

        """
        self.chess_board.push(move)

    def serialize_move(self, move):
        """
        We represent the policy π(a|s) by a 8×8×73 stack of planes encoding a probability distribution over 4,672
        possible moves. Each of the8×8positions identifies the square from which to “pick up” a piece.
        The first 56 planes encodepossible ‘queen moves’ for any piece:  a number of squares [1..7] in which
        the piece will be moved, along one of eight relative compass directions {N,NE,E,SE,S,SW,W,NW}.
        The next 8 planes encode possible knight moves for that piece.  The final 9 planes encode possible
        underpromotions for pawn moves or captures in two possible diagonals, to knight, bishop orrook  respectively.
        Other  pawn  moves  or  captures  from  the  seventh  rank  are  promoted  to  aqueen.

        Parameters
        ----------
        move: str
             Move according UCI rules. For example, "d2d4".

        Returns
        -------
        move_serialized: np.array
            Serialized moved as an [8x8x73] numpy array.

        """
        # calculate initial and final position. x and y are 0 based.
        x_initial, y_initial, x_final, y_final, promotion = self.get_positions_from_move(move)
        # calculate the relative movement
        dx = x_final - x_initial
        dy = y_final - y_initial
        # get piece
        piece = self.get_piece_at_position(x_initial, y_initial)
        # calculate encoding
        move_serialized = np.zeros([8, 8, 73]).astype(np.uint8)
        plane = None
        if piece in ["R", "B", "Q", "K", "P", "r", "b", "q", "k", "p"] and (promotion is None or promotion == "queen"):
            # serialize queen-like moves [0 ... 55]
            if dx == 0 and dy != 0:
                if dy > 0:
                    # north [0 ... 6]
                    plane = -1 + dy
                elif dy < 0:
                    # south [7 ... 13]
                    plane = 6 + -dy
            elif dx != 0 and dy == 0:
                if dx > 0:
                    # east [14 ... 20]
                    plane = 13 + dx
                elif dx < 0:
                    # west [21 ... 27]
                    plane = 20 + -dx
            elif dx == dy:
                if dx > 0:
                    # north-east [28 ... 34]
                    plane = 27 + dx
                elif dx < 0:
                    # south-west [35 ... 41]
                    plane = 34 + -dx
            elif dx == -dy:
                if dx > 0:
                    # north-west [42 ... 48]
                    plane = 41 + dx
                elif dx < 0:
                    # south-east [49 ... 55]
                    plane = 48 + -dx
        elif piece in ["n", "N"]:
            # serialize knight-like moves [56 ... 63]
            if (dy == 2) and (dx == 1):
                # north-north-east [56]
                plane = 56
            elif (dy == 2) and (dx == -1):
                # north-north-west [57]
                plane = 57
            elif (dy == 1) and (dx == 2):
                # north-east-east [58]
                plane = 58
            elif (dy == 1) and (dx == -2):
                # north-west-west [59]
                plane = 59
            elif (dy == -2) and (dx == 1):
                # south-south-east [60]
                plane = 60
            elif (dy == -2) and (dx == -1):
                # south-south-west [61]
                plane = 61
            elif (dy == -1) and (dx == 2):
                # south-east-east [62]
                plane = 62
            elif (dy == -1) and (dx == -2):
                # south-west-west [63]
                plane = 63
        elif piece in ["p", "P"] and (y_final == 0 or y_final == 7) and promotion is not None:
            # underpromotions [64 ... 72]
            if abs(dy) == 1 and dx == 0:
                if promotion == "rook":
                    # rook underpromotions with north/south move [64]
                    plane = 64
                if promotion == "knight":
                    # knight underpromotions with north/south move [65]
                    plane = 65
                if promotion == "bishop":
                    # bishop underpromotions with north/south move [66]
                    plane = 66
            elif abs(dy) == 1 and dx == -1:
                if promotion == "rook":
                    # rook underpromotions with north-west/south-west move [67]
                    plane = 67
                if promotion == "knight":
                    # knight underpromotions with north-west/south-west move [68]
                    plane = 68
                if promotion == "bishop":
                    # bishop underpromotions with north-west/south-west move [69]
                    plane = 69
            if abs(dy) == 1 and dx == 1:
                if promotion == "rook":
                    # rook underpromotions with north-east/south-east move [70]
                    plane = 70
                if promotion == "knight":
                    # knight underpromotions with north-east/south-east move [71]
                    plane = 71
                if promotion == "bishop":
                    # bishop underpromotions with north-east/south-east move [72]
                    plane = 72

        # copy information into np array
        assert plane is not None, "[ERROR] " + str(move) + " not supported"
        move_serialized[x_initial, y_initial, plane] = 1
        return move_serialized

    def deserialize_move(self, move_serialized):
        """
        Calculate the move according to UCI standards give the serialized movec [8x8x19].

        Parameters
        ----------
        move_serialized: np.array
            Serialized moved as an [8x8x73] numpy array.

        Returns
        -------
        move: str
             Move according UCI rules. For example, "d2d4".

        """
        assert np.sum(move_serialized) == 1
        # x and y are 0 based
        x_initial, y_initial, plane = np.where(move_serialized == 1)
        x_initial, y_initial, plane = x_initial[0], y_initial[0], plane[0]
        dx, dy, promotion = None, None, ""
        if 0 <= plane <= 55:
            # queen - like moves[0 ... 55]
            if 0 <= plane <= 6:
                # north [0 ... 6]
                dx = 0
                dy = plane + 1
            elif 7 <= plane <= 13:
                # south [7 ... 13]
                dx = 0
                dy = 6 - plane
            elif 14 <= plane <= 20:
                # east [14 ... 20]
                dx = plane - 13
                dy = 0
            elif 21 <= plane <= 27:
                # west [21 ... 27]
                dx = 20 - plane
                dy = 0
            elif 28 <= plane <= 34:
                # north-east [28 ... 34]
                dx = plane - 27
                dy = plane - 27
            elif 35 <= plane <= 41:
                # south-west [35 ... 41]
                dx = 34 - plane
                dy = 34 - plane
            elif 42 <= plane <= 48:
                # north-west [42 ... 48]
                dx = plane - 41
                dy = -dx
            elif 49 <= plane <= 55:
                # south-east [49 ... 55]
                dx = 48 - plane
                dy = -dx
            piece = self.get_piece_at_position(x_initial, y_initial)
            if (piece == "p" and dy == -1 and y_initial == 1) or (piece == "P" and dy == 1 and y_initial == 6):
                # if move done by p/P and got to the last row, then promotion to queen
                promotion = "queen"
        elif 56 <= plane <= 63:
            # serialize knight - like moves[56 ... 63]
            if plane == 56:
                # north-north-east [56]
                dx = 1
                dy = 2
            elif plane == 57:
                # north-north-west [57]
                dx = -1
                dy = 2
            elif plane == 58:
                # north-east-east [58]
                dx = 2
                dy = 1
            elif plane == 59:
                # north-west-west [59]
                dx = -2
                dy = 1
            elif plane == 60:
                # south-south-east [60]
                dx = 1
                dy = -2
            elif plane == 61:
                # south-south-west [61]
                dx = -1
                dy = -2
            elif plane == 62:
                # south-east-east [62]
                dx = 2
                dy = -1
            elif plane == 63:
                # south-west-west [63]
                dx = -2
                dy = -1
        else:
            # underpromotions [64 ... 72]
            if plane == 64:
                # rook underpromotions with north/south move [64]
                promotion = "rook"
                dx = 0
            elif plane == 65:
                # knight underpromotions with north/south move [65]
                promotion = "knight"
                dx = 0
            elif plane == 66:
                # bishop underpromotions with north/south move [66]
                promotion = "bishop"
                dx = 0
            elif plane == 67:
                # rook underpromotions with north-west/south-west move [67]
                promotion = "rook"
                dx = -1
            elif plane == 68:
                # knight underpromotions with north-west/south-west move [68]
                promotion = "knight"
                dx = -1
            elif plane == 69:
                # bishop underpromotions with north-west/south-west move [69]
                promotion = "bishop"
                dx = -1
            elif plane == 70:
                # rook underpromotions with north-east/south-east move [70]
                promotion = "rook"
                dx = 1
            elif plane == 71:
                # knight underpromotions with north-east/south-east move [71]
                promotion = "knight"
                dx = 1
            elif plane == 72:
                # bishop underpromotions with north-east/south-east move [72]
                promotion = "bishop"
                dx = 1
            if y_initial == 1:
                dy = -1
            else:
                assert y_initial == 6
                dy = 1
        # assert dx and dy not None
        assert dx is not None
        assert dy is not None
        # calculate the final position
        x_final = dx + x_initial
        y_final = dy + y_initial
        x_initial = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h"}[x_initial]
        x_final = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h"}[x_final]
        promotion = {"queen": "q", "rook": "r", "knight": "n", "bishop": "b", "": ""}[promotion]
        move = x_initial + str(y_initial + 1) + x_final + str(y_final + 1) + promotion
        return move

    def serialize_board(self):
        """
        The input to the neural network is a [8x8x19] array that represents state.
        [0 ... 5]: position of white pieces.
        [6 ... 11]: positions of black pieces.
        [12]: turn (white is reresented by 1).
        [13]: white has rigths to queenside castling
        [14]: white has rigths to kingside castling
        [15]: black has rigths to queenside castling
        [16]: black has rigths to kingside castling
        [17]: total move counter
        [18]: no progress counter
        TODO [19]: repetitions white
        TODO [20]: repetetions black

        Parameters
        ----------

        Returns
        -------

        """
        serialized_board = np.zeros([8, 8, 19]).astype(np.uint8)
        piece_dict = {"R": 0, "N": 1, "B": 2, "Q": 3, "K": 4, "P": 5,
                      "r": 6, "n": 7, "b": 8, "q": 9, "k": 10, "p": 11}
        for index_row in range(8):
            for index_column in range(8):
                # pieces [0 ... 11]
                piece = self.get_piece_at_position(index_column, index_row)
                if piece != "":
                    serialized_board[index_row, index_column, piece_dict[piece]] = 1
        if self.chess_board.turn == chess.WHITE:
            # player to move [12]
            serialized_board[:, :, 12] = 1
        if self.chess_board.has_queenside_castling_rights(chess.WHITE):
            # white has rigths to queenside castling [13]
            serialized_board[:, :, 13] = 1
        if self.chess_board.has_kingside_castling_rights(chess.WHITE):
            # white has rigths to kingside castling [14]
            serialized_board[:, :, 14] = 1
        if self.chess_board.has_queenside_castling_rights(chess.BLACK):
            # black has rigths to queenside castling [15]
            serialized_board[:, :, 15] = 1
        if self.chess_board.has_kingside_castling_rights(chess.BLACK):
            # black has rigths to kingside castling [16]
            serialized_board[:, :, 16] = 1
        # total mvoe counter
        serialized_board[:, :, 17] = self.chess_board.fullmove_number
        # no progress counter
        serialized_board[:, :, 18] = self.chess_board.halfmove_clock
        # return serialized_board
        return serialized_board

    def get_piece_at_position(self, x, y):
        """
        Get the piece give the posicion.

        Parameters
        ----------
        x: int
            Chess board column (0-base).
        y: int
            Chess board row (0-base).

        Returns
        -------
        piece: str
            Piece symbol as str ()

        """
        position = y * 8 + x
        piece = self.chess_board.piece_at(position)
        if piece is None:
            return ""
        else:
            return piece.symbol()

    @staticmethod
    def get_positions_from_move(move):
        """
        Calculates the initial position, final position and promotion (if it is the case) given a move according to
        UCI standard.

        Parameters
        ----------
        move: str
            Move according UCI rules. For example, "d2d4"

        Returns
        -------
        x_initial: int
            Initial x (column) position of the move (from 0 to 7)
        y_initial: int
            Initial y (row) position of the move (from 0 to 7)
        x_final: int
            Final x (column) position of the move (from 0 to 7)
        y_final: int
            Final y (row) position of the move (from 0 to 7)
        promotion: str
            It can take the value "queen" (promotion) or rook, knight" or "bishop" (under-promotion).

        """
        if len(move) == 4:
            x_initial = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}[move[0]]
            y_initial = int(move[1]) - 1
            x_final = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}[move[2]]
            y_final = int(move[3]) - 1
            promotion = None
        else:
            assert len(move) == 5
            x_initial = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}[move[0]]
            y_initial = int(move[1]) - 1
            x_final = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}[move[2]]
            y_final = int(move[3]) - 1
            promotion = {"q": "queen", "r": "rook", "n": "knight", "b": "bishop"}[move[4]]

        return x_initial, y_initial, x_final, y_final, promotion
