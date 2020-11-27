"""
Implementation of actual game with display

@author: DiTurr

"""
import chess
import chess.svg
from PyQt5.QtSvg import QSvgWidget
from PyQt5 import QtWidgets, QtCore
import sys

from chess_engine.game import ChessGame # NOQA


class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        """
        This is the main window from my GUI
        """
        super().__init__()
        # Generate chess board
        self.setGeometry(0, 0, 620, 620)
        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(10, 10, 600, 600)
        self.chessboard = chess.Board()
        self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg)

        # configure and start thrread
        self.displayer_chess_game = ChessEngineBackend()
        self.thread = QtCore.QThread(self)
        self.displayer_chess_game.move.connect(self.move_callback) # NOQA
        self.displayer_chess_game.moveToThread(self.thread)
        self.thread.started.connect(self.displayer_chess_game.chess_engine_backend) # NOQA
        self.thread.start()

    @QtCore.pyqtSlot(str)
    def move_callback(self, move):
        """


        """
        self.chessboard.push(chess.Move.from_uci(move))
        self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg)


class ChessEngineBackend(QtCore.QObject):
    move = QtCore.pyqtSignal(str)
    
    @QtCore.pyqtSlot()
    def chess_engine_backend(self):
        game = ChessGame(path_load_model="/home/ditu/Documents/03_Projects/chess_engine/models/alphazero_model.pth",
                         func_emit=self.move.emit) # NOQA
        game.play_game()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create('Plastique'))
    myGUI = MyWidget()
    myGUI.show()
    sys.exit(app.exec_())
