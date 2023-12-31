from PyQt5.QtWidgets import QMainWindow, QApplication

from .gui import Ui_MainWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.show()

if __name__ == "__main__":
    app = QApplication([])
    main_window = MainWindow()
    app.exec_()