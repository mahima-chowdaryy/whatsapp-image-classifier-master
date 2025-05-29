from PyQt5 import QtGui
from PyQt5.QtWidgets import QDialog, QApplication
from ui import Ui_Dialog
from image_classifier import classify


class Dialog(QDialog):
    def __init__(self):
        super(Dialog, self).__init__()

        # Set up the user interface from Designer.
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # Connect up the buttons.
        self.ui.run.clicked.connect(self.run)
        self.ui.cancel.clicked.connect(self.cancel)



    def run(self):
        path = self.ui.textBox.text()
        if path is not '':
            classify(self,path)

    def cancel(self):
        sys.exit()



import sys        
app = QApplication(sys.argv)
window = Dialog() 
ui = Ui_Dialog()
ui.setupUi(window)

window.show()
sys.exit(app.exec_())