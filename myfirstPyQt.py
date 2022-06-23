import sys

import ctypes
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("myappid")
from PyQt5.QtWidgets import QApplication, QWidget,QPushButton,QLabel,QLineEdit,QDesktopWidget
from PyQt5.QtGui import QIcon
if __name__ == '__main__':
    app = QApplication(sys.argv)


    w = QWidget()

    # 设置窗口图标
    w.setWindowIcon(QIcon('C:/Users/Richard/Desktop/ComputerVisual/picture_file/ico1.png'))
    # 设置窗口标题
    w.setWindowTitle("LOL免费刷皮肤")
    w.resize(300,300) # 调整窗口的大小
    # w.move(0,0) # 将窗口设置在屏幕的左上角
    # 将窗口设置在屏幕的中央位置
    center_pointer = QDesktopWidget().availableGeometry().center()
    print(center_pointer)
    x = center_pointer.x()
    y = center_pointer.y()
    w.move(x,y) 
   
    # 纯文本
    label1 = QLabel("账号：",w)
    label2 = QLabel("密码：",w)

    # 显示位置与大小
    label1.setGeometry(20,20,30,30)
    label2.setGeometry(20,40,30,30)

    # 文本框
    edit1 = QLineEdit(w)
    edit1.setPlaceholderText("请输入账号")
    edit2 = QLineEdit(w)
    edit2.setPlaceholderText("请输入密码")
    edit1.setGeometry(55,20,100,20)
    edit2.setGeometry(55,40,100,20)

    # 按钮
    btn = QPushButton("开始刷皮肤",w)
    btn.setGeometry(70,70,70,30)
    

    
    

    # 展示窗口
    w.show()

    # 程序进行循环等待状态
    app.exec_()