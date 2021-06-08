import random

from kivy.app import App
from kivy.core.window import Window
from kivy.graphics import Color, Line, Rectangle
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import save_img


import os
import matplotlib.pyplot as plt

# RGBA = Red,Green,Blue, Opacity
Window.clearcolor = (0, 0, 0, 1)
model = load_model("mnist_model.h5")

class PaintWindow(Widget):
    def on_touch_down(self, touch):
        colorR = random.randint(0, 255)
        colorG = random.randint(0, 255)
        colorB = random.randint(0, 255)
        self.canvas.add(Color(rgb=(1, 1, 1)))
        d = 10
        touch.ud['line'] = Line(points=(touch.x, touch.y), width = d)
        self.canvas.add(touch.ud['line'])

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]


# Root Window = Paint Window + Button
class PaintApp(App):
    def build(self):
        rootWindow = Widget()
        self.painter = PaintWindow()
        clearBtn = Button(text='Clear', pos=(0, -1))
        clearBtn.bind(on_release=self.clear_canvas)
        predictBtn = Button(text='Predict', pos=(100, -1))
        predictBtn.bind(on_release=self.predict_digit)
        self.lbl = Label(text = "The digit recognized will appear here...", pos=(300, -1), color=[1, 1, 1, 1.0])
        with self.lbl.canvas.before:
            Color(0.35, 0.35, 0.35, 1.0)
            Rectangle(pos=(200, 0), size=(300,100))
        rootWindow.add_widget(self.painter)
        rootWindow.add_widget(clearBtn, index = 0)
        rootWindow.add_widget(self.lbl, index = 1)
        rootWindow.add_widget(predictBtn, index = 2)

        return rootWindow

    def clear_canvas(self, obj):
        self.painter.canvas.clear()
        self.lbl.text = "The digit recognized will appear here..."
        if os.path.exists("digit.png"):
            os.remove("digit.png")
        if os.path.exists("digit0001.png"):
            os.remove("digit0001.png")


    def predict_digit(self, obj):
        Window.screenshot("digit.png")
        img = load_img("digit0001.png")
        img_array = img_to_array(img)[:-100]
        save_img('digit.png', img_array)
        if os.path.exists("digit0001.png"):
            os.remove("digit0001.png")
        print("modified image")
        arr = np.array(load_img("digit.png", target_size=(28,28) ,grayscale=True))
        test = arr
        self.lbl.text = str(model.predict(np.array([test])).argmax(axis=1)[0])





PaintApp().run()