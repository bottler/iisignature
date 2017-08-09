#This depends on kivy being properly installed - install its dependencies first.
#It allows you to draw paths freehand with the mouse in the canvas, and
#see their log signatures printed on the console.
#based on https://kivy.org/docs/tutorials/firstwidget.html

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line
import numpy as np, os, sys

#add the parent directory, so we find our iisignature build if it was built --inplace
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import iisignature

np.set_printoptions(suppress=True)

m=2
s=iisignature.prepare(2,m)

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        with self.canvas:
            Color(1, 1, 0)
            touch.ud['line'] = Line(points=(touch.x, touch.y))

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]
    
    def on_touch_up(self, touch):
        path=np.reshape(touch.ud['line'].points,(-1,2))
        print(iisignature.logsig(path,s))


class MyPaintApp(App):
    def build(self):
        return MyPaintWidget()

if __name__ == '__main__':
    MyPaintApp().run()