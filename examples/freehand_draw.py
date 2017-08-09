#This allows you to draw paths freehand with the mouse on the canvas, and
#see their log signatures printed on the console.
#based on http://www.tkdocs.com/tutorial/canvas.html

from tkinter import *
from tkinter import ttk
import sys, os, numpy as np

#add the parent directory, so we find our iisignature build if it was built --inplace
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import iisignature

np.set_printoptions(suppress=True, precision=6)

m=3
s=iisignature.prepare(2,m)

lastx, lasty = 0, 0
points=[]

root = Tk()
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

canvas = Canvas(root)

def xy(event):
    global lastx, lasty, points
    lastx, lasty = event.x, event.y
    points=[[lastx,-lasty]]

def addLine(event):
    global lastx, lasty
    canvas.create_line((lastx, lasty, event.x, event.y))
    lastx, lasty = event.x, event.y
    points.append([lastx,-lasty])

def doneStroke(event):
    #scaled_points = points / np.array([canvas.winfo_width(), canvas.winfo_height()])
    scaled_points = points / np.array(max(canvas.winfo_width(), canvas.winfo_height()))
    #print(iisignature.sig(points,m))
    #print(iisignature.logsig(points,s))
    #print(iisignature.sig(scaled_points,m))
    print(iisignature.logsig(scaled_points,s))

canvas.grid(column=0, row=0, sticky=(N, W, E, S))
canvas.bind("<Button-1>", xy)
canvas.bind("<B1-Motion>", addLine)
canvas.bind("<B1-ButtonRelease>", doneStroke)
root.mainloop()
