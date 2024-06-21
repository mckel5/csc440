import copy
from tkinter import Button
from tkinter import Canvas
from tkinter import NORMAL
from tkinter import PhotoImage
from tkinter import Tk

from convex_hull import compute_hull


def draw_point(canvas, x, y):
    return canvas.create_image((x, y), image=ram, state=NORMAL)


def add_point(event):
    draw_point(w, event.x, event.y)
    points.append((event.x, event.y))
    return


def draw_hull():
    hull = copy.copy(compute_hull(points))
    hull.append(hull[0])
    for i in range(0, len(hull) - 1):
        x1 = hull[i][0]
        y1 = hull[i][1]
        x2 = hull[i + 1][0]
        y2 = hull[i + 1][1]
        w.create_line(x1, y1, x2, y2, width=3)
    return


if __name__ == '__main__':
    master, points = Tk(), list()

    submit_button = Button(master, text="Draw Hull", command=draw_hull)
    submit_button.pack()
    quit_button = Button(master, text="Quit", command=master.quit)
    quit_button.pack()

    canvas_width = 1000
    canvas_height = 800
    w = Canvas(master,
               width=canvas_width,
               height=canvas_height)
    ram = PhotoImage(file="ram-sm.gif")
    w.pack()
    w.bind('<Button-1>', add_point)

    w.mainloop()
