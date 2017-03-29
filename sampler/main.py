from tkinter import Frame
from tkinter import Tk
from tkinter import Scrollbar
from tkinter import Canvas
from PIL import Image, ImageTk

from tkinter import filedialog

if __name__ == "__main__":
    root = Tk()

    #setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief="sunken")
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)


    xscroll = Scrollbar(frame, orient="horizontal")
    xscroll.grid(row=1, column=0, sticky='e'+'w')
    yscroll = Scrollbar(frame)

    yscroll.grid(row=0, column=1, sticky='n'+'s')
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky='n'+'s'+'e'+'w')

    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill="both",expand=1)

    #adding the image
    File = filedialog.askopenfilename(parent=root, initialdir="C:/",title='Choose an image.')
    img = ImageTk.PhotoImage(Image.open(File))
    canvas.create_image(0,0,image=img, anchor="nw")
    canvas.config(scrollregion=canvas.bbox("all"))

    #function to be called when mouse is clicked
    def printcoords(event):
        #outputting x and y coords to console
        print (event.x,event.y)
    #mouseclick event
    canvas.bind("<Button 1>",printcoords)

    root.mainloop()