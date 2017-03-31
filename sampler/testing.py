from tkinter import Frame, Scrollbar, Canvas, Button, Text
from tkinter.constants import *
from tkinter import filedialog
from tkinter import Tk

import os
import cv2
import numpy as np
from collections import OrderedDict

from PIL import Image, ImageTk

EXPORT_DIR = "C:/Users/acer/Desktop/TestSamples/ML-Dataset/LBP/liver/training/"

class textureSampler(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        master.geometry("532x512")
        master.title("Texture Sampler")
        #master.resizable(False, True)
        self.image_files = []
        self.coordinates = OrderedDict()
        self.canvas_image_index = 0

    def initUI(self):

        self._init_directoryFrame()
        self._init_canvas()
        self._init_menu()




    def _init_directoryFrame(self):
        self.directory_frame = Frame(self.master, bg="gray", width=self.master['width'], height="32p")
        self.directory_frame.grid(column=0, row=0, columnspan =3, sticky='nswe')

        # dir button
        self.directory_button = Button(self.directory_frame, text="Select Folder", command=lambda: self._request_new_directory(),width=11)
        self.directory_button.grid(column=0, row=0, sticky='ne')

        # fg / foreground is the font color!
        self.directory_texts = Text(self.directory_frame,
                                    width=49,
                                    height=1,
                                    font=('Helvetica', 12),
                                    bg="black",
                                    fg="#37FF00",
                                    highlightcolor="#47FF00",
                                    highlightbackground="#DCDCDC",
                                    insertbackground="#37FF00",
                                    wrap = NONE)
        self.directory_texts.grid(column=1, row=0, columnspan=2)

        # Applying previous settings to directory path.
        file = open("latestdirectory.dat", "r")
        fol_path = file.readline()
        file.close()
        self._set_new_directory(fol_path)



    def _init_canvas(self):
        #loading the image
        if(len(self.image_files) == 0):
            #Frame for canvas
            try:
                self.canvas.delete("all")
                self.canvas.configure(bg="black")
                self.canvas.create_text(text="NO IMAGE FOUND IN DIRECTORY", fill="#FFFFFF", font=('Helvetica', 36), anchor=CENTER)
            except:
                print(" No Canvas found.")


        else:
            # Opening the image
            self.img = ImageTk.PhotoImage(
                Image.open(self.directory_texts.get(1.0, END).rstrip() + self.image_files[self.canvas_image_index]))

            # Frame for canvas
            self.canvas_frame = Frame(self.master,
                                      bg="black",
                                      width=self.img.width(),
                                      height=self.img.height(),
                                      padx=0,
                                      pady=0,
                                      bd=0)
            self.canvas_frame.grid(row=1,
                                   column=0,
                                   columnspan=2,
                                   sticky='nswe')

            #Canvas Scrollbars
            self.canvas_yscroll = Scrollbar(self.canvas_frame, orient=VERTICAL)
            self.canvas_xscroll = Scrollbar(self.canvas_frame, orient=HORIZONTAL)
            self.canvas_yscroll.pack(side = RIGHT, fill=Y, expand=True)
            self.canvas_xscroll.pack(side=BOTTOM, fill=X, expand=True)
            #Canvas
            self.canvas = Canvas(self.canvas_frame,
                                 width=self.img.width(),
                                 height=self.img.height(),
                                 relief="flat",
                                 selectborderwidth=0,
                                 bd=0,
                                 highlightthickness=0,
                                 yscrollcommand = self.canvas_yscroll.set,
                                 xscrollcommand = self.canvas_xscroll.set,
                                 scrollregion = (0,0,self.img.width(),self.img.height()))

            self.canvas_yscroll.configure(command=self.canvas.yview)
            self.canvas_xscroll.configure(command=self.canvas.xview)
            self.image_id = self.canvas.create_image(0, 0, image=self.img, anchor="nw")
            self.canvas.pack(fill=BOTH)
            print("Image {0} of {1}".format(self.canvas_image_index + 1, len(self.image_files)))

            # button events button 1 and 3 are mouse buttons respectively
            self.canvas.bind("<Button 1>", self._onCanvasClicked)
            self.canvas.bind("<Button 3>", self._onCanvasRClicked)
            self.master.bind("<Control-z>", self._onUndo)

    def _init_menu(self):
        #Menu Frame
        self.menu_frame = Frame(self.master,
                                  bg="green",
                                  padx=0,
                                  pady=0,
                                  width=512,
                                  height=64)
        self.menu_frame.grid(row=3, column=0, columnspan = 2, sticky='nswe')


        #Menu Buttons
        self.previous_image_button = Button(self.menu_frame, text="Previous Image", width=23, command=lambda: self._draw_previous_image())
        self.next_image_button = Button(self.menu_frame, text="Next Image", width=23, command=lambda: self._draw_next_image())
        self.export_textures_button = Button(self.menu_frame, text="Export Textures", width=23, command=lambda: self._export_textures())

        self.previous_image_button.grid(row=2, column=0)
        self.next_image_button.grid(row=2, column=1)
        self.export_textures_button.grid(row=2, column=2)

        if(len(self.image_files) == 0):
            self.previous_image_button['state'] = DISABLED
            self.next_image_button['state'] = DISABLED

        if(len(self.coordinates) == 0):
            self.export_textures_button['state'] = DISABLED


    def _draw_previous_image(self):
        if(len(self.image_files) == 0):
            print("No Image to display.")
        elif(self.canvas_image_index == 0):
            self._export_textures()
            print("No Previous Image to Index")
            print("Image {0} of {1}".format(self.canvas_image_index + 1, len(self.image_files)))
        else:
            self._export_textures()
            self.canvas_image_index -= 1
            self.img = ImageTk.PhotoImage(
                Image.open(self.directory_texts.get(1.0, END).rstrip() + self.image_files[self.canvas_image_index]))
            self.canvas.delete(ALL)
            self.image_id = self.canvas.create_image(0, 0, image=self.img, anchor="nw")
            print("Image {0} of {1}".format(self.canvas_image_index + 1, len(self.image_files)))

        self.coordinates = OrderedDict()
        self.export_textures_button['state'] = DISABLED

    def _draw_next_image(self):
        if(len(self.image_files) == 0):
            print("No Image to display.")
        elif(self.canvas_image_index == len(self.image_files) -1):
            self._export_textures()
            print("No Next Image to Index")
            print("Image {0} of {1}".format(self.canvas_image_index + 1, len(self.image_files)))

        else:
            self._export_textures()
            self.canvas_image_index += 1
            self.img = ImageTk.PhotoImage(
                Image.open(self.directory_texts.get(1.0, END).rstrip() + self.image_files[self.canvas_image_index]))
            self.canvas.delete(ALL)
            self.image_id = self.canvas.create_image(0, 0, image=self.img, anchor="nw")
            print("Image {0} of {1}".format(self.canvas_image_index + 1, len(self.image_files)))

        self.coordinates = OrderedDict()
        self.export_textures_button['state'] = DISABLED



    def _export_textures(self):
        img = cv2.imread(self.directory_texts.get(1.0, END).rstrip()+ self.image_files[self.canvas_image_index],0)
        if(not EXPORT_DIR + 'exports/'):
            os.makedirs(EXPORT_DIR + 'exports/')

        if(len(self.coordinates) == 0):
            print("Nothing to export !")

        else:
            i = 0
            for _ , (x,y) in self.coordinates.items():
                export_dir = EXPORT_DIR + 'exports/['+ str(i) + ']' + self.image_files[self.canvas_image_index]
                x0 = x - 18
                x1 = x + 18
                y0 = y - 18
                y1 = y + 18

                if self._isValidCoordinates(x,y):
                    texture = img[y0:y1,x0:x1]

                else:
                    tx0 = 0
                    ty0 = 0
                    tx1 = 36
                    ty1 = 36

                    texture = np.zeros((36,36))
                    if(x0 < 0):
                        tx0 = 36-x-18
                        x0 = 0

                    if(x1 > self.img.width()):
                        tx1 = 36- (x1 - self.img.width())
                        x1 = self.img.width()

                    if(y0 < 0):
                        ty0 = 36-y-18
                        y0 = 0

                    if(y1 > self.img.height()):
                        ty1 = 36 - (y1 - self.img.height())
                        y1 = self.img.height()

                    texture[ty0:ty1,tx0:tx1] = img[y0:y1,x0:x1]

                cv2.imwrite( export_dir,
                            texture)
                i += 1
                print("done!")



    def _request_new_directory(self):
        fol_path = filedialog.askdirectory(parent=self.master, initialdir=self.directory_texts.get(1.0, END).rstrip(), title='Please Specify a FOLDER')
        if(fol_path):
            fol_path = fol_path + '/'
            self._set_new_directory(fol_path)
            file = open('latestdirectory.dat','w')
            file.writelines(fol_path)
            file.close()
            self._init_canvas()

        else:
            print("No new directory selected.")



        print(fol_path)

    def _set_new_directory(self, fol_path):
        self.image_files = []
        self.canvas_image_index = 0
        self._acquire_image_files(fol_path)
        self.directory_texts.replace(1.0,END,fol_path)

        try:
            if (len(self.image_files) == 0):
                self.previous_image_button['state'] = DISABLED
                self.next_image_button['state'] = DISABLED
                self.export_textures_button['state'] = DISABLED
            else:
                self.previous_image_button['state'] = NORMAL
                self.next_image_button['state'] = NORMAL
                self.export_textures_button['state'] = NORMAL
        except:
            print("No Buttons Created Yet")

    def _acquire_image_files(self, path):
        for file in os.listdir(path):
            if file.endswith('.jpg') or file.endswith('.png'):
                self.image_files.append(file)


    def _clear_image_files(self):
        self.image_files = []

    def _isValidCoordinates(self, x,y):
        return (x-18 >= 0 and y-18 >= 0 and x+18 <= self.img.width() and y+18 <= self.img.height())

    def _onCanvasClicked(self, event):
        x = event.x
        y = event.y
        if(len(self.image_files) > 0):
            if self._isValidCoordinates(x,y):
                rectID = self.canvas.create_rectangle((x -18), (y -18), (x +18), (y +18), outline="red", activeoutline="#37FF00", width=3)
                self.coordinates.update({rectID : (x,y)})

            else:
                rectID = self.canvas.create_rectangle((x -18), (y -18), (x +18), (y +18), outline="#00FFFD", activeoutline="#37FF00", width=3)
                self.coordinates.update({rectID : (x,y)})

            if(self.export_textures_button['state'] == DISABLED):
                self.export_textures_button['state'] = NORMAL

    def _onCanvasRClicked(self, event):
        if(len(self.coordinates) > 0 and self.canvas.find_withtag(CURRENT)):
            if (self.canvas.find_withtag(CURRENT) != self.canvas.find_withtag(self.image_id)):
                rectID = self.canvas.find_withtag(CURRENT)[0]
                self.coordinates.pop(rectID)
                self.canvas.delete(CURRENT)

        if(len(self.coordinates) == 0):
            self.export_textures_button['state'] = DISABLED

    def _onUndo(self, event):
        if (len(self.coordinates) > 0):
            rectID, _ = self.coordinates.popitem()
            self.canvas.delete((rectID,))

        if(len(self.coordinates) == 0):
            self.export_textures_button['state'] = DISABLED


    def execute(self):
        self.master.mainloop()




if __name__ == "__main__":
    application = textureSampler(Tk())
    application.initUI()
    application.execute()