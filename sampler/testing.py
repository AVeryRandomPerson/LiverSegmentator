from tkinter import Frame,Scrollbar,Canvas, Button, Text, INSERT, END, CENTER
from tkinter import N, S, W, E, BOTH,HORIZONTAL, ALL
from tkinter import filedialog
from tkinter import PhotoImage
from tkinter import Tk
from tkinter import font
import os


from PIL import Image, ImageTk



class textureSampler(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        master.geometry("512x512")
        master.title("Texture Sampler")
        self.image_files = []
        self.canvas_image_index = 0

    def initUI(self):

        self._init_directoryFrame()
        self._init_canvas()
        self._init_menu()



    def _init_directoryFrame(self):
        self.directory_frame = Frame(self.master, bg="black", width="512p", height="32p")
        self.directory_frame.grid(column=0, row=0)

        # fg / foreground is the font color!
        self.directory_texts = Text(self.directory_frame,
                                    width=48,
                                    height=1,
                                    font=('Helvetica', 12),
                                    bg="black",
                                    fg="#37FF00",
                                    highlightcolor="#47FF00",
                                    highlightbackground="#DCDCDC")
        self.directory_texts.grid(column=0, row=0)

        # Applying previous settings to directory path.
        file = open("latestdirectory.dat", "r")
        fol_path = file.readline()
        file.close()
        self._set_new_directory(fol_path)

        # dir button
        self.directory_button = Button(self.directory_frame, text="Select Folder", command=lambda: self._request_new_directory())
        self.directory_button.grid(column=1, row=0)

    def _init_canvas(self):
        #loading the image
        if(len(self.image_files) == 0):
            #Frame for canvas
            try:
                self.canvas.delete("all")
                self.canvas.configure(bg="black")
                self.canvas.create_text(text="NO IMAGE FOUND IN DIRECTORY", fill="#FFFFFF", font=('Helvetica', 24), anchor=CENTER)
            except:
                print(" No Canvas found.")

        else:
            self.img = ImageTk.PhotoImage(
                Image.open(self.directory_texts.get(1.0, END).rstrip() + self.image_files[self.canvas_image_index]))

            # Frame for canvas
            self.canvas_frame = Frame(self.master,
                                      bg="black",
                                      width=self.img.width(),
                                      height=self.img.height(),
                                      padx=0,
                                      pady=0)
            self.canvas_frame.grid(row=1,
                                   column=0,
                                   columnspan=2,
                                   sticky='nswe')


            #Canvas
            self.canvas = Canvas(self.canvas_frame,
                                 width=self.img.width(),
                                 height=self.img.height(),
                                 relief="flat",
                                 selectborderwidth=0,
                                 bd=0,
                                 highlightthickness=0)

            self.canvas.create_image(0, 0, image=self.img, anchor="nw")
            self.canvas.pack()
            self.canvas_frame.grid(row=1, column=0)

            # mouseclick event
            self.canvas.bind("<Button 1>", self._onCanvasClicked)

    def _init_menu(self):
        #Menu Frame
        self.menu_frame = Frame(self.master,
                                  bg="green",
                                  padx=0,
                                  pady=0,
                                  width=512,
                                  height=64)
        self.menu_frame.grid(row=2, column=0, columnspan = 2, sticky='nswe')


        #Menu Buttons
        self.previous_image_button = Button(self.menu_frame, text="Previous Image", command=lambda: self._draw_previous_image())
        self.next_image_button = Button(self.menu_frame, text="Next Image", command=lambda: self._draw_next_image())
        self.export_textures_button = Button(self.menu_frame, text="Export Textures", command=lambda: self._export_textures())

        self.previous_image_button.grid(row=2, column=0)
        self.next_image_button.grid(row=2, column=1)
        self.export_textures_button.grid(row=2, column=2)


    def _draw_previous_image(self):
        if(len(self.image_files) == 0):
            print("No Image to display.")
        elif(self.canvas_image_index == 0):
            print("No Previous Image to Index")
        else:
            self.canvas_image_index -= 1
            self.img = ImageTk.PhotoImage(
                Image.open(self.directory_texts.get(1.0, END).rstrip() + self.image_files[self.canvas_image_index]))

            self.canvas.create_image(0, 0, image=self.img, anchor="nw")

    def _draw_next_image(self):
        if(len(self.image_files) == 0):
            print("No Image to display.")
        elif(self.canvas_image_index == len(self.image_files) -1):
            print("No Next Image to Index")
        else:
            self.canvas_image_index += 1
            self.img = ImageTk.PhotoImage(
                Image.open(self.directory_texts.get(1.0, END).rstrip() + self.image_files[self.canvas_image_index]))

            self.canvas.create_image(0, 0, image=self.img, anchor="nw")

    def _export_textures(self):
        print("Noob")

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

    def _acquire_image_files(self, path):
        for file in os.listdir(path):
            if file.endswith('.jpg') or file.endswith('.png'):
                self.image_files.append(file)


    def _clear_image_files(self):
        self.image_files = []

    def _onCanvasClicked(self, event):
        if(len(self.image_files) > 0):
            print(event.x, event.y)
            self.canvas.create_oval((event.x -8), (event.y -8), (event.x +8), (event.y +8), outline="red", activeoutline="#37FF00")


    def execute(self):
        self.master.mainloop()

application = textureSampler(Tk())
application.initUI()
application.execute()


'''
if __name__ == "__main__":
    root = Tk()
    root.geometry("512x512")
    root.resizable(height=False, width=False)
    root.title("Texture Sampler")

    #create a general frame.
    mainFrame = Frame(root, bd=2, relief="ridge", height="512p", width="512p")
    mainFrame.grid_rowconfigure(0, weight=1)
    mainFrame.grid_columnconfigure(0, weight=1)


    xScrollBar = Scrollbar(mainFrame, orient=HORIZONTAL)
    xScrollBar.grid(row=1, column=0, sticky=E+W)
    yScrollBar = Scrollbar(mainFrame)
    yScrollBar.grid(row=0, column=1, sticky=N+S)

    canvas = Canvas(mainFrame, bd=0, xscrollcommand=xScrollBar.set, yscrollcommand=yScrollBar.set, height="512p", width="512p")
    canvas.grid(row=0, column=0)#, sticky=N+S+E+W)

    xScrollBar.config(command=canvas.xview)
    yScrollBar.config(command=canvas.yview)
    mainFrame.pack(fill=BOTH,expand=1)

    #adding the image
    #File = filedialog.askopenfilename(parent=root, initialdir="C:/",title='Choose an image.')
    img = ImageTk.PhotoImage(Image.open("C:/Users/acer/Desktop/TestSamples/BodyOnly/Mixed/I0000049.jpg"))#File))
    canvas.create_image(0,0,image=img,anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))

    #function to be called when mouse is clicked
    def printcoords(event):
        #outputting x and y coords to console
        print (event.x,event.y)
    #mouseclick event
    canvas.bind("<Button 1>",printcoords)

    root.mainloop()

'''