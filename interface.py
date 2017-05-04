from tkinter import Frame, Scrollbar, Canvas, Button, Text, Label, IntVar
from tkinter.constants import *
from tkinter import filedialog
from tkinter import Tk
from tkinter import ttk

from mainLBP import runTrainingProgramme
import CONSTANTS
import os
import cv2
import numpy as np

import time

from collections import OrderedDict

from PIL import Image, ImageTk


class textureSampler(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        master.geometry("570x265")
        master.title("LBP Trainer")
        #master.resizable(False, True)

    def initUI(self):
        self._init_menu()

    def _init_menu(self):
        self.menu_frame = Frame(self.master,
                                  padx=0,
                                  pady=0)
        self.menu_frame.grid(row=0, column=0, sticky='nswe')
        self.directory_button = Button(self.menu_frame, text="Select Folder",
                                       command=lambda: self._request_new_directory(), width=16)
        self.textBox_directory = Text(self.menu_frame,
                                    width=49,
                                    height=1,
                                    font=('Helvetica', 12),
                                    bg="black",
                                    fg="#37FF00",
                                    highlightcolor="#47FF00",
                                    highlightbackground="#DCDCDC",
                                    insertbackground="#37FF00",
                                    wrap = NONE)
        self.textBox_directory.delete(1.0, END)
        self.textBox_directory.insert(END, CONSTANTS.BASE_DIR)


        self.label_cList = Label(self.menu_frame, text="[ Linear SVM ]\t C LIST\t\t\t\t\t:", anchor=NW, justify=LEFT)
        self.label_lbpNP = Label(self.menu_frame, text="[ LBP ]\t\t No. Neighbourhood Points\t\t:")
        self.label_lbpRad = Label(self.menu_frame, text="[ LBP ]\t\t Radius\t\t\t\t\t:")
        self.label_tileSize = Label(self.menu_frame, text="[ TILE Size ]\t Sliding Window Size ( X , Y )\t\t:")
        self.label_folds = Label(self.menu_frame, text="[ Kfolds ]\t\t No. of folds\t\t\t\t:")
        self.label_gamma = Label(self.menu_frame, text="[ Pre-processing ]\t Gamma (1.0 = No Changes)\t\t:")
        self.label_histEQ = Label(self.menu_frame, text="[ Pre-processing ]\t Use Histogram Equalization\t\t:")
        self.label_canny = Label(self.menu_frame, text="[ Pre-processing ]\t Use Histogram Canny Edge\t\t:")
        self.label_sdv = Label(self.menu_frame, text="[ Other Features ]\t Use Standard Deviation Coefficient\t\t:")
        self.label_euclidDist = Label(self.menu_frame, text="[ Other Features ]\t Use Distance from approximate Centre\t:")

        self.textBox_cList = Text(self.menu_frame, height=1, width=15)
        self.textBox_lbpNP = Text(self.menu_frame, height=1, width=15)
        self.textBox_lbpRad = Text(self.menu_frame, height=1, width=15)
        self.textBox_tileSize = Text(self.menu_frame, height=1, width=15)
        self.textBox_folds = Text(self.menu_frame, height=1, width=15)
        self.textBox_gamma = Text(self.menu_frame, height=1, width=15)

        self.isCheckedHistEQ = IntVar(value = 1)
        self.isCheckedCanny = IntVar(value = 0)
        self.isCheckedSdv = IntVar(value = 1)
        self.isCheckedEuclidDist = IntVar(value = 0)

        self.checkBox_histEQ = ttk.Checkbutton(self.menu_frame, variable = self.isCheckedHistEQ)
        self.checkBox_canny = ttk.Checkbutton(self.menu_frame, variable = self.isCheckedCanny)
        self.checkBox_sdv = ttk.Checkbutton(self.menu_frame, variable = self.isCheckedSdv)
        self.checkBox_euclidDist = ttk.Checkbutton(self.menu_frame, variable = self.isCheckedEuclidDist)

        self.executeButton = Button(self.menu_frame, text="RUN", width=80, command=lambda: self.runModel())

        self.textBox_directory.grid(row=0, column=0, sticky=W)
        self.label_cList.grid(row=1, column=0, sticky=W)
        self.label_lbpNP.grid(row=2, column=0, sticky=W)
        self.label_lbpRad.grid(row=3, column=0, sticky=W)
        self.label_tileSize.grid(row=4, column=0, sticky=W)
        self.label_folds.grid(row=5, column=0, sticky=W)
        self.label_gamma.grid(row=6, column=0, sticky=W)
        self.label_histEQ.grid(row=7, column=0, sticky=W)
        self.label_canny.grid(row=8, column=0, sticky=W)
        self.label_sdv.grid(row=9, column=0, sticky=W)
        self.label_euclidDist.grid(row=10, column=0, sticky=W)

        self.directory_button.grid(row=0, column=1, sticky=W)
        self.textBox_cList.grid(row=1, column=1, sticky=W)
        self.textBox_lbpNP.grid(row=2, column=1, sticky=W)
        self.textBox_lbpRad.grid(row=3, column=1, sticky=W)
        self.textBox_tileSize.grid(row=4, column=1, sticky=W)
        self.textBox_folds.grid(row=5, column=1, sticky=W)
        self.textBox_gamma.grid(row=6, column=1, sticky=W)
        self.checkBox_histEQ.grid(row=7, column=1, sticky=W)
        self.checkBox_canny.grid(row=8, column=1, sticky=W)
        self.checkBox_sdv.grid(row=9, column=1, sticky=W)
        self.checkBox_euclidDist.grid(row=10, column=1, sticky=W)

        self.executeButton.grid(row=11, columnspan=2, sticky=W)

    def _request_new_directory(self):
        fol_path = filedialog.askdirectory(parent=self.master, initialdir=self.textBox_directory.get(1.0, END).rstrip(), title='Please Specify a FOLDER')
        if(fol_path):
            self.textBox_directory.delete(1.0,END)
            self.textBox_directory.insert(END,fol_path)

        else:
            print("No new directory selected.")

    def runModel(self):
        baseDir = self.textBox_directory.get(1.0, END).rstrip()
        cParams = cList = list(map(int,self.textBox_cList.get(1.0, END).rstrip().split(',')))
        lbpNP = int(self.textBox_lbpNP.get(1.0, END).rstrip())
        lbpRad = int(self.textBox_lbpRad.get(1.0, END).rstrip())
        folds = int(self.textBox_folds.get(1.0, END).rstrip())
        gamma = float(self.textBox_gamma.get(1.0, END).rstrip())
        tileSize = self.textBox_tileSize.get(1.0, END).rstrip().split(',')
        tileWidth = int(tileSize[0])
        tileHeight = int(tileSize[1])

        useHistEQ = self.isCheckedHistEQ.get() == True
        useCanny = self.isCheckedCanny.get() == True
        useSDV = self.isCheckedSdv.get() == True
        useCCM = self.isCheckedEuclidDist.get() == True

        print(baseDir, cParams, lbpNP, lbpRad, folds, gamma, useHistEQ, useCanny, useSDV, useCCM, (tileWidth,tileHeight))
        runTrainingProgramme(dsBaseDir=baseDir, cList=cParams, descNPoints=lbpNP, descRadius=lbpRad,
                             folds=folds,useCannyEdge=useCanny,gamma=gamma,useHistEQ=useHistEQ,useSDV=useSDV,
                            useCCostMeasure=useCCM, tile_dimensions = (tileWidth,tileHeight))

        print("Model Running !")

    def execute(self):
        self.master.mainloop()




if __name__ == "__main__":
    application = textureSampler(Tk())
    application.initUI()
    application.execute()