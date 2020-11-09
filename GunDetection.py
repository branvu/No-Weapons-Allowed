import tkinter as tk                # python 3
from tkinter import font  as tkfont # python 3
import os
#import Tkinter as tk     # python 2
#import tkFont as tkfont  # python 2

class SampleApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")

        self.title("GunDetection.py")

        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, PageTwo):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.configure(bg='grey')
        self.controller = controller
        label = tk.Label(self, text="Welcome to GunDetection.py!", font=controller.title_font, bg='grey')
        label.pack(side="top", fill="x", pady=10)


        button2 = tk.Button(self, text="Python Programs",
                            command=lambda: controller.show_frame("PageTwo"))
        quitButton = tk.Button(self,
                               text="Exit",
                               command=lambda: quit())


        button2.pack()
        quitButton.pack()





class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Python Programs", font=controller.title_font, bg='grey')
        label.pack(side="top", fill="x", pady=10)
        self.configure(bg='grey')
        button = tk.Button(self, text="Home",
                           command=lambda: controller.show_frame("StartPage"))
        speedButton = tk.Button(self,
                                 text="Quick Detection with OpenCV",
                                 command=lambda: os.system('python cv_detection.py'))
        deepLearningButton = tk.Button(self,
                                        text="Deep Learning Detection",
                                        command=lambda: os.system('python deep_learning_detection.py'))
        mixedLearningButtonv1 = tk.Button(self,
                                                             text="Mixed Learning Detection",
                                                             command=lambda:os.system('python mixed_detection.py'))


        button.pack(side='bottom')
        speedButton.pack()
        deepLearningButton.pack()
        mixedLearningButtonv1.pack()



if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()