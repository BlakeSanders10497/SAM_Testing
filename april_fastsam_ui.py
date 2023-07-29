import argparse
import os
import numpy as np
import cv2

# UI requirements
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter.ttk import *
from PIL import ImageTk, Image

# FastSAM requirements
from fastsam import FastSAM, FastSAMVideoPrompt
import ast
import torch
from utils.tools import convert_box_xywh_to_xyxy

class App(Frame):
    """ Tkinter application for creating and displaying the water edge detection tool. """

    def print_debug(self, msg):
        """ Print a debug message if the \'verbose\' argument was passed. """
        if(args.verbose):
            print('[Debug]:', msg)

    def browse_files(self):
        """ Spawn a file browser from which the user can select an image. """
        filename = filedialog.askopenfilename(initialdir='.',
                                                title='Select a File',
                                                filetypes= (('Images', '.tif .jpg .png'),
                                                            ('All files', '*.*'))) 
        # Check if the user cancelled instead of selecting a file
        if not filename:
            self.print_debug('no file selected.')
            return

        self.img_filename.configure(text='Image: '+filename[filename.rfind('/')+1:])
        with Image.open(filename) as img:
            self.img = img
            self.tk_img = ImageTk.PhotoImage(img)
            self.img_view.configure(image=self.tk_img)

    def segment(self):
        """ Begin segmentation on the selected image. """

        # Check if we've selected an image yet
        if not self.img:
            print("No image has been selected yet! Unable to segment.")
            return

        self.device = torch.device(
            'cuda'  if torch.cuda.is_available() else
            'mps'   if torch.backends.mps.is_available() else
            'cpu')

        self.print_debug('Segmenting!')

        # Create model
        #breakpoint()
        model = FastSAM(args.model_path)
        everything_results = model(
            self.img,
            device=self.device,
            retina_masks=args.retina,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou)

        # Create a prompt receiver that acts on the selected image
        prompt_process = FastSAMVideoPrompt(np.array(self.img), everything_results, device=self.device)

        # Return all segments so the user can select one manually
        annotations = prompt_process.everything_prompt()

        # Get output image
        result = prompt_process.plot(
            annotations=annotations,
            mask_random_color=args.random_color)

        # Display output image
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(result))
        self.img_view.configure(image=self.tk_img)

        self.print_debug('Segmentation complete.')
        return

    def poly_edit_changed(self):
        """ Respond to polygon editing being enabled or disabled. """
        state_msg = 'Polygon editing ' + ('enabled' if self.allow_edit_poly.get() else 'disabled')
        self.print_debug(state_msg)
        pass

    def latlong_edit_changed(self):
        """ Respond to lat-long editing being enabled or disabled. """
        state_msg = 'Lat-long editing ' + ('enabled' if self.allow_edit_latlong.get() else 'disabled')
        self.print_debug(state_msg)
        pass

    def export_to_csv(self):
        """ Export the selected segment's polygon as lat-long points to a CSV file. """
        self.print_debug('exporting!')
        pass
    
    def __init__(self, root):
        ttk.Frame.__init__(self, root)
        self.pack()

        self.winfo_toplevel().title('APRILab Water Body Detection Tool (one day maybe lol)')

        # State variables
        self.allow_edit_poly    = tk.BooleanVar(value=False)
        self.allow_edit_latlong = tk.BooleanVar(value=False)


        # Top row
        self.img_sel        = ttk.Button        (self, text='Select image file', command=self.browse_files)
        self.status         = ttk.Label         (self, text='Open an image')
        self.img_filename   = ttk.Label         (self, text='Image file name displays here')

        # Side bar
        self.segment        = ttk.Button        (self, text='Segment Image',    command=self.segment)
        self.edit_label     = ttk.Label         (self, text='Edit:')
        self.edit_polygon   = ttk.Checkbutton   (self, text='Bounding polygon', command=self.poly_edit_changed,     variable=self.allow_edit_poly)
        self.edit_latlong   = ttk.Checkbutton   (self, text='Lat-long points',  command=self.latlong_edit_changed,  variable=self.allow_edit_latlong)
        self.export         = ttk.Button        (self, text='Export to CSV',    command=self.export_to_csv)

        # Image display
        self.img_view       = ttk.Label         (self, text='Image appears here')

        # Gridding
        self.img_sel        .grid(row=0, column=0, sticky=[N, W, E, S])
        self.status         .grid(row=0, column=1, sticky=[N, W, E, S])
        self.img_filename   .grid(row=0, column=2, sticky=[N, W, E, S])
        self.segment        .grid(row=1, column=0, sticky=[N, W, E])
        self.edit_label     .grid(row=2, column=0, sticky=[N, W, E])
        self.edit_polygon   .grid(row=3, column=0, sticky=[N, W, E])
        self.edit_latlong   .grid(row=4, column=0, sticky=[N, W, E])
        self.export         .grid(row=5, column=0, sticky=[W, E])

        self.img_view       .grid(row=1, column=1, rowspan=5, columnspan=2)
    

def parse_args():
    """ Use an ArgumentParser to collect arguments into variables. """
    parser = argparse.ArgumentParser()

    parser.add_argument('-d',   '--dark_theme',     type=bool,  default=False,  action=argparse.BooleanOptionalAction)
    parser.add_argument('-v',   '--verbose',        type=bool,  default=False,  action=argparse.BooleanOptionalAction)
    parser.add_argument('-m',   '--model_path',     type=str,   default='./weights/FastSAM-x.pt')
    parser.add_argument(        '--retina',         type=bool,  default=True)
    parser.add_argument(        '--imgsz',          type=int,   default=1024)
    parser.add_argument(        '--conf',           type=float, default=0.4)
    parser.add_argument(        '--iou',            type=float, default=0.9)
    parser.add_argument(        '--random_color',   type=bool,  default=True,   action=argparse.BooleanOptionalAction)

    return parser.parse_args()


def main(args):
    root = tk.Tk()
    root.minsize(400, 200)

    if(args.dark_theme):
        root.tk.call('lappend', 'auto_path', '/home/blake/Documents/GitHub/FastSAM/FastSAMUI/themes/awthemes-10.4.0')
        root.tk.call('package', 'require', 'awdark')
        s = ttk.Style()
        s.theme_use('awdark')
        root.configure(bg='#33393b')

    app = App(root)
    root.mainloop()
  
if __name__=="__main__":
    args = parse_args()
    main(args)
