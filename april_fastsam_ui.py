from enum import Enum
import argparse
import os
import numpy as np
import cv2
import csv

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


# create_circle function addition to tkinter
# Source: https://stackoverflow.com/questions/17985216/simpler-way-to-draw-a-circle-with-tkinter
def _create_circle(self, x, y, r, **kwargs):
    return self.create_oval(x-r, y-r, x+r, y+r, **kwargs)
tk.Canvas.create_circle = _create_circle


# Current state of the app, which drives what is enabled/disabled in the UI,
#   as well as the status message at the top of the app window.
AppState = Enum('AppState', ['IMAGE_SELECT',
                               'IMAGE_SEGMENT', 
                               'CONTOUR_SELECT',
                               'CONTOUR_EDIT'])
class App(Frame):
    """ Tkinter application for creating and displaying the water edge detection tool. """

    def print_debug(self, msg):
        """ Print a debug message if the \'verbose\' argument was passed. """
        if(args.verbose):
            print('[Debug]:', msg)

    def disable_menu_options(self):
        """ Disable all UI options except for opening a new image. """

        self.btn_segment    .config(state='disabled') 
        self.check_polygon  .config(state='disabled') 
        self.check_latlong  .config(state='disabled') 
        self.entry_left     .config(state='disabled') 
        self.entry_right    .config(state='disabled') 
        self.entry_top      .config(state='disabled') 
        self.entry_bottom   .config(state='disabled') 
        self.btn_export     .config(state='disabled') 
        
        self.label_edit     .config(foreground='grey') 
        self.label_left     .config(foreground='grey') 
        self.label_right    .config(foreground='grey') 
        self.label_top      .config(foreground='grey') 
        self.label_bottom   .config(foreground='grey') 

    def enable_img_options(self):
        """ Enable UI checkboxes and edit label. """

        self.check_polygon  .config(state='enabled') 
        self.check_latlong  .config(state='enabled') 
        self.btn_export     .config(state='enabled') 

        self.label_edit     .config(foreground='black') 

    def set_segmenting_state(self, new_state):
        """ Update the state of the segmenting button. Just a wrapper for readability purposes. """

        self.btn_segment.config(state=new_state) 

    def update_state(self, new_state):
        """ Update the state of the app and perform any resulting changes to the UI. """

        # Update the class member
        self.app_state = new_state

        # Temp variable to hold status message
        status_msg = ''

        # Perform any actions required by the new state
        if new_state == AppState.IMAGE_SELECT:
            status_msg = 'Open an image'
            self.disable_menu_options()

        elif new_state == AppState.IMAGE_SEGMENT:
            status_msg = 'Segment the image, or select a new image'
            self.set_segmenting_state('enabled')

        elif new_state == AppState.CONTOUR_SELECT:
            status_msg = 'Select a contour, or select a new image'
            self.set_segmenting_state('disabled')

        elif new_state == AppState.CONTOUR_EDIT:
            status_msg = 'Edit the selected contour, select a new contour, export the contour to CSV, or select a new image'
            self.enable_img_options()

        else:
            self.print_debug(f'State {new_state} has not been accounted for in update_state!')

        # Update status message
        self.label_status.configure(text=status_msg)


    def on_left_mouse_button(self, event):
        """ Event handler to be triggered by pressing the left mouse button. """

        if self.app_state == AppState.CONTOUR_SELECT:
            # Check if we clicked on a segment
            segment_contour_full = self.check_for_segment(event.x, event.y)
            if segment_contour_full is None:
                return

            # Simplify contour before drawing and storing
            # Reference: https://stackoverflow.com/questions/41879315/opencv-visualize-polygonal-curves-extracted-with-cv2-approxpolydp
            epsilon = (2e-3)*cv2.arcLength(segment_contour_full, True)
            self.segment_contour = cv2.approxPolyDP(segment_contour_full, epsilon, True)

            self.print_debug(f'Simplified: {self.segment_contour.shape[0]} points')
            self.print_debug(f'Original: {segment_contour_full.shape[0]} points')
            
            # Update the contour on the canvas
            self.redraw_contour()

            self.update_state(AppState.CONTOUR_EDIT)

        elif self.app_state == AppState.CONTOUR_EDIT:
            # Make sure we are actively editing the polygon
            if not self.allow_edit_poly.get():
                return

    def point_drag_start(self, event):
        """ Start dragging a point in the contour that is currently being edited. """
        # Determine which point we clicked on
        self._clicked_point, self._clicked_point_index = self.check_for_contour_point(event.x, event.y)

        # Get its canvas ID
        self._clicked_point_canvas_id = self.img_canvas.find_withtag('current')[0]

        # Capture initial position
        self._prev_x = event.x
        self._prev_y = event.y
        self.print_debug(f'Drag start! ({event.x}, {event.y})')

    def point_drag_motion(self, event):
        """ Drag a point in the contour that is currently being edited. """
        dx = event.x - self._prev_x
        dy = event.y - self._prev_y

        self._prev_x = event.x
        self._prev_y = event.y

        self.img_canvas.move(self._clicked_point_canvas_id, dx, dy)
        self.print_debug(f'Point {self._clicked_point_index}, id {self._clicked_point_canvas_id} update: ({event.x}, {event.y})')

    def point_drag_end(self, event):
        """ Update and redraw the contour that is currently being edited. """
        self._clicked_point[0][0] = event.x
        self._clicked_point[0][1] = event.y


        self.print_debug(f'point {self._clicked_point_index}, id {self._clicked_point_canvas_id} released at {event.x}, {event.y}')

        self.redraw_contour()


    def redraw_contour(self):
        """ Deletes the old contour, if any, and draws a new contour on the image canvas. """

        # Delete old polygon
        self.img_canvas.delete(self.contour_tag)

        # List out contour coordinates like Tk is expecting: [x0, y0, x1, y1, ...]
        contour_points_list = []
        for point in self.segment_contour:
            contour_points_list.append(point[0][0])
            contour_points_list.append(point[0][1])

        # Draw the new polygon
        self.img_canvas.create_polygon(contour_points_list, outline='green', fill='', width=3, tags=(self.contour_tag))

        # If we are editing the contour, redraw the contour points
        if self.allow_edit_poly.get():
            self.redraw_contour_points()

    def redraw_contour_points(self):
        self.img_canvas.delete(self.contour_points_tag)

        # Draw the contour points
        for point in self.segment_contour:
            self.img_canvas.create_circle(x=point[0][0], y=point[0][1], r=self.contour_point_radius, fill='blue', outline='white', tags=self.contour_points_tag)

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

        self.img_filename.configure(text='Image: '+ filename[filename.rfind('/')+1:])
        with Image.open(filename) as img:
            # Update the image
            self.img = img
            self.tk_img = ImageTk.PhotoImage(img)
            
            self.update_canvas_image()

            self.update_state(AppState.IMAGE_SEGMENT)

    def update_canvas_image(self):
        """ Updates the image to display on the Canvas while resizing the Canvas. """
        # Resize canvas to match the new image
        self.img_canvas.configure(width=self.tk_img.width(), height=self.tk_img.height())

        # Add image to canvas
        self.img_canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        
    def segment(self):
        """ Begin segmentation on the selected image. """

        # Check if we've selected an image yet
        if not self.img:
            print("No image has been selected yet! Unable to segment.")
            return

        # Pick a device to run segmentation on
        device = torch.device(
            'cuda'  if torch.cuda.is_available() else
            'mps'   if torch.backends.mps.is_available() else
            'cpu')

        self.print_debug('Segmenting!')

        # Create a model and run it on the input image
        model = FastSAM(args.model_path)
        everything_results = model(
            self.img,
            device=device,
            retina_masks=args.retina,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou)

        # Create a prompt receiver that acts on the selected image
        prompt_process = FastSAMVideoPrompt(np.array(self.img), everything_results, device=device)

        # Return all segments so the user can select one manually
        annotations = prompt_process.everything_prompt()

        # Get output image and contours for each segment
        result, self.segment_contour_list = prompt_process.plot(
            annotations=annotations,
            mask_random_color=args.random_color)

        # Display output image
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(result))

        self.update_canvas_image()
        
        self.update_state(AppState.CONTOUR_SELECT)

        self.print_debug('Segmentation complete.')
        return

    def check_for_segment(self, x, y):
        """ Returns the contour of a segment, if any, at the specified (x, y) coordinate. """

        selected_contour = None

        # Check if we are inside any contour
        for contour in self.segment_contour_list:
            if cv2.pointPolygonTest(contour, (x, y), measureDist=False) >= 0:
                selected_contour = contour
                break

        return selected_contour

    def check_for_contour_point(self, x, y):
        """ Returns the point and index within the currently selected 
            contour, if any, at the specified (x, y) coordinate. """

        selected_point = None
        point_index = -1

        # Check if we clicked on a point
        for i, point in enumerate(self.segment_contour):

            # L2 norm to calculate Euclidean distance between the click and contour point
            if cv2.norm(src1=np.array([[x,y]]), src2=point, normType=cv2.NORM_L2) < self.contour_point_radius:
                selected_point = point
                point_index = i
                
                self.print_debug(f'clicked on point {point_index} at ({selected_point[0][0], selected_point[0][1]})')
                break

        return selected_point, point_index


    def disable_latlong_entry(self):
        """ Grey out labels and disable entry fields for lat/long points. """

        # Grey out entry box labels
        self.label_left     .configure(foreground='grey')
        self.label_right    .configure(foreground='grey')
        self.label_top      .configure(foreground='grey')
        self.label_bottom   .configure(foreground='grey')

        # Disable latlong entry boxes
        self.entry_left     .configure(state='disabled')
        self.entry_right    .configure(state='disabled')
        self.entry_top      .configure(state='disabled')
        self.entry_bottom   .configure(state='disabled')

    def enable_latlong_entry(self):
        """ Revert greying out of labels and enable entry fields for lat/long points. """
        # Revert greying out of entry box labels
        self.label_left     .configure(foreground='black')
        self.label_right    .configure(foreground='black')
        self.label_top      .configure(foreground='black')
        self.label_bottom   .configure(foreground='black')

        # Enable latlong entry boxes
        self.entry_left     .configure(state='enabled')
        self.entry_right    .configure(state='enabled')
        self.entry_top      .configure(state='enabled')
        self.entry_bottom   .configure(state='enabled')


    def poly_edit_changed(self):
        """ Respond to polygon editing being enabled or disabled. """

        editing_enabled = self.allow_edit_poly.get()
        state_msg = 'Polygon editing ' + ('enabled' if editing_enabled else 'disabled')
        self.print_debug(state_msg)

        # Check if we deselected
        if not editing_enabled:
            # Clear polygon point tags
            self.img_canvas.delete(self.contour_points_tag)
            return

        # Disable latlong editing
        self.disable_latlong_entry()

        # Clear any drawn latlong points
        self.img_canvas.delete(self.contour_bounds_tag)

        # Draw the contour points
        for point in self.segment_contour:
            self.img_canvas.create_circle(x=point[0][0], y=point[0][1], r=self.contour_point_radius, fill='blue', outline='white', tags=self.contour_points_tag)

        return

    def latlong_edit_changed(self):
        """ Respond to lat-long editing being enabled or disabled. """

        # Check whether we selected or deselected
        editing_enabled = self.allow_edit_latlong.get()
        state_msg = 'Lat-long editing ' + ('enabled' if editing_enabled else 'disabled')
        self.print_debug(state_msg)

        # Check if we deselected
        if not editing_enabled:
            # Erase bounding points
            self.img_canvas.delete(self.contour_bounds_tag)

            self.disable_latlong_entry()

            return

        # Make sure we've selected a contour
        if self.segment_contour is None:
            self.print_debug('no contour has been selected yet!')
            self.allow_edit_latlong.deselect()
            return

        self.draw_bound_points()

        self.enable_latlong_entry()

    def draw_bound_points(self):
        """ Draw left/right/top/bottom points for the selected contour. """

        # Find left/right/top/bottom bound points
        left    = self.segment_contour[0]
        right   = self.segment_contour[0]
        top     = self.segment_contour[0]
        bottom  = self.segment_contour[0]

        for point in self.segment_contour:
            if point[0][0] < left[0][0]:
                left = point
            elif point[0][0] > right[0][0]:
                right = point
            if point[0][1] < top[0][1]:
                top = point
            elif point[0][1] > bottom[0][1]:
                bottom = point

        # Draw the bound points
        bounds_radius   = 5
        bounds_outline  = 'black'
        bounds_fill     = 'cyan'
        self.img_canvas.create_circle(left  [0][0], left    [0][1], bounds_radius, outline=bounds_outline, fill=bounds_fill, tags=(self.contour_bounds_tag))
        self.img_canvas.create_circle(right [0][0], right   [0][1], bounds_radius, outline=bounds_outline, fill=bounds_fill, tags=(self.contour_bounds_tag))
        self.img_canvas.create_circle(top   [0][0], top     [0][1], bounds_radius, outline=bounds_outline, fill=bounds_fill, tags=(self.contour_bounds_tag))
        self.img_canvas.create_circle(bottom[0][0], bottom  [0][1], bounds_radius, outline=bounds_outline, fill=bounds_fill, tags=(self.contour_bounds_tag))

        # Store bounds for use during exporting
        self.contour_bounds = [left[0][0], right[0][0],   # left_x, right_x,
                                top[0][1], bottom[0][1]]  # top_y, bottom_y

    def xy_to_latlong(self):
        """ Generate a list of latitude/longitude points from the selected contour and bounds input by the user. """
        
        # x/y pixel coordinate boundaries (retrieved from contour)
        left_x      = self.contour_bounds[0]
        right_x     = self.contour_bounds[1]
        top_y       = self.contour_bounds[2]
        bottom_y    = self.contour_bounds[3]

        # lat/long coordinate boundaries (input by user)
        long_left   = float(self.entry_left   .get())
        long_right  = float(self.entry_right  .get())
        lat_top     = float(self.entry_top    .get())
        lat_bottom  = float(self.entry_bottom .get())

        latlong_points = []

        for point in self.segment_contour:
            point_scaled = [
                (point[0][0]-left_x)/(right_x-left_x),
                (point[0][1]-top_y)/(bottom_y-top_y)]

            point_latlong = [
                long_left   + point_scaled[0]*(long_right - long_left),
                lat_top     - point_scaled[1]*(lat_top - lat_bottom)]

            latlong_points.append(point_latlong)

        return latlong_points

    def validate_entry_inputs(self):
        """ Verifies that text entries can be converted to floats. """

        try:
            float(self.entry_left.get())
        except ValueError:
            print('longitude left is invalid.')
            return False

        try:
            float(self.entry_right.get())
        except ValueError:
            print('longitude right is invalid.')
            return False

        try:
            float(self.entry_top.get())
        except ValueError:
            print('latitude top is invalid.')
            return False

        try:
            float(self.entry_bottom.get())
        except ValueError:
            print('latitude bottom is invalid.')
            return False
        
        return True
    
    def export_to_csv(self):
        """ Exports the selected segment's polygon as lat-long points to a CSV file. """

        if not self.validate_entry_inputs():
            return
        
        self.print_debug('exporting!')

        # Generate lat/long points
        latlong_points = self.xy_to_latlong()

        # Write output to a CSV file
        with open(args.output_filename, 'w', encoding='UTF-8', newline='') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(['Latitude', 'Longitude'])
            for point in latlong_points:
                writer.writerow([point[1], point[0]])

        self.print_debug('export complete.')

    def trigger_breakpoint(self):
        breakpoint()

    def __init__(self, root):
        ttk.Frame.__init__(self, root)
        self.pack()

        self.winfo_toplevel().title('APRILab Water Body Detection Tool')

        # Canvas element tags as variables to mitigate typos
        self.contour_tag        = 'contour'
        self.contour_bounds_tag = 'bounds'
        self.contour_points_tag = 'points'

        # UI Constants
        self.contour_point_radius = 5

        # State variables
        self.app_state = AppState(AppState.IMAGE_SELECT)

        self.allow_edit_poly    = tk.BooleanVar(value=False)
        self.allow_edit_latlong = tk.BooleanVar(value=False)

        # High-level frames
        self.frame_menu     = ttk.Frame(self)
        self.frame_status   = ttk.Frame(self)
        self.frame_canvas   = ttk.Frame(self)

        self.frame_menu     .grid(row=0, column=0, rowspan=2,   sticky='nsew') 
        self.frame_status   .grid(row=0, column=1,              sticky='nsew') 
        self.frame_canvas   .grid(row=1, column=1,              sticky='nsew') 

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)


        # Menu frame
        self.btn_sel_img    = ttk.Button        (self.frame_menu, text='Select image file', command=self.browse_files)
        self.btn_segment    = ttk.Button        (self.frame_menu, text='Segment Image',     command=self.segment)
        self.label_edit     = ttk.Label         (self.frame_menu, text='Edit:')
        self.check_polygon  = ttk.Checkbutton   (self.frame_menu, text='Bounding polygon',  command=self.poly_edit_changed,     variable=self.allow_edit_poly)
        self.check_latlong  = ttk.Checkbutton   (self.frame_menu, text='Lat-long points',   command=self.latlong_edit_changed,  variable=self.allow_edit_latlong)
        self.label_left     = ttk.Label         (self.frame_menu, text='Longitude - Left',      foreground='grey')
        self.entry_left     = ttk.Entry         (self.frame_menu, state='disabled')
        self.label_right    = ttk.Label         (self.frame_menu, text='Longitude - Right',     foreground='grey')
        self.entry_right    = ttk.Entry         (self.frame_menu, state='disabled')
        self.label_top      = ttk.Label         (self.frame_menu, text='Latitude - Top',        foreground='grey')
        self.entry_top      = ttk.Entry         (self.frame_menu, state='disabled')
        self.label_bottom   = ttk.Label         (self.frame_menu, text='Longitude - Bottom',    foreground='grey')
        self.entry_bottom   = ttk.Entry         (self.frame_menu, state='disabled')
        self.btn_export     = ttk.Button        (self.frame_menu, text='Export to CSV',     command=self.export_to_csv)
        self.btn_breakpoint = ttk.Button        (self.frame_menu, text='Breakpoint',        command=self.trigger_breakpoint)

        # Status frame
        self.label_status   = ttk.Label         (self.frame_status, text='Open an image')
        self.img_filename   = ttk.Label         (self.frame_status, text='Image file name displays here')

        # Canvas frame
        self.img_canvas     = Canvas            (self.frame_canvas)

        # Gridding - menu frame
        menu_padx        = 5
        menu_pady_short  = 2
        menu_pady_long   = 5
        self.btn_sel_img    .grid(row=0,    column=0, padx=menu_padx, pady=menu_pady_short, sticky='nsew')
        self.btn_segment    .grid(row=1,    column=0, padx=menu_padx, pady=menu_pady_short, sticky='nsew')
        self.label_edit     .grid(row=2,    column=0, padx=menu_padx, pady=menu_pady_long,  sticky='nsew')
        self.check_polygon  .grid(row=3,    column=0, padx=menu_padx, pady=menu_pady_long,  sticky='nsew')
        self.check_latlong  .grid(row=4,    column=0, padx=menu_padx, pady=menu_pady_long,  sticky='nsew')
        self.label_left     .grid(row=5,    column=0, padx=menu_padx, pady=menu_pady_short, sticky='nsew')
        self.entry_left     .grid(row=6,    column=0, padx=menu_padx, pady=menu_pady_long,  sticky='nsew')
        self.label_right    .grid(row=7,    column=0, padx=menu_padx, pady=menu_pady_short, sticky='nsew')
        self.entry_right    .grid(row=8,    column=0, padx=menu_padx, pady=menu_pady_long,  sticky='nsew')
        self.label_top      .grid(row=9,    column=0, padx=menu_padx, pady=menu_pady_short, sticky='nsew')
        self.entry_top      .grid(row=10,   column=0, padx=menu_padx, pady=menu_pady_long,  sticky='nsew')
        self.label_bottom   .grid(row=11,   column=0, padx=menu_padx, pady=menu_pady_short, sticky='nsew')
        self.entry_bottom   .grid(row=12,   column=0, padx=menu_padx, pady=menu_pady_long,  sticky='nsew')
        self.btn_export     .grid(row=13,   column=0, padx=menu_padx, pady=menu_pady_long,  sticky='nsew')
        self.btn_breakpoint .grid(row=14,   column=0, padx=menu_padx, pady=menu_pady_long,  sticky='nsew')

        # Gridding - status frame
        self.label_status   .grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        self.img_filename   .grid(row=0, column=1, padx=10, pady=5, sticky='nse')

        self.frame_status.rowconfigure(0, weight=1)
        self.frame_status.columnconfigure(1, weight=1)

        # Gridding - canvas frame
        self.img_canvas     .grid(row=0, column=0)

        # Event handler binding
        self.img_canvas.bind('<Button-1>', self.on_left_mouse_button)
        self.img_canvas.tag_bind(self.contour_points_tag, '<Button-1>', self.point_drag_start)
        self.img_canvas.tag_bind(self.contour_points_tag, '<B1-Motion>', self.point_drag_motion)
        self.img_canvas.tag_bind(self.contour_points_tag, '<ButtonRelease-1>', self.point_drag_end)

        # Trigger UI update for initial state
        self.update_state(self.app_state)

    

def parse_args():
    """ Use an ArgumentParser to collect arguments into variables. """
    parser = argparse.ArgumentParser()

    # Note that many defaults are taken from FastSAM source code as of the time of cloning their repo.
    parser.add_argument('-d',   '--dark_theme',     type=bool,  default=False,  action=argparse.BooleanOptionalAction)
    parser.add_argument('-v',   '--verbose',        type=bool,  default=False,  action=argparse.BooleanOptionalAction)
    parser.add_argument('-m',   '--model_path',     type=str,   default='./weights/FastSAM-x.pt')
    parser.add_argument(        '--retina',         type=bool,  default=True)
    parser.add_argument(        '--imgsz',          type=int,   default=1024)
    parser.add_argument(        '--conf',           type=float, default=0.4)
    parser.add_argument(        '--iou',            type=float, default=0.9)
    parser.add_argument(        '--random_color',   type=bool,  default=True,   action=argparse.BooleanOptionalAction)
    parser.add_argument('-o',   '--output_filename', type=str,  default='output.csv')

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
