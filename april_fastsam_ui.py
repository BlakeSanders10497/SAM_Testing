""" APRILab Water Body Edge Detection Tool

This script generates a GUI for segmenting an image, selecting a water body, and generating a path
from that water body's bounding polygon. It was created using the Tkinter framework and allows for
live editing of the bounding polygon and input of the water body's latitudinal and longitudinal
boundaries. This data is used to convert the bounding polygon from pixel space to GPS coordinate
space.

Future work will implement drawing the path generated by the path planner on the water body image.

A virtual environment can be created to satisfy FastSAM's requirements. That environment should
be used to run this script. See (https://github.com/CASIA-IVA-Lab/FastSAM/tree/main#installation)
for installation instructions.

Author: Blake Sanders (https://github.com/BlakeSanders10497)

"""

from enum import Enum
import argparse
import csv

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import torch

import cv2
import numpy as np

from PIL import ImageTk, Image
from fastsam import FastSAM, FastSAMVideoPrompt

from utils.path_planner_2023 import *

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
class App(ttk.Frame):
    """ Tkinter application for creating and displaying the water edge detection tool. """

    def __print_debug(self, msg):
        """ Print a debug message if the \'verbose\' argument was passed. """
        if args.verbose:
            print('[Debug]:', msg)

    def __disable_menu_options(self):
        """ Disable all UI options except for opening a new image. """

        self.__btn_segment    .config(state='disabled')
        self.__check_polygon  .config(state='disabled')
        self.__check_latlong  .config(state='disabled')
        self.__entry_left     .config(state='disabled')
        self.__entry_right    .config(state='disabled')
        self.__entry_top      .config(state='disabled')
        self.__entry_bottom   .config(state='disabled')
        self.__btn_export     .config(state='disabled')

        self.__label_edit     .config(foreground='grey')
        self.__label_left     .config(foreground='grey')
        self.__label_right    .config(foreground='grey')
        self.__label_top      .config(foreground='grey')
        self.__label_bottom   .config(foreground='grey')

    def __enable_img_options(self):
        """ Enable UI options related to editing and using the selected water body edge. """

        self.__check_polygon  .config(state='enabled')
        self.__check_latlong  .config(state='enabled')
        self.__btn_export     .config(state='enabled')
        self.__btn_path       .config(state='enabled')

        self.__label_edit     .config(foreground='black')

    def __set_segmenting_state(self, new_state):
        """ Update the state of the segmenting button. Just a wrapper for readability purposes. """

        self.__btn_segment.config(state=new_state)

    def __update_state(self, new_state):
        """ Update the state of the app and perform any resulting changes to the UI. """

        # Update the class member
        self.__app_state = new_state

        # Temp variable to hold status message
        status_msg = ''

        # Perform any actions required by the new state
        if new_state == AppState.IMAGE_SELECT:
            status_msg = 'Open an image'
            self.__disable_menu_options()

        elif new_state == AppState.IMAGE_SEGMENT:
            status_msg = 'Segment the image, or select a new image'
            self.__set_segmenting_state('enabled')

        elif new_state == AppState.CONTOUR_SELECT:
            status_msg = 'Select a contour, or select a new image'
            self.__set_segmenting_state('disabled')

        elif new_state == AppState.CONTOUR_EDIT:
            status_msg = ('Edit the selected contour, select a new contour, '+
                            'export the contour to CSV, or select a new image')
            self.__enable_img_options()

        else:
            self.__print_debug(f'State {new_state} has not been accounted for in update_state!')

        # Update status message
        self.__label_status.configure(text=status_msg)


    def __on_left_mouse_button(self, event):
        """ Event handler to be triggered by pressing the left mouse button. """

        if (self.__app_state == AppState.CONTOUR_SELECT or
            self.__app_state == AppState.CONTOUR_EDIT):
            # Check if we clicked on a segment
            segment_contour_full = self.__check_for_segment(event.x, event.y)
            if segment_contour_full is None or self.__last_clicked_segment is segment_contour_full:
                return


            self.__last_clicked_segment = segment_contour_full

            # Simplify contour before drawing and storing
            # Reference: https://stackoverflow.com/questions/41879315/opencv-visualize-polygonal-curves-extracted-with-cv2-approxpolydp
            epsilon = (2e-3)*cv2.arcLength(segment_contour_full, True)
            self.__segment_contour = cv2.approxPolyDP(segment_contour_full, epsilon, True)

            self.__print_debug(f'Simplified: {self.__segment_contour.shape[0]} points')
            self.__print_debug(f'Original: {segment_contour_full.shape[0]} points')

            # Update the contour on the canvas
            self.__redraw_contour()

            self.__update_state(AppState.CONTOUR_EDIT)

    def __point_drag_start(self, event):
        """ Start dragging a point in the contour that is currently being edited. """
        # Determine which point we clicked on
        self.__clicked_point, self.__clicked_point_index = self.__check_for_contour_point(event.x, event.y)

        # Get its canvas ID
        self.__clicked_point_canvas_id = self.__img_canvas.find_withtag('current')[0]

        # Capture initial position
        self.__prev_x = event.x
        self.__prev_y = event.y
        self.__print_debug(f'Drag start! ({event.x}, {event.y})')

    def __point_drag_motion(self, event):
        """ Drag a point around inside the contour that is currently being edited. """
        dx = event.x - self.__prev_x
        dy = event.y - self.__prev_y

        self.__prev_x = event.x
        self.__prev_y = event.y

        self.__img_canvas.move(self.__clicked_point_canvas_id, dx, dy)
        self.__print_debug(f'Point {self.__clicked_point_index}, id {self.__clicked_point_canvas_id} update: ({event.x}, {event.y})')

    def __point_drag_end(self, event):
        """ Update and redraw the contour that is currently being edited. """
        self.__clicked_point[0][0] = event.x
        self.__clicked_point[0][1] = event.y


        self.__print_debug(f'point {self.__clicked_point_index}, id {self.__clicked_point_canvas_id} released at {event.x}, {event.y}')

        self.__redraw_contour()


    def __redraw_contour(self):
        """ Deletes the old contour, if any, and draws a new contour on the image canvas. """

        # Delete old polygon
        self.__img_canvas.delete(self.__contour_tag)

        # List out contour coordinates like Tk is expecting: [x0, y0, x1, y1, ...]
        contour_coords = []
        for point in self.__segment_contour:
            contour_coords.append(point[0][0])
            contour_coords.append(point[0][1])

        # Draw the new polygon
        self.__img_canvas.create_polygon(contour_coords,
                outline='green', fill='', width=3, tags=(self.__contour_tag))

        # If we are editing the contour, redraw the contour points
        if self.__allow_edit_polygon.get():
            self.__redraw_contour_points()

        # If we are editing latlong bounds, redraw the bound points
        if self.__allow_edit_latlong.get():
            self.__redraw_latlong_points()


    def __redraw_contour_points(self):
        """ Delete old contour points and redraw new ones. """
        # Clear old points
        self.__img_canvas.delete(self.__contour_points_tag)

        # Draw the contour points
        for point in self.__segment_contour:
            self.__img_canvas.create_circle(x=point[0][0], y=point[0][1], r=self.__contour_point_r,
                    fill='blue', outline='white', tags=self.__contour_points_tag)

    def __redraw_latlong_points(self):
        """ Delete old latlong bound points and redraw new ones. """
        # Clear old points
        self.__img_canvas.delete(self.__contour_bounds_tag)

        # Draw new points
        self.__draw_bound_points()

    def __browse_files(self):
        """ Spawn a file browser from which the user can select an image. """
        filename = filedialog.askopenfilename(initialdir='.',
                                                title='Select a File',
                                                filetypes= (('Images', '.tif .jpg .png'),
                                                            ('All files', '*.*'))) 
        # Check if the user cancelled instead of selecting a file
        if not filename:
            self.__print_debug('no file selected.')
            return

        self.__img_filename.configure(text='Image: '+ filename[filename.rfind('/')+1:])
        with Image.open(filename) as img:
            # Update the image
            self.__img = img
            self.__tk_img = ImageTk.PhotoImage(img)

            # Clear all drawn shapes from canvas
            self.__img_canvas.delete('all')

            # Draw the new image
            self.__update_canvas_image()

            self.__update_state(AppState.IMAGE_SEGMENT)

    def __update_canvas_image(self):
        """ Updates the image to display on the Canvas while resizing the Canvas. """
        # Resize canvas to match the new image
        self.__img_canvas.configure(width=self.__tk_img.width(), height=self.__tk_img.height())

        # Add image to canvas
        self.__img_canvas.create_image(0, 0, anchor="nw", image=self.__tk_img)

    def __segment(self):
        """ Begin segmentation on the selected image. """

        # Check if we've selected an image yet
        if not self.__img:
            print("No image has been selected yet! Unable to segment.")
            return

        # Pick a device to run segmentation on
        device = torch.device(
            'cuda'  if torch.cuda.is_available() else
            'mps'   if torch.backends.mps.is_available() else
            'cpu')

        self.__print_debug('Segmenting!')

        # Create a model and run it on the input image
        model = FastSAM(args.model_path)
        everything_results = model(
            self.__img,
            device=device,
            retina_masks=args.retina,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou)

        # Create a prompt receiver that acts on the selected image
        prompt_process = FastSAMVideoPrompt(np.array(self.__img), everything_results, device=device)

        # Return all segments so the user can select one manually
        annotations = prompt_process.everything_prompt()

        # Get output image and contours for each segment
        result, self.__segment_contour_list = prompt_process.plot(
            annotations=annotations,
            mask_random_color=args.random_color)

        num_segments = len(self.__segment_contour_list)

        print(f'{"Warning: " if num_segments == 0 else ""}Segmentation found {num_segments} segments.')

        # Display output image
        self.__tk_img = ImageTk.PhotoImage(Image.fromarray(result))

        self.__update_canvas_image()

        self.__update_state(AppState.CONTOUR_SELECT)

        self.__print_debug('Segmentation complete.')
        return

    def __check_for_segment(self, x, y):
        """ Returns the contour of a segment, if any, at the specified (x, y) coordinate. """

        selected_contour = None

        # Check if we are inside any contour
        for contour in self.__segment_contour_list:
            if cv2.pointPolygonTest(contour, (x, y), measureDist=False) >= 0:
                selected_contour = contour
                break

        return selected_contour

    def __check_for_contour_point(self, x, y):
        """ Returns the point and index within the currently selected 
            contour, if any, at the specified (x, y) coordinate. """

        selected_point = None
        point_index = -1

        # Check if we clicked on a point
        for i, point in enumerate(self.__segment_contour):

            # Points can be clicked on the edge where they are outside the point radius,
            #   so a fudge factor is used in the radius comparison to extend it.
            radius_fudge = 3

            # L2 norm to calculate Euclidean distance between the click and contour point
            if cv2.norm(src1=np.array([[x,y]]), src2=point, normType=cv2.NORM_L2) < (self.__contour_point_r + radius_fudge):
                selected_point = point
                point_index = i
                
                self.__print_debug(f'clicked on point {point_index} at ({selected_point[0][0], selected_point[0][1]})')
                break

        return selected_point, point_index

    def __disable_latlong_entry(self):
        """ Disables and clears latlong checkbox, grey out labels, and disable entry fields for lat/long points. """

        # Clear checkbox
        self.__allow_edit_latlong.set(0)

        # Grey out entry box labels
        self.__label_left     .configure(foreground='grey')
        self.__label_right    .configure(foreground='grey')
        self.__label_top      .configure(foreground='grey')
        self.__label_bottom   .configure(foreground='grey')

        # Disable latlong entry boxes
        self.__entry_left     .configure(state='disabled')
        self.__entry_right    .configure(state='disabled')
        self.__entry_top      .configure(state='disabled')
        self.__entry_bottom   .configure(state='disabled')

    def __enable_latlong_entry(self):
        """ Enable polygon checkbox, revert greying out of labels, and enable entry fields for lat/long points. """

        # Enable checkbox
        self.__check_latlong.configure(state='enabled')

        # Revert greying out of entry box labels
        self.__label_left     .configure(foreground='black')
        self.__label_right    .configure(foreground='black')
        self.__label_top      .configure(foreground='black')
        self.__label_bottom   .configure(foreground='black')

        # Enable latlong entry boxes
        self.__entry_left     .configure(state='enabled')
        self.__entry_right    .configure(state='enabled')
        self.__entry_top      .configure(state='enabled')
        self.__entry_bottom   .configure(state='enabled')


    def __polygon_edit_changed(self):
        """ Respond to polygon editing being enabled or disabled. """

        editing_enabled = self.__allow_edit_polygon.get()
        state_msg = 'Polygon editing ' + ('enabled' if editing_enabled else 'disabled')
        self.__print_debug(state_msg)

        # Check if we deselected
        if not editing_enabled:
            # Clear polygon point tags
            self.__img_canvas.delete(self.__contour_points_tag)

            return

        self.__disable_latlong_entry()

        # Clear any drawn latlong points
        self.__img_canvas.delete(self.__contour_bounds_tag)

        # Draw the contour points
        for point in self.__segment_contour:
            self.__img_canvas.create_circle(x=point[0][0], y=point[0][1], r=self.__contour_point_r,
                    fill='blue', outline='white', tags=self.__contour_points_tag)

        return

    def __latlong_edit_changed(self):
        """ Respond to latlong editing being enabled or disabled. """

        # Check whether we selected or deselected
        editing_enabled = self.__allow_edit_latlong.get()
        state_msg = 'Lat-long editing ' + ('enabled' if editing_enabled else 'disabled')
        self.__print_debug(state_msg)

        # Check if we deselected
        if not editing_enabled:
            # Erase bounding points
            self.__img_canvas.delete(self.__contour_bounds_tag)

            self.__disable_latlong_entry()

            return

        # Clear any drawn polygon points
        self.__img_canvas.delete(self.__contour_points_tag)

        # Clear the polygon editing checkbox
        self.__allow_edit_polygon.set(0)

        self.__draw_bound_points()

        self.__enable_latlong_entry()

    def __draw_bound_points(self):
        """ Draw left/right/top/bottom points for the selected contour. """

        # Find left/right/top/bottom bound points
        left    = self.__segment_contour[0]
        right   = self.__segment_contour[0]
        top     = self.__segment_contour[0]
        bottom  = self.__segment_contour[0]

        for point in self.__segment_contour:
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
        self.__img_canvas.create_circle(left  [0][0], left    [0][1], bounds_radius, outline=bounds_outline, fill=bounds_fill, tags=(self.__contour_bounds_tag))
        self.__img_canvas.create_circle(right [0][0], right   [0][1], bounds_radius, outline=bounds_outline, fill=bounds_fill, tags=(self.__contour_bounds_tag))
        self.__img_canvas.create_circle(top   [0][0], top     [0][1], bounds_radius, outline=bounds_outline, fill=bounds_fill, tags=(self.__contour_bounds_tag))
        self.__img_canvas.create_circle(bottom[0][0], bottom  [0][1], bounds_radius, outline=bounds_outline, fill=bounds_fill, tags=(self.__contour_bounds_tag))

        # Store bounds for use during exporting
        self.__contour_bounds = [left[0][0], right[0][0],   # left_x, right_x,
                                top[0][1], bottom[0][1]]  # top_y, bottom_y

    def __xy_to_latlong(self):
        """ Generate a list of latitude/longitude points from the selected contour and bounds input by the user. """
        
        # x/y pixel coordinate boundaries (retrieved from contour)
        left_x      = self.__contour_bounds[0]
        right_x     = self.__contour_bounds[1]
        top_y       = self.__contour_bounds[2]
        bottom_y    = self.__contour_bounds[3]

        # lat/long coordinate boundaries (input by user)
        long_left   = float(self.__entry_left   .get())
        long_right  = float(self.__entry_right  .get())
        lat_top     = float(self.__entry_top    .get())
        lat_bottom  = float(self.__entry_bottom .get())

        latlong_points = []

        for point in self.__segment_contour:
            point_scaled = [
                (point[0][0]-left_x)/(right_x-left_x),
                (point[0][1]-top_y)/(bottom_y-top_y)]

            point_latlong = [
                long_left   + point_scaled[0]*(long_right - long_left),
                lat_top     - point_scaled[1]*(lat_top - lat_bottom)]

            latlong_points.append(point_latlong)

        return latlong_points

    def __validate_entry_inputs(self):
        """ Verifies that text entries can be converted to floats. """

        try:
            float(self.__entry_left.get())
        except ValueError:
            print('longitude left is invalid.')
            return False

        try:
            float(self.__entry_right.get())
        except ValueError:
            print('longitude right is invalid.')
            return False

        try:
            float(self.__entry_top.get())
        except ValueError:
            print('latitude top is invalid.')
            return False

        try:
            float(self.__entry_bottom.get())
        except ValueError:
            print('latitude bottom is invalid.')
            return False

        return True

    def __export_to_csv(self):
        """ Exports the selected segment's polygon as lat-long points to a CSV file. """

        if not self.__validate_entry_inputs():
            return

        self.__print_debug('exporting!')

        # Generate lat/long points
        latlong_points = self.__xy_to_latlong()

        # Write output to a CSV file
        with open(args.output_filename, 'w', encoding='UTF-8', newline='') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(['Latitude', 'Longitude'])
            for point in latlong_points:
                writer.writerow([point[1], point[0]])

        print(f'Export to {args.output_filename} complete.')

    def __generate_path(self):
        """ Generates and renders a path returned by the path planner. """

        # Create separate lists of x and y coordinates from bounding contour points
        x_coords = [point[0][0] for point in self.__segment_contour]
        y_coords = [point[0][1] for point in self.__segment_contour]

        # Generate a path
        waypoint_coords = fullPath(x_coords, y_coords)

        # Draw the path
        self.__img_canvas.create_line(waypoint_coords, fill='red', width=2)

    def __init__(self, root):
        ttk.Frame.__init__(self, root)
        self.pack()

        self.winfo_toplevel().title('APRILab Water Body Detection Tool')

        # Canvas element tags as variables to mitigate typos
        self.__contour_tag        = 'contour'
        self.__contour_bounds_tag = 'bounds'
        self.__contour_points_tag = 'points'

        # UI Constants
        self.__contour_point_r = 5

        # State variables
        self.__app_state = AppState(AppState.IMAGE_SELECT)

        self.__allow_edit_polygon   = tk.BooleanVar(value=False)
        self.__allow_edit_latlong   = tk.BooleanVar(value=False)

        self.__last_clicked_segment = None

        # High-level frames
        self.__frame_menu     = ttk.Frame(self)
        self.__frame_status   = ttk.Frame(self)
        self.__frame_canvas   = ttk.Frame(self)

        self.__frame_menu     .grid(row=0, column=0, rowspan=2,   sticky='nsew') 
        self.__frame_status   .grid(row=0, column=1,              sticky='nsew') 
        self.__frame_canvas   .grid(row=1, column=1,              sticky='nsew') 

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)


        # Menu frame
        self.__btn_sel_img    = ttk.Button      (self.__frame_menu, text='Select image file', command=self.__browse_files)
        self.__btn_segment    = ttk.Button      (self.__frame_menu, text='Segment Image',     command=self.__segment)
        self.__label_edit     = ttk.Label       (self.__frame_menu, text='Edit:')
        self.__check_polygon  = ttk.Checkbutton (self.__frame_menu, text='Bounding polygon',  command=self.__polygon_edit_changed,     variable=self.__allow_edit_polygon)
        self.__check_latlong  = ttk.Checkbutton (self.__frame_menu, text='Lat-long points',   command=self.__latlong_edit_changed,  variable=self.__allow_edit_latlong)
        self.__label_left     = ttk.Label       (self.__frame_menu, text='Longitude - Left',      foreground='grey')
        self.__entry_left     = ttk.Entry       (self.__frame_menu, state='disabled')
        self.__label_right    = ttk.Label       (self.__frame_menu, text='Longitude - Right',     foreground='grey')
        self.__entry_right    = ttk.Entry       (self.__frame_menu, state='disabled')
        self.__label_top      = ttk.Label       (self.__frame_menu, text='Latitude - Top',        foreground='grey')
        self.__entry_top      = ttk.Entry       (self.__frame_menu, state='disabled')
        self.__label_bottom   = ttk.Label       (self.__frame_menu, text='Longitude - Bottom',    foreground='grey')
        self.__entry_bottom   = ttk.Entry       (self.__frame_menu, state='disabled')
        self.__btn_export     = ttk.Button      (self.__frame_menu, text='Export to CSV',     command=self.__export_to_csv)
        self.__btn_path       = ttk.Button      (self.__frame_menu, text='Generate Path',     command=self.__generate_path, state='disabled') 

        # Status frame
        self.__label_status   = ttk.Label       (self.__frame_status, text='Open an image')
        self.__img_filename   = ttk.Label       (self.__frame_status, text='Image: none')

        # Canvas frame
        self.__img_canvas     = tk.Canvas       (self.__frame_canvas)

        # Gridding - menu frame
        menu_padx        = 5
        menu_pady_short  = 2
        menu_pady_long   = 5
        self.__btn_sel_img    .grid(row=0,    column=0, padx=menu_padx, pady=menu_pady_short, sticky='nsew')
        self.__btn_segment    .grid(row=1,    column=0, padx=menu_padx, pady=menu_pady_short, sticky='nsew')
        self.__label_edit     .grid(row=2,    column=0, padx=menu_padx, pady=menu_pady_long,  sticky='nsew')
        self.__check_polygon  .grid(row=3,    column=0, padx=menu_padx, pady=menu_pady_long,  sticky='nsew')
        self.__check_latlong  .grid(row=4,    column=0, padx=menu_padx, pady=menu_pady_long,  sticky='nsew')
        self.__label_left     .grid(row=5,    column=0, padx=menu_padx, pady=menu_pady_short, sticky='nsew')
        self.__entry_left     .grid(row=6,    column=0, padx=menu_padx, pady=menu_pady_long,  sticky='nsew')
        self.__label_right    .grid(row=7,    column=0, padx=menu_padx, pady=menu_pady_short, sticky='nsew')
        self.__entry_right    .grid(row=8,    column=0, padx=menu_padx, pady=menu_pady_long,  sticky='nsew')
        self.__label_top      .grid(row=9,    column=0, padx=menu_padx, pady=menu_pady_short, sticky='nsew')
        self.__entry_top      .grid(row=10,   column=0, padx=menu_padx, pady=menu_pady_long,  sticky='nsew')
        self.__label_bottom   .grid(row=11,   column=0, padx=menu_padx, pady=menu_pady_short, sticky='nsew')
        self.__entry_bottom   .grid(row=12,   column=0, padx=menu_padx, pady=menu_pady_long,  sticky='nsew')
        self.__btn_export     .grid(row=13,   column=0, padx=menu_padx, pady=menu_pady_long,  sticky='nsew')
        self.__btn_path       .grid(row=14,   column=0, padx=menu_padx, pady=menu_pady_long,  sticky='nsew')

        # Gridding - status frame
        self.__label_status   .grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        self.__img_filename   .grid(row=0, column=1, padx=10, pady=5, sticky='nse')

        self.__frame_status.rowconfigure(0, weight=1)
        self.__frame_status.columnconfigure(1, weight=1)

        # Gridding - canvas frame
        self.__img_canvas     .grid(row=0, column=0)

        # Event handler binding
        self.__img_canvas.bind('<Button-1>', self.__on_left_mouse_button)
        self.__img_canvas.tag_bind(self.__contour_points_tag, '<Button-1>', self.__point_drag_start)
        self.__img_canvas.tag_bind(self.__contour_points_tag, '<B1-Motion>', self.__point_drag_motion)
        self.__img_canvas.tag_bind(self.__contour_points_tag, '<ButtonRelease-1>', self.__point_drag_end)

        # Trigger UI update for initial state
        self.__update_state(self.__app_state)


def parse_args():
    """ Use an ArgumentParser to collect arguments into variables. """
    parser = argparse.ArgumentParser()

    # Note that many defaults are taken from FastSAM source code as of the time of cloning their repo.
    #parser.add_argument('-d',   '--dark_theme',     type=bool,  default=False,  action=argparse.BooleanOptionalAction)
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
    """ Launches the GUI. """
    root = tk.Tk()
    root.minsize(400, 200)

    #if(args.dark_theme):
    #    root.tk.call('lappend', 'auto_path', '/home/blake/Documents/GitHub/FastSAM/FastSAMUI/themes/awthemes-10.4.0')
    #    root.tk.call('package', 'require', 'awdark')
    #    s = ttk.Style()
    #    s.theme_use('awdark')
    #    root.configure(bg='#33393b')

    app = App(root)
    root.mainloop()
  
if __name__=="__main__":
    args = parse_args()
    main(args)
