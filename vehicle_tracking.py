import statistics
from collections import deque

import cv2
import numpy as np
from scipy.ndimage.measurements import label

from helper import convert_colour, grad_hist, bin_spatial, color_hist

class Vehicle(object):
    def __init__(self, vid):
        self.vid = vid
        self.ndets = 10 # number of detections to cache
        self.buffer_depth = 20 # the buffer depth for smoothing

        # Number of times vehicle was detected/not detected over past n frames
        self.dbuffer = deque(self.ndets*[np.nan], self.ndets)
        self.detected = True # indicates vehicle was detected in current frame
        self.display = False # indicates vehicle should be displayed this frame

        # x position of last n detections
        self.xbuffer = deque(self.buffer_depth*[np.nan], self.buffer_depth)
        self.bestx = None # running average of buffered x positions
        # y position of last n detections
        self.ybuffer = deque(self.buffer_depth*[np.nan], self.buffer_depth)
        self.besty = None # running average of buffered y positions
        # width of last n detections
        self.wbuffer = deque(self.buffer_depth*[np.nan], self.buffer_depth)
        self.bestw = None # running average of buffered widths
        # height of last n detections
        self.hbuffer = deque(self.buffer_depth*[np.nan], self.buffer_depth)
        self.besth = None # running average of buffered heights

    def new_position(self, vehicle_number, labels):
        # Find pixels with each vehicle_number label value
        xy_pixels = (labels[0] == vehicle_number).nonzero()
        xpixels = np.array(xy_pixels[1]) # Seperate x and y values
        ypixels = np.array(xy_pixels[0])
        self.add_position(xpixels, ypixels) # Add position to vehicle

    def add_position(self, xpixels, ypixels):
        # Append to detection buffers
        self.xbuffer.append(np.int64(statistics.median(xpixels)))
        self.ybuffer.append(np.int64(statistics.median(ypixels)))
        self.wbuffer.append(np.max(xpixels) - np.min(xpixels))
        self.hbuffer.append(np.max(ypixels) - np.min(ypixels))
        # Calculate best fit
        self.bestx = np.int64(np.nanmean(self.xbuffer, axis=0))
        self.besty = np.int64(np.nanmean(self.ybuffer, axis=0))
        self.bestw = np.int64(np.nanmean(self.wbuffer, axis=0))
        self.besth = np.int64(np.nanmean(self.hbuffer, axis=0))

    def compare_positions(self, xpixels, ypixels, compare_results=True):
        """Returns a positive match if there is more than a thirty percent
        overlap in both x and y between the two regions.
        """
        rangex, rangey = self.xy_pixels()
        resultx = len(list(set(rangex) & set(xpixels))) / (len(rangex) + len(xpixels))
        resulty = len(list(set(rangey) & set(ypixels))) / (len(rangey) + len(ypixels))
        if compare_results:
            if resultx > 0.3 and resulty > 0.3:
                return True
            return False
        else:
            return resultx + resulty

    def xy_pixels(self):
        """Returns a range of x and y coordinates using best fit parameters.
        """
        rangex = range(np.int64(self.bestx - self.bestw/2),
                       np.int64(self.bestx + np.ceil(self.bestw/2) + 1))
        rangey = range(np.int64(self.besty - self.besth/2),
                       np.int64(self.besty + np.ceil(self.besth/2) + 1))
        return rangex, rangey

    def bbox(self, coord=None):
        """Returns a bounding box based on min/max x and y.
        """
        if coord == 0:
            return ((self.bestx-np.int64(self.bestw/2),
                     self.besty-np.int64(self.besth/2)))
        elif coord == 1:
            return ((self.bestx+np.int64(self.bestw/2),
                     self.besty+np.int64(self.besth/2)))

    def report(self):
        print('\nID={} x={}  y={}  width={}  height={}'.format(
            self.vid, self.bestx, self.besty, self.bestw, self.besth), end='')
        print('\n\tDetection status: {} | {}\n\t{}'.format(
            ['Detected' if self.detected else 'NOT detected'][0],
            ['Displaying' if self.display else 'NOT displaying'][0],
            self.dbuffer), end='')

class VehicleTracking(object):
    def __init__(self, svc, scaler,
                 spatial_params, hoc_params, hog_params, search_params):
        # Input parameters
        self.svc = svc
        self.scaler = scaler
        self.spatial_params = spatial_params
        self.hoc_params = hoc_params
        self.hog_params = hog_params
        self.search_params = search_params

        # Vehicle object detections/tracking
        self.detected_vehicles = [] # a list of detected vehicle objects
        self.tracked_vehicles = [] # a list of vehicle objects being tracked

        # Image arrays
        self.frame_image = None # the original input image
        self.draw_image = None # the final output image
        self.window_image = None # illustration of current frame's window search
        self.heatmap = None # a heatmap of predicted detections
        self.vehicle_labels = None # (a labeled image, number of labels in image)

        # Misc. variables
        self.processing_video = False
        self.debug_mode = False
        self.vid_counter = 0

    def pipeline(self, image):
        self.frame_image = image
        self.init() # initialise instance variables
        self.detect() # perform window search and predict vehicle detections
        self.track() # update vehicle tracking objects
        self.draw() # draw bounding boxes around tracked vehicles
        return self.draw_image

    def init(self):
        self.detected_vehicles = []
        self.heatmap = None
        self.window_image = np.copy(self.frame_image)
        self.draw_image = np.copy(self.frame_image)

    def detect(self):
        # iterate through search parameters to search for vehicles
        for idx in range(len(self.search_params['ystart'])):
            self.find_vehicles(self.search_params['ystart'][idx],
                               self.search_params['ystop'][idx],
                               self.search_params['scale'][idx])
        # threshold and label heatmap
        self.apply_heatmap_threshold()
        self.label_heatmap()
        # iterate through all new detections
        for vehicle_number in range(1, self.vehicle_labels[1] + 1):
            vhcl = Vehicle(self.vid_counter)
            self.vid_counter += 1
            vhcl.new_position(vehicle_number, self.vehicle_labels)
            self.detected_vehicles.append(vhcl)
            if self.debug_mode:
                print('\nAdding new vehicle object {}'.format(vhcl.vid), end='')

    def track(self):
        if self.processing_video:
            add_to_tracked = list(self.detected_vehicles) # list of new objects to track
            remove_from_tracked = [] # list of objects to stop tracking

            for vhcl in self.tracked_vehicles:
                vhcl.detected = False # reset detection status for tracked

            # iterate through detected vehicles to match vehicle objects
            # if matches with more than one, merge with strongest match
            for dvhcl in self.detected_vehicles:
                strongest_match = None
                value_to_beat = 0.6
                dv_rangex, dv_rangey = dvhcl.xy_pixels()
                for tvhcl in self.tracked_vehicles:
                    result = tvhcl.compare_positions(dv_rangex, dv_rangey, False)
                    if result > value_to_beat:
                        strongest_match = tvhcl
                        value_to_beat = result
                # If match found and has not already been matched
                if strongest_match is not None and not strongest_match.detected:
                    strongest_match.add_position(dv_rangex, dv_rangey)
                    strongest_match.detected = True
                    add_to_tracked.remove(dvhcl)
                    if self.debug_mode:
                        print('\nMerging {} with {}'.format(
                            dvhcl.vid, strongest_match.vid), end='')

            # add new unique detections to tracked
            self.tracked_vehicles.extend(add_to_tracked)

            # Iterate through all tracked vehicles to update tracking status
            for vhcl in self.tracked_vehicles:
                # Update detection buffer
                if vhcl.detected:
                    vhcl.dbuffer.append(1)
                else:
                    vhcl.dbuffer.append(0)
                dbuffer = list(vhcl.dbuffer)

                # Don't display if first detection (could be false positive)
                if dbuffer[-1] == 1 and dbuffer[:vhcl.ndets-1] == (vhcl.ndets-1)*[np.nan]:
                    vhcl.display = False
                # Don't display if second detection (could be false positive)
                elif dbuffer[-2:] == [1, 1] and dbuffer[:vhcl.ndets-2] == (vhcl.ndets-2)*[np.nan]:
                    vhcl.display = False
                # Don't display if less than 80% cached positive detections
                elif np.nanmean(dbuffer, axis=0) < 0.8:
                    vhcl.display = False
                # Otherwise, display vehicle object
                else:
                    vhcl.display = True

                # Delete false positives
                if dbuffer[-2:] == [1, 0] and dbuffer[:vhcl.ndets-2] == (vhcl.ndets-2)*[np.nan]:
                    remove_from_tracked.append(vhcl)
                    if self.debug_mode:
                        print('\nID {} has been deleted (false positive)'.format(
                            vhcl.vid), end='')
                # Delete if no detected frames left in buffer
                elif dbuffer.count(0) == vhcl.ndets:
                    remove_from_tracked.append(vhcl)
                    if self.debug_mode:
                        print('\nID {} has been deleted'.format(vhcl.vid),
                              end='.')

            # Remove vehicles marked for removal
            for vhcl in remove_from_tracked:
                self.tracked_vehicles.remove(vhcl)

        else: # processing single image, track all detections
            self.tracked_vehicles = self.detected_vehicles
            for vhcl in self.detected_vehicles:
                vhcl.display = True

    def draw(self):
        vehicle_count = 0
        # Iterate through all detected vehicles
        for vhcl in self.tracked_vehicles:
            if vhcl.display:
                cv2.rectangle(self.draw_image,
                              vhcl.bbox(0), vhcl.bbox(1),
                              (0, 0, 255), 6)
                vehicle_count += 1
            if self.debug_mode:
                vhcl.report()
        self.text('Tracking '+str(vehicle_count)+' vehicles', 100)

    def text(self, text, ypos):
        font = cv2.FONT_HERSHEY_SIMPLEX
        textpos = (10, ypos) # coordinates for bottom left of text
        fontscale = 2
        fontcolor = (255, 255, 255)
        linetype = 8
        return cv2.putText(self.draw_image, text, textpos, font, fontscale,
                           fontcolor, linetype)

    def label_heatmap(self):
        self.vehicle_labels = label(self.heatmap)

    def apply_heatmap_threshold(self):
         # Zero pixels below threshold
        self.heatmap[self.heatmap < self.search_params['threshold']] = 0

    def find_vehicles(self, ystart, ystop, scale):
        """A single function to extract features using hog sub-sampling and
        make predictions.
        scale is used to reduce (by increasing from positive 1) the resolution of
        the image. For a scale of 1, the windows cover 64x64 pixels of original
        image. For a scale of 2, the windows cover 128x128 pixels of original image.
        """
        windows = []

        hog_params = self.hog_params.copy()
        hog_conv = hog_params.pop('conv')
        hog_params.pop('channels')

        # Make a heatmap of zeros to start with
        image_cpy = np.copy(self.frame_image)
        heatmap = np.zeros_like(image_cpy[:, :, 0])

        # Crop the image frame to include only the desired search area
        image_cpy = np.copy(self.frame_image.astype(np.float32) / 255)
        search_image = image_cpy[ystart:ystop, :, :]

        # Convert colour if provided
        if hog_conv is not None:
            hog_image = convert_colour(search_image, hog_conv)
        else:
            hog_image = search_image

        # Increase search window (by reducing image resolution)
        if scale != 1:
            imshape = search_image.shape
            search_image = cv2.resize(search_image, (np.int64(imshape[1]/scale),
                                                     np.int64(imshape[0]/scale)))
            hog_image = cv2.resize(hog_image, (np.int64(imshape[1]/scale),
                                               np.int64(imshape[0]/scale)))

        # Compute individual channel HOG features for the entire image
        ch1 = hog_image[:, :, 0]
        ch2 = hog_image[:, :, 1]
        ch3 = hog_image[:, :, 2]

        # Define block size
        nxblocks = (ch1.shape[1] // hog_params['pix_per_cell']) - hog_params['cell_per_block'] + 1
        nyblocks = (ch1.shape[0] // hog_params['pix_per_cell']) - hog_params['cell_per_block'] + 1

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window_size = 64
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nblocks_per_window = ((window_size // hog_params['pix_per_cell']) -
                              hog_params['cell_per_block'] + 1)
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

        # calculate hog over image for each colour channel
        hog1 = grad_hist(ch1, feature_vec=False, **hog_params)
        hog2 = grad_hist(ch2, feature_vec=False, **hog_params)
        hog3 = grad_hist(ch3, feature_vec=False, **hog_params)

        # for each cell, extract hog features, then calculate colour features
        for xstep in range(nxsteps):
            for ystep in range(nysteps):
                xpos = xstep*cells_per_step
                ypos = ystep*cells_per_step

                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window,
                                 xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window,
                                 xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window,
                                 xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*hog_params['pix_per_cell']
                ytop = ypos*hog_params['pix_per_cell']

                # Extract the image patch for color features
                sub_image = cv2.resize(search_image[ytop:ytop+window_size,
                                                    xleft:xleft+window_size], (64, 64))
                spatial_features = bin_spatial(sub_image, **self.spatial_params)
                hist_features = color_hist(sub_image, **self.hoc_params)

                # Scale features and make a prediction
                test_features = self.scaler.transform(np.hstack(
                    (spatial_features, hist_features, hog_features)).reshape(1, -1))
                test_prediction = self.svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int64(xleft*scale)
                    ytop_draw = np.int64(ytop*scale)
                    win_draw = np.int64(window_size*scale)
                    window = ((xbox_left, ytop_draw+ystart),
                              (xbox_left+win_draw, ytop_draw+win_draw+ystart))
                    self.window_image = cv2.rectangle(self.window_image,
                                                      window[0], window[1],
                                                      (0, 0, 255), 4)
                    windows.append(window)
                    heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1

        if self.debug_mode:
            print('Found', len(windows), 'windows', end='. ')

        if self.heatmap == None:
            self.heatmap = heatmap
        else:
            self.heatmap = cv2.addWeighted(self.heatmap, 1, heatmap, 1, 0)

        self.heatmap = cv2.GaussianBlur(self.heatmap, (5, 5), 0)
