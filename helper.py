import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt


def data_look(car_list, notcar_list):
    """A function to return some characteristics of the dataset.
    """
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    # Define a key "image_shape" and store the test image shape 3-tuple
    img = mpimg.imread(car_list[0])
    data_dict["image_shape"] = img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = img.dtype
    # Return data_dict
    return data_dict

def bin_spatial(image, conv=None, size=(32, 32)):
    """A function to compute binned color features. Use size to reduce the
    resolution of the image and visualise to return the resized image.
    """
    # Convert colour space if specified
    if conv is not None:
        feature_image = convert_colour(image, conv)
    else:
        feature_image = np.copy(image)
    ch0_bins = cv2.resize(feature_image[:,:,0], size, cv2.INTER_LINEAR)
    ch1_bins = cv2.resize(feature_image[:,:,1], size, cv2.INTER_LINEAR)
    ch2_bins = cv2.resize(feature_image[:,:,2], size, cv2.INTER_LINEAR)
    features = np.hstack((ch0_bins, ch1_bins, ch2_bins))
    return features.ravel()

def color_hist(image, conv=None, bins=32):
    """A function to compute color histogram features.
    """
    # Convert colour space if specified
    if conv is not None:
        feature_image = convert_colour(image, conv)
    else:
        feature_image = np.copy(image)
    # Compute the histogram of the color channels separately
    ch0_hist = np.histogram(feature_image[:,:,0], bins=bins)
    ch1_hist = np.histogram(feature_image[:,:,1], bins=bins)
    ch2_hist = np.histogram(feature_image[:,:,2], bins=bins)
    # Concatenate the histograms into a single feature vector
    return np.concatenate((ch0_hist[0], ch1_hist[0], ch2_hist[0]))

def grad_hist(image, visualise=False, feature_vec=True, orientations=9,
                pix_per_cell=8, cell_per_block=3):
    """A function to compute gradient histogram features.
    """
    return hog(image, orientations=orientations, visualise=visualise,
                pixels_per_cell=(pix_per_cell, pix_per_cell),
                cells_per_block=(cell_per_block, cell_per_block),
                feature_vector=feature_vec, transform_sqrt=False,
                block_norm="L2")

def extract_features(image_files, spatial_params, hoc_params, hog_params):
    """A function to extract features from a list of images using,
        - spatial binning
        - histogram of colours (hoc)
        - histogram of gradients (hog)
    """
    features = []

    if hog_params is not None:
        hog_params = hog_params.copy()
        hog_conv = hog_params.pop('conv')
        hog_channels = hog_params.pop('channels')

    for image_file in image_files:
        file_features = []
        image = mpimg.imread(image_file)
        feature_image = np.copy(image)

        if spatial_params is not None:
            spatial_features = bin_spatial(feature_image, **spatial_params)
            file_features.append(spatial_features)

        if hoc_params is not None:
            hoc_features = color_hist(feature_image, **hoc_params)
            file_features.append(hoc_features)

        if hog_params is not None:
            if hog_conv is not None:
                hog_image = convert_colour(feature_image, hog_conv)
            else:
                hog_image = feature_image

            hog_features = []
            for ch in hog_channels:
                hog_features.append(grad_hist(hog_image[:,:,ch], **hog_params))
            hog_features = np.ravel(hog_features)
            file_features.append(hog_features)

        features.append(np.concatenate(file_features))
    return features

def extract_features_single(image, visualise_hog, spatial_params, hoc_params, hog_params):
    """A function to extract features from an images using,
        - spatial binning
        - histogram of colours (hoc)
        - histogram of gradients (hog)
    """
    features = []

    if hog_params is not None:
        hog_params = hog_params.copy()
        hog_conv = hog_params.pop('conv')
        hog_channels = hog_params.pop('channels')

    feature_image = np.copy(image)

    if spatial_params is not None:
        spatial_features = bin_spatial(feature_image, **spatial_params)
        features.append(spatial_features)

    if hoc_params is not None:
        hoc_features = color_hist(feature_image, **hoc_params)
        features.append(hoc_features)

    if hog_params is not None:
        if hog_conv is not None:
            hog_image = convert_colour(feature_image, hog_conv)
        else:
            hog_image = feature_image

        hog_features = []
        for ch in hog_channels:
            if visualise_hog:
                hog_features, hog_image = grad_hist(hog_image[:,:,ch], visualise_hog, **hog_params)
            else:
                hog_features.append(grad_hist(hog_image[:,:,ch], visualise_hog, **hog_params))
        hog_features = np.ravel(hog_features)
        features.append(hog_features)

    if visualise_hog:
        return np.concatenate(features), hog_image
    return np.concatenate(features)

def slide_window(image, x_start_stop=(None, None), y_start_stop=(None, None),
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5), show_stats=False):
    """A function that takes an image, start and stop positions in both x and y,
    a window size (x and y dimensions), and overlap fraction (for both x and y).
    """
    # If x and/or y start/stop positions not defined, set to image size
    if None in x_start_stop:
        x_start_stop = (0, image.shape[1])
    if None in y_start_stop:
        y_start_stop = (0, image.shape[0])
    # Span of the region to be searched
    span_x = x_start_stop[1] - x_start_stop[0]
    span_y = y_start_stop[1] - y_start_stop[0]
    # Number of pixels per step in x/y
    nb_pixels_x = np.int(xy_window[0] * (1 - xy_overlap[0]))
    nb_pixels_y = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Number of windows in x/y
    #nb_windows_x = np.int((span_x - xy_window[0]) / nb_pixels_x) + 1
    #nb_windows_y = np.int((span_y - xy_window[1]) / nb_pixels_y) + 1
    nb_windows_x = np.int(span_x/nb_pixels_x) - 1
    nb_windows_y = np.int(span_y/nb_pixels_y) - 1
    # Calculate x and y window positions
    window_list = []
    for y in range(nb_windows_y):
        for x in range(nb_windows_x):
            x1, y1 = x*nb_pixels_x+x_start_stop[0], y*nb_pixels_y+y_start_stop[0]
            x2, y2 = x1+xy_window[0], y1+xy_window[1]
            window_list.append(((x1, y1), (x2, y2)))
    if show_stats is True:
        print('shape:', image.shape)
        print('span (x, y):', span_x, span_y)
        print('pixels per step (x, y):', nb_pixels_x, nb_pixels_y)
        print('num windows (x, y):', nb_windows_x, nb_windows_y)
    # Return list of windows
    return window_list

def search_windows(img, windows, clf, scaler, spatial_params, hoc_params,
                    hog_params):
    """A function that takes an image,  a list of windows to be searched
    (output of slide_window()), a trained classifier, a scaler and feature
    parameters to return a list of windows where a car was predicted to exist
    """
    # create an empty list to receive positive detection windows
    on_windows = []
    # iterate over all windows in the list
    for window in windows:
        # extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1],
                                window[0][0]:window[1][0]], (64, 64))
        # extract features for that window
        features = extract_features_single(test_img, False, spatial_params,
                                             hoc_params, hog_params)
        # scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # predict using classifier
        prediction = clf.predict(test_features)
        # if positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # return windows for positive detections
    return on_windows


def find_cars(img, ystart, ystop, scale, svc, X_scaler,
                spatial_params, hoc_params, hog_params):
    """A single function to extract features using hog sub-sampling and
    make predictions.
    scale is used to reduce (by increasing from positive 1) the resolution of
    the image. For a scale of 1, the windows cover 64x64 pixels of original
    image. For a scale of 2, the windows cover 128x128 pixels of original image.
    """
    count=0
    hog_params = hog_params.copy()
    hog_conv = hog_params.pop('conv')
    hog_channels = hog_params.pop('channels')

    draw_img = np.copy(img)
    img = img.astype(np.float32)/255

    # Make a heatmap of zeros
    heatmap = np.zeros_like(draw_img[:,:,0])

    img_tosearch = img[ystart:ystop,:,:]

    if hog_conv is not None:
        ctrans_tosearch = convert_colour(img_tosearch, hog_conv)
    else:
        ctrans_tosearch = img_tosearch

    if scale != 1:
        imshape = img_tosearch.shape
        img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    # Compute individual channel HOG features for the entire image
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // hog_params['pix_per_cell']) - hog_params['cell_per_block'] + 1
    nyblocks = (ch1.shape[0] // hog_params['pix_per_cell']) - hog_params['cell_per_block'] + 1
    nfeat_per_block = hog_params['orientations']*hog_params['cell_per_block']**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // hog_params['pix_per_cell']) - hog_params['cell_per_block'] + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    hog1 = grad_hist(ch1, feature_vec=False, **hog_params)
    hog2 = grad_hist(ch2, feature_vec=False, **hog_params)
    hog3 = grad_hist(ch3, feature_vec=False, **hog_params)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            count +=1
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*hog_params['pix_per_cell']
            ytop = ypos*hog_params['pix_per_cell']

            # Extract the image patch for color features
            subimg = cv2.resize(img_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            spatial_features = bin_spatial(subimg, **spatial_params)
            hist_features = color_hist(subimg, **hoc_params)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw] +=1

    return draw_img, heatmap, count

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def convert_colour(image, conv):
    return cv2.cvtColor(image, eval('cv2.COLOR_'+conv))

def draw_boxes(image, bboxes, colour=(0, 0, 255), thick=6):
    """A function that takes an image, a list of bounding boxes, a colour tuple,
    and a line thickness, and draws boxes on the output.
    """
    # make a copy of the image
    draw_image = np.copy(image)
    # draw each bounding box on your image copy
    for bbox in bboxes:
        cv2.rectangle(draw_image, bbox[0], bbox[1], colour, thick)
    # return the image copy with boxes drawn
    return draw_image

def find_matches(img, template_list, method=cv2.TM_CCOEFF):
    """A function that takes an image and a list of templates, then searches the
    image and returns a list of bounding boxes for each matched template.
    """
    # empty list to take bbox coordinates
    bbox_list = []
    # iterate through template list
    for template in template_list:
        h = template.shape[0]
        w = template.shape[1]
        # search the image
        res = cv2.matchTemplate(img, template, method)
        # extract location of best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # determine bounding box corners for the match
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        bbox_list.append((top_left, bottom_right))
    # return the list of bounding boxes
    return bbox_list

def print_min_max(image):
    print('chan0 min max', image[..., 0].min(), image[..., 0].max())
    print('chan1 min max', image[..., 1].min(), image[..., 1].max())
    print('chan2 min max', image[..., 2].min(), image[..., 2].max())

def visualise(fig, rows, cols, imgs, titles):
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        plt.title(i+1)
        img_dims = len(img.shape)
        if img_dims < 3:
            plt.imshow(img, cmap='hot')
            plt.title(titles[i])
        else:
            plt.imshow(img)
            plt.title(titles[i])
