import numpy as np
import cv2
import sys


def combine_images(img_1, img_2, img_3, img_4):
    """
    concatenates four images to one image.

    :param img_1: first image
    :param img_2: second image
    :param img_3: third image
    :param img_4: fourth image
    :return: combined image
    """

    # first, stack images horizontally, two at a time
    first_stack = np.concatenate((img_1, img_2), axis=1)
    second_stack = np.concatenate((img_3, img_4), axis=1)

    # then, combine the two stacks vertically
    combined = np.concatenate((first_stack, second_stack), axis=0)

    return combined


def resize_img(img):
    """
    downsizes an image if it has a height larger than 600
    and/or a width larger than 480 while keeping aspect ratio.

    :param img: image to be resized
    :return: image with new or original size
    """

    height, width = img.shape[:2]

    if height > 960:
        new_height = 960
        ratio = float(new_height / height)
        height = new_height
        width = int(width * ratio)

    if width > 540:
        new_width = 540
        ratio = float(new_width / width)
        width = new_width
        height = int(height * ratio)

    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    return resized_img


def remove_noise(mask):
    """
    removes black points and white noise from mask.

    :param mask: mask of a frame
    :return: mask without noise
    """

    kernel = np.ones((2, 2), np.uint8)
    kernel2 = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel2, iterations=2)
    mask = cv2.dilate(mask, kernel2, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), cv2.BORDER_DEFAULT)

    return mask


def classify_objects(stats):
    """
    performs a simple classification of objects
    as car, person, or other  and counts occurrences
    based on the following assumptions:

    - an object with a larger width than height is a car
    - an object with a larger height than width is a person
    - an objects with an equal width and height is other

    :param stats: array with statistics output for each label
                  identified in connected components analysis
    :return: a dict with the counts of each object type
    """

    classifications = {"persons":0,
                       "cars": 0,
                       "others": 0}

    for i in range(1,len(stats)):
        height = stats[i, cv2.CC_STAT_HEIGHT]
        width = stats[i, cv2.CC_STAT_WIDTH]
        ratio = float(width / height)

        if ratio > 1:
            classifications["cars"] += 1

        elif ratio < 1:
            classifications["persons"] += 1

        else:
            classifications["others"] += 1

    return classifications


def print_stats(frame_count, object_count, classifications):
    """
    prints frame number and number of identified objects.

    :param frame_count:
    :param object_count:
    :param classifications:
    """

    frame_str = str(frame_count).zfill(4)
    output = "Frame {0}: {1} object(s)".format(frame_str, object_count)

    if object_count > 0:
        output += " ({0} person(s), {1} car(s) and {2} other(s)".format(classifications.get("persons"),
                                                                        classifications.get("cars"),
                                                                        classifications.get("others"))
    print(output)


def run_background_modelling(videofile):
    """
    runs background modelling functions on each frame of a
    specified video and displays the following in a single window:

    - original frame
    - estimated background frame
    - detected moving pixels before filtering
    - detected objects

    :param videofile: file with the video to be processed
    """

    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20)
    cap = cv2.VideoCapture(videofile)
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if frame is None:
            break

        frame_count += 1
        frame = resize_img(frame)

        # calculate foreground mask
        fg_mask = back_sub.apply(frame)

        # get background
        background = back_sub.getBackgroundImage()

        # remove noise
        fg_mask_without_noise = remove_noise(fg_mask)

        # connected component analysis
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask_without_noise)

        # extract objects and make background black
        frame_with_objects = frame.copy()
        frame_with_objects[labels==0] = 0

        # classify objects
        classifications = classify_objects(stats)

        # print stats
        print_stats(frame_count, num_labels-1, classifications)

        # display frames in one window
        combined = combine_images(frame, background, cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR), frame_with_objects)
        cv2.imshow("Background Modelling", combined)

        keyboard = cv2.waitKey(30)
        if keyboard == "q" or keyboard == 27:
            break


    cv2.destroyAllWindows()
    cap.release()


def get_area(w, h):
    """
    calculates area of a box.

    :param w: width of box
    :param h: height of box
    :return: area of box
    """

    return w * h


def find_closest(areas, boxes):
    """
    finds up to three boxes that are closest to the camera
    based on the assumption that the largest boxes are the closest.

    :param areas: list of areas of boxes
    :param boxes: list of boxes with x coordinate, y coordinate,
                  width, height values for detected full bodies
    :return: list with up to three closest/largest boxes
    """

    closest = []
    n = 0

    if len(boxes) >= 3:
        n = 3

    else:
        n = len(boxes)

    for i in range(0, n):
        area_index = areas.index(max(areas))
        closest.append(boxes[area_index])
        areas[area_index] = -1

    return closest


def find_center(x, y, w, h):
    """
    finds the center of a box.

    :param x: x coordinate of a box
    :param y: y coordinate of a box
    :param w: width of a box
    :param h: height of a box
    :return: center
    """

    x1 = int(w/2)
    y1 = int(h/2)

    cx = x + x1
    cy = y + y1

    return cx, cy


def find_euc_dist(cx1, cy1, cx2, cy2):
    """
    finds the euclidian distance between two centers of boxes.

    :param cx1: x coordinate of first center
    :param cy1: y coordinate of first center
    :param cx2: x coordinate of second center
    :param cy2: y coordinate of second center
    :return: euclidian distance
    """

    return np.sqrt(float((cx1 - cx2) ** 2) + float((cy1 - cy2) ** 2))


def run_pedestrian_detection(videofile):
    """
    runs pedestrian detection functions on a specified video
    using Haar Cascade full body detector and displays the
    following in a single window:

    - original frame
    - frame with overlapped detected bounding boxes
    - frame with detected and tracked (labelled) bounding boxes
    - frame with up to three detected objects closest to camera

    :param videofile:
    :return:
    """

    cap = cv2.VideoCapture(videofile)
    body_classifier = cv2.CascadeClassifier("haarcascade_fullbody.xml") # path to detector file

    prev_objects = []
    unique_count = 0

    while True:
        ret, frame = cap.read()

        if frame is None:
            break

        frame = resize_img(frame)
        frame_b = frame.copy()
        frame_c = frame.copy()
        frame_d = frame.copy()

        # detect full bodies
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = body_classifier.detectMultiScale(gray, 1.05, 2)
        areas = []
        frame_obj_ids = dict()

        # determine id and draw bounding boxes
        for (x, y, w, h) in boxes:
            cx, cy = find_center(x, y, w, h)
            id = ""
            min_dist = float('inf') # initialise as large value
            areas.append(get_area(w, h))  # needed to find closest to camera

            # no previous objects
            if len(prev_objects) == 0:
                unique_count += 1
                prev_objects.append(((cx, cy), unique_count))
                id = str(unique_count)

            else:
                found = False
                dist_thresh = 8.0

                for i in range(len(prev_objects)):
                    center, num = prev_objects[i]
                    prev_cx, prev_cy = center

                    # find euclidian distance
                    euc_dist = find_euc_dist(cx, cy, prev_cx, prev_cy)

                    # if object matches a previously detected object
                    if euc_dist < dist_thresh:
                        if euc_dist < min_dist:
                            min_dist = euc_dist
                            id = str(num)
                            found = True

                            # update previous objects with new centers
                            prev_objects[i] = ((cx, cy), num)

                # if object matches no previously detected object
                if not found:
                    unique_count += 1
                    prev_objects.append(((cx, cy), unique_count))
                    id = str(unique_count)


            # save object and id/label
            frame_obj_ids[(x, y, w, h)] = id

            # draw bounding boxes and add ids to tracked objects
            cv2.putText(frame_c, id, (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 102, 255), 2)
            cv2.rectangle(frame_c, (x, y), (x + w, y + h), (255, 102, 255), 1)
            cv2.rectangle(frame_b, (x, y), (x + w, y + h), (255, 102, 255), 1)


        # find three objects that are closest to camera
        closest = find_closest(areas, boxes)

        # draw boxes and add ids
        for (x, y, w, h) in closest:
            id = frame_obj_ids[(x,y,w,h)]
            cv2.putText(frame_d, id, (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 102, 255), 2)
            cv2.rectangle(frame_d, (x, y), (x + w, y + h), (255, 102, 255), 1)


        # display all four images in one window
        combined = combine_images(frame, frame_b, frame_c, frame_d)
        cv2.imshow("Pedestrian Detection and Tracking", combined)


        keyboard = cv2.waitKey(30)
        if keyboard == "q" or keyboard == 27:
            break

    cv2.destroyAllWindows()
    cap.release()


def main():
    args = sys.argv
    option = args[1]
    videofile = args[2]

    if option == "-b":
        run_background_modelling(videofile)

    elif option == "-d":
        run_pedestrian_detection(videofile)


if __name__ == '__main__':
    main()
