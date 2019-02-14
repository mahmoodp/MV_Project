# USAGE
# python scan.py --image images/page.jpg

# import the necessary packages
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import cv2
import imutils

'''
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())
'''
def order_points_old(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def find_parts(input_image, warped_image ):
	
    # find contours in the cropped image
	cnts = cv2.findContours(input_image.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	# sort the contours from left-to-right and initialize the bounding box
	# point colors
	(cnts, _) = contours.sort_contours(cnts)
	colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))
	# loop over the contours individually
	for (i, c) in enumerate(cnts):
		# if the contour is not sufficiently large, ignore it
		if cv2.contourArea(c) < 150:
			continue

		# compute the rotated bounding box of the contour, then
		# draw the contours
		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int")
		cv2.drawContours(warped_image, [box], -1, (0, 255, 0), 2)

		# show the original coordinates
		print("Object #{}:".format(i + 1))
		print(box)

		# order the points in the contour such that they appear
		# in top-left, top-right, bottom-right, and bottom-left
		# order, then draw the outline of the rotated bounding
		# box
		rect = order_points_old(box)

		# show the re-ordered coordinates
		print(rect.astype("int"))
		print("")

		# loop over the original points and draw them
		for ((x, y), color) in zip(rect, colors):
			cv2.circle(warped_image, (int(x), int(y)), 5, color, -1)

		# draw the object num at the top-left corner
		cv2.putText(image, "Object ",
		(int(rect[0][0] - 15), int(rect[0][1] - 15)),
		cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

		# show the image
		cv2.imshow("edge", input_image)
		cv2.imshow("Image", warped_image)
		cv2.waitKey(0)

       




# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
#image = cv2.imread(args["image"])
image = cv2.imread("images/export.png")
ratio = image.shape[0] / 800.0
orig = image.copy()
image = imutils.resize(image, height = 800)

'''
lower = np.array([10,100,20])  #-- Lower range --
upper = np.array([20,255,200])  #-- Upper range --
mask = cv2.inRange(image, lower, upper)
res = cv2.bitwise_and(image, image, mask = mask)
cv2.imshow('result', res)
'''



# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)


edged = cv2.Canny(gray, 120, 200)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)


# show the original image and the edge detected image
print("STEP 1: Edge Detection")
#cv2.imshow("Image", image)
#cv2.imshow("Edged", edged)
#cv2.imshow("Gray", gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#runfile('scan.py', args='--image images/receipt.jpg')

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
test= cnts
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break

# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
#cv2.imshow("Outline", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# apply the four point transform to obtain a top-down
# view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
edged_warped = cv2.Canny(warped, 75, 200)
edged_warped = cv2.dilate(edged_warped, None, iterations=1)
edged_warped = cv2.erode(edged_warped, None, iterations=1)
#local_thresh = threshold_local(warped, 19, offset = 10, method = "gaussian")
#warped = (warped > local_thresh).astype("uint8") * 255
resized_edged_warped = imutils.resize(edged_warped, height = 650)
#rotated = imutils.rotate(warped, angle=90)
# show the original and scanned images
print("STEP 3: Apply perspective transform")
#cv2.imshow("Original", imutils.resize(orig, height = 650))
#cv2.imshow("Scanned", resized_warped)
#cv2.imshow("Rotated", imutils.resize(rotated, width = 650))
#cv2.waitKey(0)


#Define the dimensions of index table
row_number = 4
column_number = 8
x_start = 130
y_start = 130
x_step = 130
y_step = 130
cell_gap = 10


'''
cropped = resized_warped[130:255, 130:260]
print(cropped)
croppedline = resized_warped[130:255, 260:270]
cropped1 = resized_warped[130:255, 270:400]
cropped2 = resized_warped[130:255, 410:540]
cropped3 = resized_warped[130:255, 550:680]
'''

#Loop over the captured figure to crop each cell and 
# check if it is emmpty or occupied
'''
for i in range(1, row_number+1):
	y_start = i * (y_step + cell_gap) - cell_gap
	for j in range (1, column_number+1):
		x_start = j* (x_step+ cell_gap) - cell_gap
		print(y_start,y_step,x_start,x_step)
		cropped = warped[y_start:y_start+y_step, x_start:x_start+x_step]
		cropped_edged = resized_edged_warped[y_start:y_start+y_step, x_start:x_start+x_step]
		#cv2.imshow("cropped:{},{}".format(i,j), cropped)
		find_part(cropped,cropped_edged)
		print(cropped.shape)
		cv2.waitKey(0)
	cv2.waitKey(0)	
'''

find_parts(warped,edged_warped)



