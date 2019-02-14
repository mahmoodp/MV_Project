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
from barcode_scanner import scan_barcode

'''
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())
'''
def extract_index_table(image):

	# load the image and compute the ratio of the old height
	# to the new height, clone it, and resize it

	ratio = image.shape[0] / 650.0
	orig = image.copy()
	image = imutils.resize(image, height = 650)

	# convert the image to grayscale, blur it, and find edges
	# in the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	edged = cv2.Canny(gray, 120, 200)
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)


	'''
	# show the original image and the edge detected image
	print("STEP 1: Edge Detection")
	cv2.imshow("Image", image)
	cv2.imshow("Edged", edged)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''

	# find the contours in the edged image, keeping only the
	# largest ones, and initialize the screen contour


	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

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

	'''
	# show the contour (outline) of the piece of paper
	print("STEP 2: Find contours of paper")
	cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
	cv2.imshow("Outline", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''


	# apply the four point transform to obtain a top-down
	# view of the original image
	warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
	warped = imutils.resize(warped, height = 650)

	# convert the warped image to grayscale, then threshold it
	# to give it that 'black and white' paper effect
	#warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	#T = threshold_local(warped, 19, offset = 10, method = "gaussian")
	#warped = (warped > T).astype("uint8") * 255

	# show the original and scanned images
	#print("STEP 3: Apply perspective transform")
	#cv2.imshow("Original", imutils.resize(orig, height = 650))
	#cv2.imshow("Scanned", imutils.resize(warped, height = 650))
	#cv2.waitKey(0)
	return warped
	
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

def find_parts(image_input):
	
	#TARGET_PIXEL_AREA = 500000.0
	#ratio = image_input.shape[0] / 800.0
	#orig = image_input.copy()
	#image = imutils.resize(image_input, height = 800)
	gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0, 0)

	# perform edge detection, then perform a dilation + erosion to
	# close gaps in between object edges
	edged = cv2.Canny(gray, 120, 200)
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)



	# find contours in the edge map
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	print(len(cnts))

	# sort the contours from left-to-right and initialize the bounding box
	# point colors
	if len(cnts) < 4:
		print('the cell is occupied')
		return True
	else:
		print('the cell is occupied')
		return False

		'''
		(cnts, _) = contours.sort_contours(cnts)
		colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))

		# loop over the contours individually
		for (i, c) in enumerate(cnts):
			# if the contour is not sufficiently large, ignore it
			if cv2.contourArea(c) < 10:
				continue

			# compute the rotated bounding box of the contour, then
			# draw the contours
			box = cv2.minAreaRect(c)
			box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
			box = np.array(box, dtype="int")
			cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

			# show the original coordinates
			print("Object #{}:".format(i + 1))
			print(box)

			# order the points in the contour such that they appear
			# in top-left, top-right, bottom-right, and bottom-left
			# order, then draw the outline of the rotated bounding
			# box00
			rect = order_points_old(box)

			# show the re-ordered coordinates
			print(rect.astype("int"))
			print("")

			# loop over the original points and draw them
			for ((x, y), color) in zip(rect, colors):
				cv2.circle(image, (int(x), int(y)), 5, color, -1)

			# draw the object num at the top-left corner
			cv2.putText(image, "Object ",
				(int(rect[0][0] - 15), int(rect[0][1] - 15)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

			# show the image
			cv2.imshow("edge", edged)
			cv2.imshow("Image", image)
			cv2.waitKey(0)
		'''


# load the image 


#image = cv2.imread(args["image"])


image = cv2.imread("images/test.jpg")


# read the barcode
barcode_data = scan_barcode(image)

# extract the specs of index table
# the order of fileds in barcode data is as following:
# part number, row number, coumn number, x_start, y_start, x_step, y_step, cell_gap
part_number = barcode_data[0]
row_number = int(barcode_data[1])
column_number = int(barcode_data[2])
x_start = int(barcode_data[3])
y_start = int(barcode_data[4])
x_step = int(barcode_data[5])
y_step = int(barcode_data[6])
cell_gap = int(barcode_data[7])




cv2.imshow("InputImage", imutils.resize(image, height = 650))
cv2.waitKey(0)

warped = extract_index_table(image)

cv2.imshow("Scanned", warped)
cv2.waitKey(0)

'''
#Define the dimensions of index table
row_number = 4
column_number = 8
x_start = 130
y_start = 130
x_step = 130
y_step = 130
cell_gap = 10
'''

'''
cropped = resized_warped[130:255, 130:260]
print(cropped)
croppedline = resized_warped[130:255, 260:270]
cropped1 = resized_warped[130:255, 270:400]
cropped2 = resized_warped[130:255, 410:540]
cropped3 = resized_warped[130:255, 550:680]
'''

index_table_matrix = np.zeros((row_number,column_number))


#Loop over the index table to crop each cell and 
# check if it is emmpty or occupied


for i in range(1, row_number+1):
	y_start = i * (y_step + cell_gap) - cell_gap
	for j in range (1, column_number+1):
		x_start = j* (x_step+ cell_gap) - cell_gap
		#print(y_start,y_step,x_start,x_step)
		cropped = warped[y_start:y_start+y_step, x_start:x_start+x_step]
		edged = cv2.Canny(cropped, 120, 200)
		edged = cv2.dilate(edged, None, iterations=1)
		edged = cv2.erode(edged, None, iterations=1)
		#get_dominant_color(cropped)
		cv2.imshow("cropped:{},{}".format(i,j), cropped)
		cv2.waitKey(0)
		cv2.imshow("edge_cropped:{},{}".format(i,j), edged)
		#
		cell_check = find_parts(cropped)
		if cell_check:
			index_table_matrix[i-1,j-1]=1
		cv2.waitKey(0)
	cv2.waitKey(0)	


print(index_table_matrix)

#find_parts(warped)



