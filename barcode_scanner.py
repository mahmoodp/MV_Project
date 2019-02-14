# USAGE
# The function is used to read data of barcode mounted on index table
# The fields in barcode are seprated by "," and the order is as following:
# part number, row number, coumn number, x_start, y_start, x_step, y_step, cell_gap


# import the necessary packages
from pyzbar import pyzbar



def scan_barcode(input_image):

	# load the input image
	#image = cv2.imread(input_image)

	# find the barcodes in the image and decode each of the barcodes
	barcode = pyzbar.decode(input_image)

	#print(barcode)

	barcodeData = barcode[0].data.decode("utf-8")

	# split the barcode data and return Data
	Data = barcodeData.split(",")
	return Data
	
