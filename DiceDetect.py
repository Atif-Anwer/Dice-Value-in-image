#------------------------------------------------------------------------
# __author__ = "Atif Anwer"
# __version__ = "1.0.0"
# __email__ = "contact@atifanwer.xyz"
#
# Command line arguments:
# -p : image path (optional)
# -t : threshold value (optional)
#
# **** Uses python 2.7 ***
#------------------------------------------------------------------------

import numpy as np
import argparse
import cv2
from os import listdir 
from PIL import Image as all_Images
import glob

#------------------------------------------
# Read all images in the folder
 # args: "path" = path to folder
#------------------------------------------
def read_images_in_folder(path):
	image_stack = []
	# build path string, sort by name
    	for img in sorted(glob.glob(path+'*.jpg')): 
		image_stack.append(cv2.imread(img))
    	return image_stack


#------------------------------------------
# Main Loop
 # args: "path" = path to folder (optional)
 #	 "threshold"  = threshold value for greyscale (optional)
#------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument( "-p", "--path", default = "./images/", help = "Path to the image")
ap.add_argument("-t", "--threshold", type = int, default = 100, help = "Enter threshold value")
args = vars(ap.parse_args())

image_stack = read_images_in_folder(args["path"])
print("\nNo of images in folder: {0}".format(len(image_stack)))

# image name index
q=0

# looping through images
for image in image_stack:
	# convert to grayscale
	grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# NOTE: kernel values iterations
	# kernel of 2x2 : maximum dots , mixes close together dice
	# kernel 3x3 : splits dice, lower no of dots
	# kernel 5x5 :  more dice, less dots
	kernel1 = np.ones((4,4), np.uint8)	# for dice
	kernel2 = np.ones((2,2), np.uint8)	# for dots
	# process the image for a mask
	grey = cv2.erode(grey, kernel1, iterations=1)
	grey = cv2.dilate(grey, kernel2, iterations=1)
	# grey = cv2.blur(grey, (3,3))

	
	#----------------------------------------------------------------------------
	# Threshold and crop  and draw image to segment the dice
	methods = [
		("THRESH_BINARY", cv2.THRESH_BINARY)
		#("THRESH_BINARY_INV", cv2.THRESH_BINARY_INV)
		]
	for (threshName, threshMethod) in methods:
		(T, threshImage) = cv2.threshold(grey, args["threshold"], 255, threshMethod)
		##Debug display
		# cv2.namedWindow(threshName,cv2.WINDOW_NORMAL)
		# cv2.resizeWindow(threshName, 600,600)
		# cv2.imshow(threshName, threshImage)
		# cv2.waitKey(0)


	# Crop target image using thresh as mask
	masked = cv2.bitwise_and(grey, grey, mask = threshImage)

	# find contours
	(_, contours, _) = cv2.findContours(masked.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	print ("\nNo. of dice detected ={}".format(len(contours)))

	# draw the contours on top of the original image and display
	dice = image.copy()
	cv2.drawContours(dice, contours, -1, (255, 0, 0), 2)
	# # Debug display
	# cv2.namedWindow("Contours",cv2.WINDOW_NORMAL)
	# cv2.resizeWindow("Contours", 600,600)
	# cv2.imshow("Contours", dice)
	# cv2.waitKey(0)
	# print ("Cropping individual dice for processing...")
	
	#----------------------------------------------------------------------------
	# Go over all contours found, remove everything else 
	# and find the dots in the current contour
	for index in contours:
		
		# create blank mask every loop
		negative_mask = np.ones(threshImage.shape[:2], dtype="uint8") * 255
		cv2.drawContours(negative_mask, [index], -1, 0, -1)
		inv = cv2.bitwise_not(negative_mask)
		cropped = cv2.bitwise_and(threshImage, threshImage, mask=inv )
		cropped= cv2.erode(cropped, kernel2, iterations=1)
		
		# # Debug display
		# cv2.namedWindow("dice",cv2.WINDOW_NORMAL)
		# cv2.resizeWindow("dice", 600,600)
		# cv2.imshow("dice", cropped)
		# cv2.waitKey(0)

		# Flood fill background
		im_floodfill = cropped.copy()
			
		h, w = cropped.shape[:2]
		mask = np.zeros((h+2, w+2), np.uint8)
		# Floodfill from point (0, 0)
		cv2.floodFill(im_floodfill, mask, (0,0), 255);
		im_floodfill_inv = cv2.bitwise_not(im_floodfill)
		# # Debug display
		# cv2.namedWindow("dice",cv2.WINDOW_NORMAL)
		# cv2.resizeWindow("dice", 600,600)
		# cv2.imshow("dice", im_floodfill_inv)
		# cv2.waitKey(0)

		#----------------------------------------------------------------------------
		# Finding contours in dice
		# cv2.CHAIN_APPROX_NONE returns more dots
		# cv2.CHAIN_APPROX_SIMPLE returns lesser dots
		(_, faceValue, _) = cv2.findContours(im_floodfill_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		dice_value = len(faceValue)

		for index2 in faceValue:
			# finding bounding rectangle for each contour
			(x2,y2),r2 = cv2.minEnclosingCircle(index2)	
			M = cv2.moments(index2)
			cX = int(M["m10"] / M["m00"]) 
			cY = int(M["m01"] / M["m00"]) 
			
			# Draw x on the detected dot
			cv2.putText(dice, "x", ((int(x2)-int(r2)/2),(int(y2)+int(r2)/2)), 
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
				
		# Finding the bounding circle and its center to place text
		(x1,y1),r1 = cv2.minEnclosingCircle(index)
		# Text position is offset by half radius to center the text on the dots
		cv2.putText(dice, str(dice_value), ((int(x1)-int(r1)/2),(int(y1)+int(r1)/2)), 
			cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 4)	
		cv2.namedWindow("Detected Face Value",cv2.WINDOW_NORMAL)
		cv2.resizeWindow("Detected Face Value", 600,600)
		cv2.imshow("Detected Face Value", dice)
	
	# Write the final image
	q+=1
	cv2.imwrite('ResultA0'+str(q)+'.jpeg', dice)

	cv2.waitKey(0)
cv2.destroyAllWindows()
#----------------------------------------------------------------------------	