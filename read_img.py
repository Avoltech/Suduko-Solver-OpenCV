import cv2
import numpy as np
from tensorflow.keras.models import load_model
import numpy as np


#Define empty suduko grid
suduko_grid = np.zeros((9,9), np.uint8)
filled_grid = np.zeros((9, 9), np.uint8)
solutions = []
image_resized = None
block_width = None
block_height = None
#Suduko backtracking algorithm
#Possible function check if a number can be put in a specific location
#such that that same number is not repeated in the row,column or the mini 3*3 block

def possible(x, y, num):
	global suduko_grid
	x_box = (x // 3) * 3
	y_box = (y // 3) * 3
	return (num not in suduko_grid[x] and num not in suduko_grid[:, y] and num not in suduko_grid[x_box:x_box+3, y_box:y_box+3])

#Solve function that is called recursively to find the possible solutions

def solve():
	global suduko_grid
	global filled_grid
	global solutions
	global image_resized
	global block_width
	global block_height
	for row in range(9):
		for col in range(9):
			if suduko_grid[row][col] == 0:
				for num in range(1, 10):
					if possible(row, col, num):
						suduko_grid[row, col] = num
						filled_grid[row, col] = num
						solve()
						suduko_grid[row, col] = 0
						filled_grid[row, col] = 0
				return

	font = cv2.FONT_HERSHEY_SIMPLEX

	#Plotting the suduko solution onto the image
	solution = image_resized.copy()
	for row in range(9):
		for col in range(9):
			if filled_grid[row, col] != 0:
				fontscale = 1
				thickness = 2

				textSize = cv2.getTextSize(str(filled_grid[row, col]), font, fontscale, thickness)[0]
				textX = (col*block_width + textSize[0]/2) 
				textY = (row*block_height + textSize[1]*1.5) 
				cv2.putText(solution, str(filled_grid[row, col]), (int(textX), int(textY)), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (0, 255, 0), thickness, cv2.LINE_AA)
	solutions.append(solution)


def solve_this_image(image=None):
	#Load the suduko puzzle image
	global image_resized
	global block_height
	global block_width
	global solutions
	global suduko_grid
	suduko_grid = np.zeros((9,9), np.uint8)
	filled_grid = np.zeros((9, 9), np.uint8)
	solutions = []
	image_resized = None
	block_width = None
	block_height = None
	
	image_resized = cv2.resize(image, (360, 360))

	#Detect the main 9x9 suduko box (biggest contour)
	#This is not necessary, but if user uses a image that has extra border then that image can be handled properly
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	ret, thresh1 = cv2.threshold(gray, 180, 255,cv2.THRESH_BINARY_INV) 
	dilation = cv2.dilate(thresh1, (3,3), iterations = 1) 
	contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	max_contour = max(contours, key=cv2.contourArea)
	x,y,w,h = cv2.boundingRect(max_contour)

	#Detecting the grid lines
	#We mask the grid lines from the image so that the numbers can be easily detected
	image = image[y:y+h, x:x+w]
	image_resized = cv2.resize(image, (360, 360))
	gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
	ret, thresh1 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV) 
	canny = cv2.Canny(cv2.bitwise_not(gray), 50, 150)
	lines = cv2.HoughLines(canny, 1, np.pi/180, 250)

	for line in lines:
		for rho, theta in line:
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))
			thresh1 = cv2.line(thresh1, (x1, y1), (x2, y2), 0, 3)

	contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

	#Load the ML trained model
	grid_coords = []
	model = load_model('pretrained model/CNN_model_50_epochs.h5')

	#Get the location of the digits in the puzzle and
	#Use the model to predict what digit it is
	for cnt in contours:
		x, y, w, h = cv2.boundingRect(cnt)
		if( h < 14 or w < 5):
			continue
		cropped = thresh1[y:y+h, x:x+w]
		cropped = cv2.bitwise_not(cropped)
		cropped_resized = cv2.resize(cropped, (30, 30))
		cropped_resized = cropped_resized.reshape(30, 30, 1)
		cropped_resized = np.expand_dims(cropped_resized, axis=0)

		cropped_resized = cropped_resized.astype('float32')  / 255.0
		pred = model.predict_classes(cropped_resized) + 1
		grid_coords.append([x, y, pred])

	#Displaying the puzzle image and the detected numbers
	# cv2.imshow("puzzle image", image_resized)
	# cv2.imshow("digits only", dilation)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()


	img_w, img_h,_ = image_resized.shape
	block_width = img_w / 9
	block_height = img_h / 9



	#Fill the grid with the detected numbers
	for i in grid_coords:
		x_pos = int(i[1] // block_width)
		y_pos = int(i[0] // block_height)	
		suduko_grid[x_pos, y_pos] = i[2]


	solve()

	return solutions
	