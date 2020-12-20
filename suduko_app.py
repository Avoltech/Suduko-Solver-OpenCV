import streamlit as st
import numpy as np
from PIL import Image
from read_img import solve_this_image
import random



input_section_title = """
	<div>
		<h2 style="
				text-align: center; 
				font-family: 'Helvetica Neue Bold', sans-serif;
				font-size:30px;
				color:#6f549a;
				">Input</h2>
	</div>
	"""

output_section_title = """
	<div>
		<h2 style="
				text-align: center; 
				font-family: 'Helvetica Neue Bold', sans-serif;
				font-size:30px;
				color:#6f549a;
				">Output</h2>
	</div>
	"""


def user_image_method():
	solve_button_click = False
	solution_idx = 1

	st.warning("While uploading the image make sure that only the puzzle is visible. Check out the images from the database to view sample.")
	upload_image = st.file_uploader("Upload Image Here", type=['jpg', 'png', 'jpeg'])


	col1, mid, col2 = st.beta_columns([3, 1, 3])
	with col1:
		st.markdown(input_section_title, unsafe_allow_html=True)

		if upload_image is not None:
			input_image = Image.open(upload_image).convert('RGB')
			st.image(input_image, width=320)
			solve_button_click = st.button('Solve')

	with col2:
		st.markdown(output_section_title, unsafe_allow_html=True)
		if solve_button_click:
			solutions = []
			solutions = solve_this_image(image=np.array(input_image))
			if len(solutions)>0:
				for s in solutions:
					output_solution = Image.fromarray(s).convert('RGB').resize((320, 320))
					st.image(output_solution)

			st.write("There exists ",len(solutions)," solutions for the given puzzle")


def database_image_method():
	solve_button_click_2 = False
	solution_idx = 1
	img_idx = None

	img_idx =st.selectbox("Select Image",[*range(1, 6)])

	col1, mid, col2 = st.beta_columns([3, 1, 3])

	with col1:
		st.markdown(input_section_title, unsafe_allow_html=True)

		input_image = Image.open("suduko sample puzzles/"+str(img_idx)+".png").convert('RGB')
		st.image(input_image, width=320)
		solve_button_click_2 = st.button("Solve")
	

	with col2:
		st.markdown(output_section_title, unsafe_allow_html=True)
		if solve_button_click_2:
			solutions = []
			solutions = solve_this_image(image=np.array(input_image))
			if len(solutions)>0:
				for s in solutions:
					output_solution = Image.fromarray(s).convert('RGB').resize((320, 320))
					st.image(output_solution)

			st.write("There exists ",len(solutions)," solutions for the given puzzle")


title = """
	<div>
		<h2 style="
				text-align: center; 
				font-family: 'Helvetica Neue Bold', sans-serif;
				font-size:50px;
				color:#fc4d4d;
				">Suduko Solver</h2>
	</div>
	"""
st.markdown(title, unsafe_allow_html=True)

choices = ["I will upload a image",
			"Select image from small database(5 imgs)"
]
USER_CHOICE = st.selectbox("Select Method",choices)
if USER_CHOICE == choices[0]:
	user_image_method()
elif USER_CHOICE == choices[1]:
	database_image_method()



