import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import os
from os.path import isfile, join
import shutil
import imghdr

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

#"F:\phone\\whatsapp\\WhatsApp Images"

def classify(self,mypath):
	
	# Load the model
	model = tensorflow.keras.models.load_model('keras_model.h5')
	data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

	counter = 0

	document_path = mypath + "\\documents"
	generic_path = mypath + "\\generic"
	non_generic_path = mypath + "\\non-generic"

	if not os.path.exists(generic_path):
	    os.makedirs(generic_path)

	if not os.path.exists(non_generic_path):
	    os.makedirs(non_generic_path)

	if not os.path.exists(document_path):
	    os.makedirs(document_path)


	files = [f for f in os.listdir(mypath) if (isfile(join(mypath, f)) and (imghdr.what(join(mypath, f))=='jpg' or imghdr.what(join(mypath, f))=='jpeg'))]
	size = (224, 224)
	
	num_of_img = len(files)
	modifier = 100//num_of_img
	self.completed = 0

	for f in files:
		try:
			img = Image.open(f)
			# images.append(img)
			path = f

			#resize the image to a 224x224 with the same strategy as in TM2:
			#resizing the image to be at least 224x224 and then cropping from the center		
			image = ImageOps.fit(img, size, Image.ANTIALIAS)

			#turn the image into a numpy array
			image_array = np.asarray(image)

			# Normalize the image
			normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

			# Load the image into the array
			data[0] = normalized_image_array

			# run the inference
			prediction = model.predict(data).tolist()
			doc = prediction[0][0]
			non_gen = prediction[0][2]
			gen = prediction[0][1]

			if doc>0.80:
				shutil.move(join(mypath,path),join(document_path,path))
			
			elif non_gen>0.80:
				shutil.move(join(mypath,path),join(non_generic_path,path))

			elif gen>0.80:
				shutil.move(join(mypath,path),join(generic_path,path))
			
			self.ui.textEdit.append("processed "+ path)
			counter+=1
			image.close()
			self.completed += modifier
			self.ui.progressBar.setValue(self.completed)


		except IOError:
			pass

	self.ui.textEdit.append("Total Images Processed: " + str(counter))