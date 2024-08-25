from flask import Flask,render_template,request
import numpy as np
import cv2
import numpy as np
import uuid
from deepface import DeepFace

app = Flask(__name__)


@app.route('/',methods=['GET'])
def home():
	return render_template('home.html')



@app.route('/predict',methods=['GET'])
def hello_world():
    return render_template('predict.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
	if request.method=='POST':
		
		imagefile =request.files['imagefile']
		
		image_path="./static/data/"+imagefile.filename
		unique_filename = str(uuid.uuid4())
		file_extension = image_path.split('.')[-1]
		new_filename = f'{unique_filename}.{file_extension}'
		new_path = "./static/data/" + new_filename
		imagefile.save(new_path)
		faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
		count=0
		sad=0
		
		happy=0
		cap= cv2.VideoCapture(new_path)
		height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
		width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
		while True:
			ret,frame=cap.read()  #read one image from video
			
			result = DeepFace.analyze(frame,actions=['emotion'],enforce_detection=False)
			count=count+1
			if ((result[0]['dominant_emotion']=='neutral')or(result[0]['dominant_emotion']=='angry')or
				(result[0]['dominant_emotion']=='fear')or (result[0]['dominant_emotion']=='sad')):
				sad=sad+1
				
			else:#if (result[0]['dominant_emotion']==('happy' or 'surprise' or 'disgust')):
				happy=happy+1
			if (count==100):
				break
		if sad>happy:
			result="depressed"
			print("depressed",sad,'%')
		else:
			result="not depressed"
			print("not depressed")
    
		
		
		return render_template('predict.html',predition=result,path=new_path)
	return render_template('predict.html')

    
if __name__ == '__main__':
 	app.run(debug=True)