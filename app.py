from flask import Flask,render_template,request
import numpy as np
import pickle
app=Flask(__name__)
model=pickle.load(open('Hotel_Reservation.pkl','rb'))
@app.route('/')
def home():
  return render_template('index.html')
@app.route('/Predict',methods=['POST'])
def Predict():
  temp_features=[float(x) for x in request.form.values()]
  final_features=[np.array(temp_features)]
  print(final_features)
  prediction=model.predict(final_features)
  return render_template('output.html',prediction_number=prediction)
if __name__==:"__main__":
    app.run(debug=True)
  
