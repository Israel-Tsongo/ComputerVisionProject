import os
from flask import Flask,render_template
from flask import Flask,request,render_template,Response,url_for
from werkzeug.utils import secure_filename
import model



UPLOAD_FOLDER = '/home/israel/FlaskFinalProjectComputerVision/images/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
path_image=[]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/",methods=["POST"])
def home():
    if  request.method== "POST":
        uploaded_file = request.files["image"]   

        if uploaded_file.filename !="":

            filename = secure_filename(uploaded_file.filename)
            path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(path)
            path_image.append(path)
            todisp=model.display(path) 
           
            return render_template("index.html",result=todisp)   
        else:
            return "<h2>Vous n'aviez pas valide le fichier uploaded </h2>"
    
    return  render_template("index.html")

@app.route("/classifer")
def classifierImg():
        class_pred=model.predict(path_image[-1])

        return render_template("index.html",pred=class_pred,result=model.display(path_image[-1]))
     


if __name__=="__main__":

     app.run(debug=True)