
import json
from flask import send_from_directory,json,flash,Flask,request,redirect, url_for,render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

import cv2
import numpy as np
from pytesseract import image_to_string
import pytesseract
import os

from PIL import Image
import fitz
import csv 

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'


app = Flask(__name__)
app.secret_key = 'super secret key'
CORS(app,supports_credentials=True,resources={"*": {"origins": "*"}})

@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response

UPLOAD_FOLDER = '/app/annotationsFiles'
ALLOWED_EXTENSIONS = {'json','txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

per = 25
PARENTFOLDER = "./Images/"
orb = cv2.ORB_create(1000)
orb2 = cv2.ORB_create(nfeatures=1000)

def findDes(images):
    desList = []
    # print(len(images))
    for idx,img in enumerate(images):
        try:
            h,w,c =img.shape
            kp,des = orb2.detectAndCompute(img,None)
            desList.append({"kp":kp,"des":des,"img":img,"h":h,"w":w,"idx":idx})
        except Exception as e:
            pass
    return desList

def findID(img,desList,thresh=100):
    kp2,des2 = orb2.detectAndCompute(img,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matchList = []
    matchList2 = []
    finalValue = -1
    good = []
    try:
        for idx,des in enumerate(desList):
            matches = bf.knnMatch(des2,des['des'],k=2)
            matches2 = bf.match(des2,des['des'])
            good = []
            for m,n in matches:
                if m.distance <0.75 * n.distance:
                    good.append([m])
            goods = matches2[:int(len(matches2)*(10))]
            matchList.append(len(good))
            matchList2.append(
                {
                    "len":len(good),
                    "idx":idx,
                    "des2":des2,
                    "des":des["des"],
                    "kp":des["kp"],
                    "kp2":kp2,
                    "good":good,
                    "good2":goods,
                    "img":img,
                    "w":des["w"],
                    "h":des["h"],
                }
            )
    except:
        pass
    if len(matchList)!=0:
        items = []
        prevVal = -1
        for idx,item in enumerate(matchList):
            if item>thresh:
                if prevVal < item:
                    finalValue = matchList2[idx]
                    prevVal =item
                # items.append(item)
        # try:
        #     finalValue = matchList2[items.index(max(items))]
        # except Exception as e:
        #     finalValue = {"err":"No Match Found Try Decreasing the Similarity Count."}
    return finalValue

def findMatchAndExtract(img,desList,classNames,path,keys,content,thresh=100):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("./Images/testingImages/"+"macthing{i}.png".format(i=path),img)
    id = findID(img,desList,thresh)
    # if "err" in id:
    #     return id
    toPutInList = {}
    if id != -1:
        roi = json.load(open(os.path.join(app.config['UPLOAD_FOLDER'],classNames[id['idx']],"result.json")))
        found = desList[id['idx']]
        # kp1 = id['kp']
        # des1 = id['des']
        # imgIdx = found['idx']
        imgQ = found['img']
        # good = id['good'] 
        # good2 = id['good2'] 
        # kp2 = id['kp2']
        # w = id["w"]
        # h = id["h"]
        try:
            h,w,c =imgQ.shape
        except ValueError as e:
             h,w =imgQ.shape
        kp1,des1 = orb.detectAndCompute(imgQ,None)
        kp2,des2=orb.detectAndCompute(img,None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des2,des1)
        good = matches[:int(len(matches)*(10))]
        # imgMatch = cv2.drawMatches(img,kp2,imgQ,kp1,good,None,flags=2)
        # cv2.imwrite("./Images/testingImages/"+"macthing{i}.png".format(i=path),imgMatch) #save image to see matches
        srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        M,_ = cv2.findHomography(srcPoints,dstPoints,cv2.RANSAC,5.0)
        imgScan = cv2.warpPerspective(img,M,(w,h))
        imgShow = imgScan.copy()
        imgMask = np.zeros_like(imgShow)
        for x,r in enumerate(roi):
            cv2.rectangle(imgMask,(int(r[0][0]),int(r[0][1])),(int(r[1][0]),int(r[1][1])),(0,0,0),cv2.FILLED)
            imgShow = cv2.addWeighted(imgShow,0.99,imgMask,0.3,0)
            # cv2.imwrite(r[3]+".png",imgShow)
            imgCrop = imgShow[int(r[0][1]):int(r[1][1]),int(r[0][0]):int(r[1][0])]
            if r[2] == 'text':
                toPutInList[r[3].lower().replace("value","")] = image_to_string(imgCrop).split('\u000c')[0].replace("\n"," ")
                toPutInList["Beg Doc"] = path
                toPutInList["End Doc"] = ""
                toPutInList["Matched With"] = classNames[id['idx']]
            else:
                keys.append(r[3].lower())
                # TESSERACT

        content.append(toPutInList)
        imgShow = cv2.resize(imgScan,(w//3,h//3))
        # cv2.imwrite("./Images/testingImages/"+path+"_fianl.png",imgShow)
    else:
        toPutInList["Beg Doc"] = path
        toPutInList["End Doc"] = ""
        # toPutInList["Extracted"] = "None"
        content.append(toPutInList)
    
    return {"keys":keys,"content":content}

import time
@app.route("/parseV2/<foldername>",methods=['POST','GET'])
def parseV2(foldername):
    if request.method=='GET':
        start = time.time()
        classNames = []
        images = []
        toProcess = foldername
        pageNumber = 0
        content = []
        thresh = 100
        if "thresh" in request.args:
            thresh = int(request.args['thresh'])
        for folder in os.listdir(os.path.join("/app/Images")):
            if ".json" not in folder:
                if os.path.exists(os.path.join("/app/Images",folder,"refrence")):
                    for file in os.listdir(os.path.join("/app/Images",folder,"refrence")):
                        if file == 'ref.png':
                            imgQ = cv2.imread(os.path.join("/app/Images",folder,"refrence","ref.png"))
                            images.append(imgQ)
                            classNames.append(folder)
                            
        print('Total Classes Detected: ',len(classNames))
        desList = findDes(images)
        files = os.listdir(os.path.join("/app/Images",toProcess))
        keys = []
        for id,path in enumerate(files):
            print((id*100/len(files)))
            if str(path).lower().endswith(".pdf"):
                with fitz.open(os.path.join(PARENTFOLDER,toProcess,path)) as docs:
                    for pages in docs:
                        if str(pages.number) == str(pageNumber):
                            page = pages
                            zoom_x = 2.0
                            zoom_y = 2.0
                            trans = fitz.Matrix(zoom_x, zoom_y).prerotate(0)
                            pix = page.get_pixmap(matrix=trans, alpha=True)
                            pdfImg = Image.frombytes('RGBA',(pix.width,pix.height),pix.samples)
                            # img = cv2.imread(pdfImg)
                            fill_color = (255,255,255)
                            background = Image.new(pdfImg.mode[:-1], pdfImg.size, fill_color)
                            background.paste(pdfImg, pdfImg.split()[-1]) # omit transparency
                            img = background
                            extracted = findMatchAndExtract(img,desList,classNames,path,keys,content,thresh)
                            if "err" in  extracted:
                                print(extracted['err'])
                                # return extracted['err']
                            else:
                                keys = extracted['keys']
                                content = extracted['content']
            elif str(path).lower().endswith(".tiff") or str(path).lower().endswith(".tif"):
                img = cv2.imread(os.path.join("/app/Images",toProcess,path))
                # img.seek(int(pageNumber))
                extracted = findMatchAndExtract(img,desList,classNames,path,keys,content,thresh)
                if "err" in  extracted:
                    print(extracted['err'])
                    # return extracted['err']
                else:
                    keys = extracted['keys']
                    content = extracted['content']
            elif str(path).lower().endswith(".png") or str(path).lower().endswith(".jpeg") or str(path).lower().endswith(".jpg"):
                img = cv2.imread(os.path.join("/app/Images",toProcess,path))
                extracted = findMatchAndExtract(img,desList,classNames,path,keys,content,thresh)
                if "err" in  extracted:
                    print(extracted['err'])
                    # return extracted['err']
                else:
                    keys = extracted['keys']
                    content = extracted['content']
        if len(keys) == 0 or len(content) == 0:
            return "No Match Found Try Decreasing the Similarity Count."
        os.makedirs(os.path.join("./annotationsFiles/{foldername}".format(foldername=foldername)), exist_ok=True)
        with open("./annotationsFiles/{foldername}/{foldername}.csv".format(foldername=foldername), 'w') as csvfile: 
            key= ["Beg Doc","End Doc","Matched With"]
            # key= ["Beg Doc","End Doc"]
            keys = set(keys)
            for k in keys:
                key.append(k)
            csvwriter = csv.DictWriter(csvfile,fieldnames=key)

            csvwriter.writeheader()
            for item in content:
                    csvwriter.writerow(item)
        end = time.time()
        print("timeTaken: ",end - start)
        return send_from_directory(app.config["UPLOAD_FOLDER"], "{foldername}/{foldername}.csv".format(foldername=foldername))

@app.route('/parse/<foldername>/',methods = ['POST', 'GET'])
def parse(foldername):
    if request.method == "GET":
        # print(foldername)
        roi = json.load(open(os.path.join(app.config['UPLOAD_FOLDER'],foldername,"result.json")))
        # roi = [
        #     [( 1130, 197 ), ( 1327, 249 ) ,"key" ,"Date"],
        #     [( 1450, 197 ), ( 1609, 249 ) ,"text" ,"Date"],
        #     [( 1142, 156 ), ( 1324, 189 ) ,"key" ,"Invoice"],
        #     [( 1435, 152 ), ( 1613, 196 ) ,"text" ,"Invoice"],
        #     [( 677, 431 ), ( 825, 475 ) ,"key" ,"ShipTo"],
        #     [( 636, 476 ), ( 1086, 613 ) ,"text" ,"ShipTo"],
        #     [( 137, 435 ), ( 263, 475 ) ,"key" ,"BillTo"],
        #     [( 133, 483 ), ( 452, 598 ) ,"text" ,"BillTo"],
        #     [( 1238, 1180 ), ( 1379, 1228 ) ,"key" ,"SubTotal"],
        #     [( 1439, 1183 ) ,( 1624, 1231 ) ,"text" ,"SubTotal"],
        #     [( 1223, 1678 ), ( 1431, 1722 ) ,"key" ,"Total"],
        #     [( 1443, 1682 ), ( 1614, 1726 ), "text" ,"Total"],
        # ]
        li = os.listdir(PARENTFOLDER)
        toProcess = foldername
        pageNumber = 0
        content = []
        for idx,item in enumerate(li):
            if item == toProcess:
                imgQ = cv2.imread(os.path.join(PARENTFOLDER,toProcess,"refrence","ref.png"))
                h,w,c =imgQ.shape
                kp1,des1 = orb.detectAndCompute(imgQ,None)
                files = os.listdir(os.path.join(PARENTFOLDER,toProcess))
                for id,path in enumerate(files):
                    print((id*100/len(files)))
                    if str(path).lower().endswith(".pdf"):
                        with fitz.open(os.path.join(PARENTFOLDER,toProcess,path)) as docs:
                            for pages in docs:
                                if str(pages.number) == str(pageNumber):
                                    page = pages
                                    zoom_x = 2.0
                                    zoom_y = 2.0
                                    trans = fitz.Matrix(zoom_x, zoom_y).prerotate(0)
                                    pix = page.get_pixmap(matrix=trans, alpha=True)
                                    pdfImg = Image.frombytes('RGBA',(pix.width,pix.height),pix.samples)
                                    # img = cv2.imread(pdfImg)
                                    fill_color = (255,255,255)
                                    background = Image.new(pdfImg.mode[:-1], pdfImg.size, fill_color)
                                    background.paste(pdfImg, pdfImg.split()[-1]) # omit transparency
                                    img = background
                                    # img.save("./Images/testingImages/"+"testin2{i}.png".format(i=path))
                                    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
                                    kp2,des2=orb.detectAndCompute(img,None)
                                    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
                                    matches = bf.match(des2,des1)
                                    # good = matches[:int(len(matches)*(per/100))]
                                    good = matches[:int(len(matches)*(10))]
                                    imgMatch = cv2.drawMatches(img,kp2,imgQ,kp1,good,None,flags=2)
                                    # cv2.imwrite("./Images/testingImages/"+"macthing{i}.png".format(i=path),imgMatch) #save image to see matches
                                    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                                    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)

                                    M,_ = cv2.findHomography(srcPoints,dstPoints,cv2.RANSAC,5.0)
                                    imgScan = cv2.warpPerspective(img,M,(w,h))
                                    imgShow = imgScan.copy()
                                    imgMask = np.zeros_like(imgShow)
                                    toPutInList = {}
                                    keys = ["Beg Doc","End Doc"]
                                    for x,r in enumerate(roi):
                                        cv2.rectangle(imgMask,(int(r[0][0]),int(r[0][1])),(int(r[1][0]),int(r[1][1])),(255,0,0),cv2.FILLED)
                                        imgShow = cv2.addWeighted(imgShow,0.99,imgMask,0.3,0)
                                        imgCrop = imgScan[int(r[0][1]):int(r[1][1]),int(r[0][0]):int(r[1][0])]
                                        cv2.imwrite(r[3]+".png",imgCrop)
                                        if r[2] == 'text':
                                            toPutInList[r[3].replace("Value","")] = image_to_string(imgCrop).split('\u000c')[0].replace("\n"," ")
                                            toPutInList["Beg Doc"] = path
                                            toPutInList["End Doc"] = ""
                                        else:
                                            keys.append(r[3])
                                            # TESSERACT

                                    content.append(toPutInList)
                                    imgShow = cv2.resize(imgScan,(w//3,h//3))
                                    cv2.imwrite("./Images/testingImages/"+path+"_fianl.png",imgShow)
        os.makedirs(os.path.join("./annotationsFiles/{foldername}".format(foldername=foldername)), exist_ok=True)
        with open("./annotationsFiles/{foldername}/{foldername}.csv".format(foldername=foldername), 'w') as csvfile: 
            csvwriter = csv.DictWriter(csvfile,fieldnames=keys) 
            csvwriter.writeheader()
            for item in content:
                try:
                    print(keys,item)
                    csvwriter.writerow(item)
                except Exception as e:
                    print("##################")
                    raise ValueError(e)
        return send_from_directory(app.config["UPLOAD_FOLDER"], "{foldername}/{foldername}.csv".format(foldername=foldername))
        # return Response(
        #     "Done",
        #     status=200,
        # )


@app.route("/createFolder",methods = ['POST','GET'])
def createFolder():
    if request.method == "POST":
        folderName = request.form.get('folderName')
        os.makedirs(os.path.join("./Images",folderName), exist_ok=True)
    return redirect(url_for("imageList"))
    

@app.route('/',methods = ['GET'])
def index():
    return redirect(url_for("imageList")) 

@app.route('/imageList',methods = ['POST', 'GET'])
def imageList():
    if request.method == "GET":
        args = request.args
        os.makedirs(os.path.join("./Images"), exist_ok=True) 
        list = os.listdir(os.path.join("./Images"))
        if "path" in args:
            list = os.listdir(os.path.join("./Images",args['path']))
        return render_template("imageList.html",list=list,len = len(list))

@app.route("/upload/<foldername>/images",methods=['POST','GET'])
def uploadImages(foldername):
    if request.method == 'POST':
        os.makedirs(os.path.join("./Images",foldername), exist_ok=True)
        for f in request.files.getlist('photos'):
            f.save(os.path.join("./Images",foldername,f.filename))
    return redirect("/imageList?path={foldername}".format(foldername=foldername))

@app.route("/upload/<foldername>/ref",methods=['POST','GET'])
def uploadRef(foldername):
    if request.method == 'POST':
        print(request.url)
        os.makedirs(os.path.join("./Images",foldername,"refrence"), exist_ok=True)
        for f in request.files.getlist('photos'):
            print(f)
            f.save(os.path.join("./Images",foldername,"refrence","ref.png"))
    return redirect("/imageList?path={foldername}".format(foldername=foldername))

@app.route('/upload/annotations',methods = ['POST', 'GET'])
def home():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            imgId = int(request.form.get('imageId')) 
            folderName = request.form.get('folderName')
            os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'],folderName), exist_ok=True)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],folderName, "providedAnnot.json"))
            with open(os.path.join(app.config['UPLOAD_FOLDER'],folderName, "providedAnnot.json")) as annot:
                annot = json.load(annot)
                newData = []
                for annotation in annot["annotations"]:
                    if annotation['image_id'] == imgId:
                        x = annotation['bbox'][0]
                        y = annotation['bbox'][1]
                        x1 = int(annotation['bbox'][2])+int(annotation['bbox'][0])
                        y1 = int(annotation['bbox'][3])+int(annotation['bbox'][1])
                        category = [cat for cat in annot["categories"] if cat['id'] == annotation['category_id']][0]
                        if "Value" in category['name']:
                            newData.append([(x, y),(x1, y1),"text",category['name']])
                        else:
                            newData.append([(x, y),(x1, y1),"key",category['name']])

            # return redirect(url_for('download_file', name=filename))
            json_object = json.dumps(newData, indent = 4)
            # Writing to sample.json
            with open(os.path.join(app.config['UPLOAD_FOLDER'],folderName, "result.json"), "w") as outfile:
                outfile.write(json_object)
            return redirect(url_for('imageList'))
            
    return '''
    <!doctype html>
    <title>Upload Annotation File</title>
    <body>
        <h1>Upload Annotation file</h1>
        <form method=post enctype=multipart/form-data>
            <input type=file name="file">
            <input type=text name="imageId" placeholder="image_id ">
            <input type=text name="folderName" placeholder="Title">
            <input type=submit value=Upload>
        </form>
    </body>
    '''
@app.route('/upload/annotationsV2',methods = ['POST', 'GET'])
def homev2():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            imgId = int(request.form.get('imageId')) 
            folderName = request.form.get('folderName')
            os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'],folderName), exist_ok=True)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],folderName, "providedAnnot.json"))
            with open(os.path.join(app.config['UPLOAD_FOLDER'],folderName, "providedAnnot.json")) as annot:
                annot = json.load(annot)
                newData = []
                for annotation in annot["annotations"]:
                    if annotation['image_id'] == imgId:
                        x = annotation['bbox'][0]
                        y = annotation['bbox'][1]
                        x1 = int(annotation['bbox'][2])+int(annotation['bbox'][0])
                        y1 = int(annotation['bbox'][3])+int(annotation['bbox'][1])
                        category = [cat for cat in annot["categories"] if cat['id'] == annotation['category_id']][0]
                        if "Value" in category['name']:
                            newData.append([(x, y),(x1, y1),"text",category['name']])
                        else:
                            newData.append([(x, y),(x1, y1),"key",category['name']])

            # return redirect(url_for('download_file', name=filename))
            json_object = json.dumps(newData, indent = 4)
            # Writing to sample.json
            with open(os.path.join(app.config['UPLOAD_FOLDER'],folderName, "result.json"), "w") as outfile:
                outfile.write(json_object)
            return redirect(url_for('imageList'))
            
    return redirect(url_for("imageList"))


@app.route('/upload/annotations/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

app.add_url_rule(
    "/upload/annotations/<name>", endpoint="download_file", build_only=True
)
app.add_url_rule(
    "/parse/<foldername>/", endpoint="parse", build_only=True
)
app.debug = True

if __name__ == '__main__':
  
    app.run(host="0.0.0.0",port=5000)