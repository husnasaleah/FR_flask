from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify
import mysql.connector
import cv2
from PIL import Image
import numpy as np
import os
import time
import base64
from datetime import date, datetime
# import datetime
from deepface import DeepFace
 
app = Flask(__name__)
 

 
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="flask_db"
)
mycursor = mydb.cursor()
 
 
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Generate dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def generate_dataset(nbr):
    # face_classifier = cv2.CascadeClassifier("C:/Users/Erik/PycharmProjects/FlaskOpencv_FaceRecognition/resources/haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def face_cropped(img):
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_scale, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # scaling factor=1.3
        # Minimum neighbor = 5
 
        if faces == ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face
 
    cap = cv2.VideoCapture(0)
 
    mycursor.execute("select ifnull(max(img_id), 0) from img_dataset")
    row = mycursor.fetchone()
    lastid = row[0]
 
    img_id = lastid
    max_imgid = img_id + 100
    count_img = 0
 
    while True:
        ret, img = cap.read()
        if face_cropped(img) is not None:
            count_img += 1
            img_id += 1
            face = cv2.resize(face_cropped(img), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
 
            file_name_path = "dataset/"+nbr+"."+ str(img_id) + ".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count_img), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
 
            mycursor.execute("""INSERT INTO `img_dataset` (`img_id`, `img_person`) VALUES
                                ('{}', '{}')""".format(img_id, nbr))
            mydb.commit()
 
            frame = cv2.imencode('.jpg', face)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
 
            if cv2.waitKey(1) == 13 or int(img_id) == int(max_imgid):
                break
    cap.release()
    cv2.destroyAllWindows()
 
 


# Initialize variables
justscanned = False
pause_cnt = 0
cnt = 0

marked_persons = {}

image_folder = "face_img/"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def face_recognition():
    global justscanned
    global pause_cnt
    global cnt
    global marked_persons

    def recognize(img):
        global justscanned
        global pause_cnt
        global cnt

        gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_scale, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        coords = []

        for (x, y, w, h) in faces:
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            coords.append((x, y, w, h))

            if not justscanned:
                # Crop the detected face
                face = img[y:y + h, x:x + w]
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                cropped_image = cv2.resize(face, (200, 200))
 
                # Perform face recognition and emotion analysis using DeepFace
                result = DeepFace.find(img, db_path="dataset/", enforce_detection=False, model_name="VGG-Face")
                objs = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
                if len(result[0]['identity']) > 0:
                    img_id = result[0]['identity'][0].split('/')[-1].split('.')[0]
                    emotion = objs[0]['dominant_emotion']
                    # gender = objs[0]['dominant_gender']
                    # age = objs[0]['age']
                    

                    

                    cnt += 1

                    n = (100 / 30) * cnt
                    w_filled = (cnt / 30) * w

                    if n < 100:
                        cv2.putText(img, str(int(n)) + ' %', (x + 20, y + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 255, 255), 2, cv2.LINE_AA)

                        cv2.rectangle(img, (x, y + h + 40), (x + w, y + h + 50), (255, 255, 0), 2)
                        cv2.rectangle(img, (x, y + h + 40), (x + int(w_filled), y + h + 50), (153, 255, 255), cv2.FILLED)
                        # Get person information from the database
                        try:
                            mycursor.execute("select a.img_person, b.emp_name "
                                        "  from img_dataset a "
                                        "  left join employee b on a.img_person = b.emp_id "
                                        " where img_id = " + img_id)
                            
                            
                            row = mycursor.fetchone()
                            emp_id = row[0]
                            emp_name = row[1]
                        except Exception as e:
                            print("Error executing SQL query or fetching data:", e)
               
                        if int(cnt) == 29:
                            cnt = 0
                            objs = DeepFace.analyze(face, actions=['gender','age'], enforce_detection=False)
                            gender = objs[0]['dominant_gender']
                            age = objs[0]['age']
                            
                        # ตรวจสอบและบันทึกรูปภาพลงในโฟลเดอร์
                        if os.path.exists(image_folder):
                            if cnt % 30 == 0:
                                image_filename = f"{emp_id}_{time.time()}.jpg"
                                image_path = os.path.join(image_folder, image_filename)
                                cv2.imwrite(image_path, cropped_image)
                                print(f"Image saved: {image_filename}")
                                
                                # เก็บที่อยู่ของไฟล์ในฐานข้อมูล
                                try:
                                    mycursor.execute("INSERT INTO detection (det_date,det_person,det_img_path, det_emo, det_age, det_gender) VALUES (%s, %s, %s, %s, %s, %s)",
                                                    (str(date.today()), emp_id,image_filename, emotion, age, gender, ))
                                    mydb.commit()
                                    print("Image path saved in the database.")
                                except Exception as e:
                                    mydb.rollback()  # Rollback changes in case of an error
                                    print("Error executing INSERT:", e)
                        else:
                            print("Image folder not found.")
                            
                        
                            cv2.putText(img, emp_name + ' | '  + '|' + str(age)+"|"+gender, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 255, 255), 2, cv2.LINE_AA)
                            cv2.putText(img, emotion, (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 255, 255), 2, cv2.LINE_AA)
                            time.sleep(1)

                            justscanned = True
                            pause_cnt = 0

                            
                        
                
                else:
                    # คำตอบในกรณี unknown
                    # if not justscanned:
                    cv2.putText(img, 'UNKNOWN', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    # else:
                    #     cv2.putText(img, ' ', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                    if pause_cnt > 80:
                        justscanned = False

        return img

    wCam, hCam = 400, 400
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)

        justscanned = False
        pause_cnt = 0
        img = recognize(img)
            

        # Encode the image and yield the frame
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        key = cv2.waitKey(1)
        if key == 27:
            break

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except FileNotFoundError:
        return None

 
 
@app.route('/')
def home():
    mycursor.execute("select prs_nbr, prs_name, prs_skill, prs_active, prs_added from prs_mstr")
    data = mycursor.fetchall()
 
    return render_template('index.html', data=data)
 
@app.route('/addprsn')
def addprsn():
    mycursor.execute("select ifnull(max(prs_nbr) + 1, 101) from prs_mstr")
    row = mycursor.fetchone()
    nbr = row[0]
    # print(int(nbr))
 
    return render_template('addprsn.html', newnbr=int(nbr))
 
@app.route('/addprsn_submit', methods=['POST'])
def addprsn_submit():
    prsnbr = request.form.get('txtnbr')
    prsname = request.form.get('txtname')
    prsskill = request.form.get('optskill')
 
    mycursor.execute("""INSERT INTO `prs_mstr` (`prs_nbr`, `prs_name`, `prs_skill`) VALUES
                    ('{}', '{}', '{}')""".format(prsnbr, prsname, prsskill))
    mydb.commit()
 
    # return redirect(url_for('home'))
    return redirect(url_for('vfdataset_page', prs=prsnbr))
 
@app.route('/vfdataset_page/<prs>')
def vfdataset_page(prs):
    return render_template('gendataset.html', prs=prs)
 
@app.route('/vidfeed_dataset/<nbr>')
def vidfeed_dataset(nbr):
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(generate_dataset(nbr), mimetype='multipart/x-mixed-replace; boundary=frame')
 
 
@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(face_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')
 
@app.route('/fr_page')
def fr_page():
    mycursor.execute("select a.det_id, a.det_person, b.emp_name, a.det_added "
                     "  from detection a "
                     "  left join employee b on a.det_person = b.emp_id "
                     " where a.det_date = curdate() "
                     " order by 1 desc")
    data = mycursor.fetchall()
 
    return render_template('fr_page.html', data=data)
 
 
@app.route('/countTodayScan')
def countTodayScan():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="flask_db"
    )
    mycursor = mydb.cursor()
 
    mycursor.execute("select count(*) "
                     "  from detection "
                     " where det_date = curdate() ")
    row = mycursor.fetchone()
    rowcount = row[0]
 
    return jsonify({'rowcount': rowcount})
 
 
 
@app.route('/loadData', methods=['GET', 'POST'])
def load_data():
    mycursor.execute("SELECT a.det_person, a.det_img_path, b.emp_name, a.det_emo, a.det_age, a.det_gender "
                     "FROM detection a "
                     "LEFT JOIN employee b ON a.det_person = b.emp_id "
                     "WHERE a.det_date = CURDATE() "
                     "ORDER BY a.det_added DESC")
    data = mycursor.fetchall()

    result = []
    for row in data:
        det_person, det_img_path, emp_name, det_emo, det_age, det_gender = row
        img_path = os.path.join('face_img', det_img_path)
        img_base64 = image_to_base64(img_path)

        if img_base64 is not None:
            result.append((det_person, img_base64, emp_name, det_emo, det_age, det_gender))
    return jsonify(response=result)


 
 
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
