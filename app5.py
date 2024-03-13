from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify
import mysql.connector
import cv2
import numpy as np
import time
import base64
from datetime import date
import threading
from deepface import DeepFace
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # ใส่เพื่อทำให้ Flask รับ request แบบ Cross-Origin Resource Sharing (CORS)

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="flask_db"
)
mycursor = mydb.cursor()
def main(img):
    # สร้างรายการของภาพหน้าตาที่ต้องการประมวลผล
    # img = [...]
    img_np = np.array(img)
    
    # Ensure correct data type (uint8)
    img_np = img_np.astype(np.uint8)

    results = []  # รายการเก็บผลลัพธ์
    lock = threading.Lock()  # เพื่อป้องกันการเข้าถึงข้อมูลร่วมใช้งาน

    threads = []
    for face_img in img:
        thread = threading.Thread(target=deepface_task, args=(face_img, results, lock))
        thread.start()
        threads.append(thread)

    # รอให้ทุก thread ทำงานเสร็จสมบูรณ์
    for thread in threads:
        thread.join()

    # ประมวลผลผลลัพธ์ที่ได้รับ
    for result in results:
        # ทำอะไรกับผลลัพธ์ต่อไปนี้
        pass

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("cascade : " +cv2.data.haarcascades)

def deepface_task(face_img, results, lock):
    # Check data type of face_img
    if not isinstance(face_img, np.ndarray):
        # Convert face_img to NumPy array if necessary
        face_img = np.array(face_img)

    # Ensure correct data type (uint8)
    if face_img.dtype != np.uint8:
        # Convert to uint8 if necessary
        face_img = face_img.astype(np.uint8)
    
    
    # ตรวจจับใบหน้าโดยใช้ OpenCV Haar cascade
    faces = face_cascade.detectMultiScale(face_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:  # ตรวจสอบว่ามีใบหน้าในภาพหรือไม่
        result = DeepFace.find(face_img, db_path="dataset/", enforce_detection=False, model_name="VGG-Face")
        emotion = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        gender = DeepFace.analyze(face_img, actions=['gender'], enforce_detection=False)
        age = DeepFace.analyze(face_img, actions=['age'], enforce_detection=False)
        img_id = result[0]['identity'][0].split('/')[-1].split('.')[0]
        print("img_id: ", img_id)
        with lock:
            results.append((result, objs, gender, age))

            # เมื่อได้ผลลัพธ์แล้ว สร้างคำสั่ง SQL สำหรับเพิ่มข้อมูลลงในฐานข้อมูล
            for res in results:
                result, objs, gender, age = res
                try:
                    mycursor.execute("INSERT INTO detection (det_date, det_emo, det_age, det_gender) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                                    (str(date.today()), emotion, age, gender, ))
                    mydb.commit()
                    print("Image path saved in the database.")
                except Exception as e:
                    mydb.rollback()  # Rollback changes in case of an error
                    print("Error executing INSERT:", e)
    else:
        # ไม่มีใบหน้าในภาพ ไม่ต้องทำอะไร
        pass





justscanned = False

def camera_task():
    def recognize(img):
        gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_scale, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        coords = []

        for (x, y, w, h) in faces:
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            coords.append((x, y, w, h))

            if not justscanned:
                face = img[y:y + h, x:x + w]  
                face = gray_scale[y:y + h, x:x + w]
                face_gray = cv2.resize(face, (200, 200))  
                # แปลงภาพให้อยู่ในรูปแบบของ NumPy array
                face_np = np.array(face_gray)
                
                main(face_np)  # เรียกใช้ main() ที่มีการแก้ไข


        return img




    wCam, hCam = 400, 400
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)

        img = recognize(img)

        # Encode the image and yield the frame
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        key = cv2.waitKey(1)
        if key == 27:
            break

@app.route('/')
def home():
    mycursor.execute("select a.det_id, a.det_person, b.emp_name, a.det_added "
                     "  from detection a "
                     "  left join employee b on a.det_person = b.emp_id "
                     " where a.det_date = curdate() "
                     " order by 1 desc")
    data = mycursor.fetchall()

    return render_template('index.html', data=data)

@app.route('/video_feed')
def video_feed():
    return Response(camera_task(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/countTodayScan')
def countTodayScan():
    mycursor.execute("select count(*) "
                     "  from detection "
                     " where det_date = curdate() ")
    row = mycursor.fetchone()
    rowcount = row[0]

    return jsonify({'rowcount': rowcount})

@app.route('/loadData', methods=['GET', 'POST'])
def load_data():
    mycursor.execute("SELECT a.det_person, a.det_img_face, IFNULL(b.emp_name, 'unknown') AS emp_name, a.det_emo, a.det_age, a.det_gender "
                     "FROM detection a "
                     "LEFT JOIN employee b ON a.det_person = b.emp_id "
                     "WHERE a.det_date = CURDATE() "
                     "ORDER BY a.det_added DESC")
    data = mycursor.fetchall()

    result = []
    for row in data:
        det_person, det_img_face, emp_name, det_emo, det_age, det_gender = row
        img_base64 = base64.b64encode(det_img_face).decode('utf-8')
        result.append((det_person, img_base64, emp_name, det_emo, det_age, det_gender))

    return jsonify(response=result)

if __name__ == "__main__":
    

    app.run(host='127.0.0.1', port=5001, debug=True)
