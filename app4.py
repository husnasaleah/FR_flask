from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify
import mysql.connector
import cv2
import time
import base64
from datetime import date
import threading
from deepface import DeepFace

app = Flask(__name__)

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="flask_db"
)
mycursor = mydb.cursor()

# ฟังก์ชันสำหรับการทำงานของกล้อง
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
justscanned = False
pause_cnt = 0
cnt = 0

def camera_task():
    global justscanned
    global pause_cnt
    global cnt

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
                face = img[y:y + h, x:x + w]
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                cropped_image = cv2.resize(face, (200, 200))

                cnt += 1
                n = (100 / 30) * cnt
                w_filled = (cnt / 30) * w
                result, objs, gender, age = deepface_task(face, results, lock)
                # deepface_task(face_img, results, lock)

                if len(result[0]['identity']) > 0:
                    img_id = result[0]['identity'][0].split('/')[-1].split('.')[0]
                    emotion = objs[0]['dominant_emotion']
                    gender = gender[0]['dominant_gender']
                    age = age[0]['age']
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


                            # Convert the image to a binary format
                            _, imgS_encoded = cv2.imencode('.jpg', cropped_image)
                            imgS_blob = imgS_encoded.tobytes()

                            _, imgL_encoded = cv2.imencode('.jpg', img)
                            imgL_blob = imgL_encoded.tobytes()

                            # เก็บที่อยู่ของไฟล์ในฐานข้อมูล
                            try:
                                mycursor.execute("INSERT INTO detection (det_date,det_person,det_img_face,det_img_env, det_emo, det_age, det_gender) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                                                (str(date.today()), emp_id,imgS_blob,imgL_blob, emotion, age, gender, ))
                                mydb.commit()
                                print("Image path saved in the database.")
                            except Exception as e:
                                mydb.rollback()  # Rollback changes in case of an error
                                print("Error executing INSERT:", e)

                            cv2.putText(img, emp_name + ' | '  + '|' + str(age)+"|"+gender, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 255, 255), 2, cv2.LINE_AA)
                            cv2.putText(img, emotion, (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 255, 255), 2, cv2.LINE_AA)
                            time.sleep(1)

                            justscanned = True
                            pause_cnt = 0
        return img

    wCam, hCam = 400, 400
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)

        img = recognize(img)

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        key = cv2.waitKey(1)
        if key == 27:
            break

    
# def process_image(face_img, results, lock):
#     deepface_task(face_img, results, lock)
# ฟังก์ชันสำหรับการทำงานของ DeepFace
def deepface_task(face_img, results, lock):
    result = DeepFace.find(face_img, db_path="dataset/", enforce_detection=False, model_name="VGG-Face")
    objs = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
    gender = DeepFace.analyze(face_img, actions=['gender'], enforce_detection=False)
    age = DeepFace.analyze(face_img, actions=['age'], enforce_detection=False)

    # สำหรับการเขียนผลลัพธ์ลงในตัวแปรร่วมใช้งาน
    with lock:
        results.append((result, objs, gender, age))

def main():
    # สร้างรายการของภาพหน้าตาที่ต้องการประมวลผล
    face_images = [...]

    results = []  # รายการเก็บผลลัพธ์
    lock = threading.Lock()  # เพื่อป้องกันการเข้าถึงข้อมูลร่วมใช้งาน

    threads = []
    for face_img in face_images:
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
    main()

    app.run(host='127.0.0.1', port=5001, debug=True)






