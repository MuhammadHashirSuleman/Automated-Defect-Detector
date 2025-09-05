import sqlite3
import hashlib
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import re
from detect import detect_defects
from preprocess import preprocess_image
from predict_cracks import detect_cracks

app = Flask(__name__, template_folder='../templates')
app.secret_key = 'supersecretkey'

# Configuration - Fixed path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # Serve the index.html page regardless of login status
    return render_template('index.html')

@app.route('/home')
def home():
    # Redirect to main if logged in, otherwise to index
    if 'username' in session:
        return redirect(url_for('main'))
    return redirect(url_for('index'))

def init_db():
    conn = sqlite3.connect('database/users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    conn.close()

init_db()

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        
        if not username:
            flash('Username is required.', 'error')
            return render_template('signup.html')
        if not password:
            flash('Password is required.', 'error')
            return render_template('signup.html')
        
        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'error')
            return render_template('signup.html')
        if not re.search(r'\d', password):
            flash('Password must contain at least one number.', 'error')
            return render_template('signup.html')
        if not re.search(r'[!@#$%^&*(),.?\":{}|<>]', password):
            flash('Password must contain at least one special character.', 'error')
            return render_template('signup.html')
        
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        conn = sqlite3.connect('database/users.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            conn.commit()
            flash('Signup successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists.', 'error')
        finally:
            conn.close()
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        
        if not username:
            flash('Username is required.', 'error')
            return render_template('login.html')
        if not password:
            flash('Password is required.', 'error')
            return render_template('login.html')
        
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        conn = sqlite3.connect('database/users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        if user:
            if user[1] == hashed_password:
                session['username'] = username
                flash('Login successful!', 'success')
                return redirect(url_for('main'))
            else:
                flash('Incorrect password.', 'error')
        else:
            flash('Username not found.', 'error')
        conn.close()
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/main', methods=['GET', 'POST'])
def main():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Verify the file was saved correctly
            if not os.path.exists(filepath):
                flash('Error saving file', 'error')
                return redirect(request.url)
            
            detection_type = request.form.get('detection_type', 'pipeline')
            
            if detection_type == 'crack':
                detected_img, defects = detect_cracks(filepath)
            else:
                preprocessed_img = preprocess_image(filepath)
                detected_img, defects = detect_defects(filepath, preprocessed_img, detection_type)
            
            detected_filename = f"detected_{filename}"
            detected_path = os.path.join(app.config['UPLOAD_FOLDER'], detected_filename)
            cv2.imwrite(detected_path, detected_img)
            
            # Verify the processed image was saved
            if not os.path.exists(detected_path):
                flash('Error saving processed image', 'error')
                return redirect(request.url)
            
            processed_image_url = url_for('uploaded_file', filename=detected_filename)
            
            report = f"Defect Count: {len(defects)}\n"
            for defect in defects:
                report += f"Type: {defect['type']}, Confidence/Area: {defect['conf']:.2f}\n"
            
            return render_template('result.html', image=processed_image_url, report=report)
        else:
            flash('Allowed file types are png, jpg, jpeg', 'error')
    return render_template('main.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)