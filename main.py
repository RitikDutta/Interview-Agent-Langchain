from flask import Flask, request, render_template
from my_assistant_gemini import DataScienceInterviewAssistant  # Import your class
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from database.user_manager import UserManager
# from database.firestoreCRUD import FirestoreCRUD
# from firebase_admin import credentials, firestore
# import firebase_admin
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import hashlib
app = Flask(__name__)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     responses = None
#     score = None

#     if request.method == 'POST':
#         question = request.form['question']
#         assistant = DataScienceInterviewAssistant("You are a helpful assistant")  # Use your instruction
#         responses, score = assistant.conduct_interview(question)

#     return render_template('index.html', responses=responses, score=score)

app = Flask(__name__)
app.secret_key = 'your_secret_key'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'please_login'

# Dummy user database
users = {'user@example.com': {'password': '123456'}}

def add_init_user(user_id='x', name='y', email='z'):
    user_manager = UserManager()
    user_manager.initialize_user(user_id, name, email)

def get_user_preference(user_id):
    user_manager = UserManager()
    preference = user_manager.get_user_setting(current_user.id)
    return preference


uploaded_files = []
def upload_if_needed(pathname: str) -> list[str]:
  path = Path(pathname)
  hash_id = hashlib.sha256(path.read_bytes()).hexdigest()
  try:
    existing_file = genai.get_file(name=hash_id)
    return [existing_file.uri]
  except:
    pass
  uploaded_files.append(genai.upload_file(path=path, display_name=hash_id))
  return [uploaded_files[-1].uri]


class User(UserMixin):
    def __init__(self, email):
        self.id = email

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@app.route('/interview')
def home():
    return render_template('home.html')

@app.route('/')
def mainp():
    return render_template('main.html')

@app.route('/login', methods=['GET', 'POST'])

def login():
    if current_user.is_authenticated:
        return render_template('already_logged_in.html')
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        user_name = request.form.get('name')
        email = request.form.get('email')
        user = User(user_id)
        print(user_id)
        print(user_name)
        print(email)
            
        login_user(user)
        session['name'] = user_name
        flash('Logged in successfully.')
        add_init_user(user_id, user_name, email)
        perference = get_user_preference(current_user.id)
        
        # print(perference.language)
        if perference['interviewer']==None or perference['language']==None:
            return redirect(url_for('settings'))
        return redirect(url_for('profile'))
    return render_template('login.html', ids = 'random_ID')



@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():

    preference = get_user_preference(current_user.id)
    if preference['language'] == 'English':
        assistant = DataScienceInterviewAssistant(instruction="instructions.txt", current_user=current_user.id)
    elif preference['language'] == 'Hindi':
        assistant = DataScienceInterviewAssistant(instruction="instructions_hindi.txt", current_user=current_user.id)

    if request.method == 'POST':
        question = request.form.get('question')
        responses, score = assistant.conduct_interview(question)
        return redirect(url_for('profile'))  
    else:
        score = None  # Define score as None for GET requests

    # Fetch the thread history for both GET and POST requests
    thread_id = assistant.get_thread_id()  # Make sure this gets the correct thread ID
    
    responses = assistant.get_messages(session.get('name'))
    
    return render_template('index.html', responses=responses, score=score, name=session.get('name'))

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template("dashboard.html")

# @app.route('/livec', methods=['GET', 'POST'])
# def livec():

#     if request.method == 'POST':
#         print("RESPONSE WEB: ", request.form.get('question'))
#         return redirect(url_for('livec'))  

#     response = {
#         'question': "What is your greatest strength?",
#         'user_response': "I am highly organized.",
#         'feedback': "Your organizational skills are a strong point.",
#         'score': 2
#     }
    
#     responses = [response]
    
#     return render_template('chat_ui.html', responses=responses, name=session.get('name'))

@app.route('/livec', methods=['GET', 'POST'])
@login_required
def livec():
    response_last = None
    preference = get_user_preference(current_user.id)
    if preference['language'] == 'English':
        assistant = DataScienceInterviewAssistant(instruction="instructions.txt", current_user=current_user.id)
    elif preference['language'] == 'Hindi':
        assistant = DataScienceInterviewAssistant(instruction="instructions_hindi.txt", current_user=current_user.id)    
    show_feedback = False

    if request.method == 'POST':
        # Check if a file was uploaded
        file = request.files.get('cv')
        if file and file.filename:
            filename = secure_filename(file.filename)
            # file.save(os.path.join('path/to/save/files', filename))
            print("File uploaded and saved: ", filename)
            # Assuming the file's path or name is used in the interview
            responses, score = assistant.conduct_interview(filename)
            
            show_feedback = True
        elif 'question' in request.form:
            question = request.form.get('question')
            responses, score = assistant.conduct_interview(question)
            show_feedback = True

        responses = assistant.get_messages(session.get('name'))
        if responses:
            response_last = assistant.convert_json_string_to_dict(responses[0])
            if response_last['feedback'] =="":
                show_feedback = False
    return render_template('chat_ui.html', responses=response_last, show_feedback=show_feedback, name=session.get('name'), preference=preference)



@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if request.method == 'POST':
        choice = request.form.get('choice')
        if 'type' not in session:
            session['type'] = choice  # First choice (interviewer)
            print(f"Interviewer selected: {choice}")
            return redirect(url_for('settings'))
        else:
            # Store the second choice (language) and proceed to confirmation
            session['language'] = choice
            print(f"Language selected: {choice}")
            store_preference(interviewer=session['type'], language=choice)
            return redirect(url_for('confirmation'))

    return render_template('settings.html')

@app.route('/confirmation')
@login_required
def confirmation():
    # Retrieve preferences from session
    interviewer = session.pop('type', None)
    language = session.pop('language', None)
    return render_template('confirmation.html', interviewer=interviewer, language=language)


def store_preference(interviewer, language):
    user_manager = UserManager()
    print(f"Storing preferences: Interviewer - {interviewer}, Language - {language}")
    user_manager.add_or_update_user_setting(user_id = current_user.id,interviewer=interviewer, language=language)
    




@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/please_login')
def please_login():
    return render_template('please_login.html')

if __name__ == '__main__':
    app.run(debug=True)