from flask import Flask, request, render_template, render_template_string, Response
from my_assistant_gemini import DataScienceInterviewAssistant  # Import your class
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from database.user_manager import UserManager
from fun_plugins.spotify import Spotify
import google.generativeai as genai
from flask import Flask, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import google.generativeai as genai
from dotenv import load_dotenv
# from database.firestoreCRUD import FirestoreCRUD
# from firebase_admin import credentials, firestore
# import firebase_admin
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
from pathlib import Path
import hashlib
from file_handler.google_cloud_storage import Handler
from google.cloud import storage
import requests
import json
from agent import stream_agent_events, get_profile



app = Flask(__name__)


app.secret_key = 'your_secret_key'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

BUCKET_NAME = 'asia.artifacts.interview-mentor-408213.appspot.com'

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'interview-mentor-408213-f5ba84c00ba7.json'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     responses = None
#     score = None

#     if request.method == 'POST':
#         question = request.form['question']
#         assistant = DataScienceInterviewAssistant("You are a helpful assistant")  # Use your instruction
#         responses, score = assistant.conduct_interview(question)

#     return render_template('index.html', responses=responses, score=score)




login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'please_login'

# Dummy user database
users = {'user@example.com': {'password': '123456'}}

def add_init_user(user_id='x', name='y', email='z', password="_"):
    user_manager = UserManager()
    user_manager.initialize_user(user_id, name, email, password)

def reset_preference(user_id):
    user_manager = UserManager()
    user_manager.reset_chat_and_preference(user_id)

def is_user(user_id):
    user_manager = UserManager()
    return user_manager.is_user(user_id)
def get_name(user_id):
    user_manager = UserManager()
    return user_manager.get_name(user_id)
def check_password(user_id, password):
    user_manager = UserManager()
    return user_manager.check_password(user_id, password)

def get_user_preference(user_id):
    user_manager = UserManager()
    preference = user_manager.get_user_setting(current_user.id)
    return preference


def get_current_song():
    spotify = Spotify()
    load_dotenv()
    client_id = os.getenv("CLIENT_ID") 
    client_secret = os.getenv("CLIENT_SECRET") 
    refresh_token = os.getenv("REFRESH_TOKEN")
    return spotify.get_song(client_id, client_secret, refresh_token)


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

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/home')
def mainp():
    current_song, image = get_current_song()
    print("SONG: ", current_song)
    if current_song == "not playing":
        current_song = {}
        current_song['song_name'] = "NONE"
    return render_template('main.html', current_song=current_song, image_url = image)

@app.route('/login', methods=['GET', 'POST'])

def login():
    if current_user.is_authenticated:
        return render_template('already_logged_in.html')
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        password = request.form.get('password')
        user = User(user_id)
        user_name = request.form.get('name')
        email = request.form.get('email')
        # signup
        try:
            if request.form['form_action'] == 'signup_submit':
                print("SIGN IN REQUEST")
                print(user_id)
                print(user_name)
                print(email)
                
                if not is_user(user_id):
                    login_user(user)
                    session['name'] = user_name
                    flash('Logged in successfully.')
                    add_init_user(user_id, user_name, email, password)
                    perference = get_user_preference(current_user.id)
                    
                    # print(perference.language)
                    if perference['interviewer']==None or perference['language']==None:
                        return redirect(url_for('settings'))
                    return redirect(url_for('livec'))
                else:
                    return render_template('user_present.html')
            # login
            elif request.form['form_action'] == 'login_submit':
                user_name = get_name(user_id)
                print("LOGIN IN REQUEST")
                if is_user(user_id):
                    if not check_password(user_id, password):
                        return render_template("wrong_password.html")
                    login_user(user)
                    session['name'] = user_name
                    flash('Logged in successfully.')
                    print('user_name', user_id)
                    if user_id == 'test_account':
                        reset_preference('test_account')
                    perference = get_user_preference(current_user.id)
                    # print(perference.language)
                    if perference['interviewer']==None or perference['language']==None:
                        return redirect(url_for('settings'))
                    return redirect(url_for('livec'))
                else:
                    return render_template('user_not_present.html')
            # elif request.form['form_action'] == 'test login':
            #     user_name = 'test_account'
            #     print("TEST LOGIN IN REQUEST")
            #     if is_user('test_account'):
            #         if not check_password('test_account', 'testpassword'):
            #             return render_template("wrong_password.html")
            #         login_user(user)
            #         session['name'] = user_name
            #         flash('Logged in successfully.')
            #         perference = get_user_preference(current_user.id)
            #         # print(perference.language)
            #         if perference['interviewer']==None or perference['language']==None:
            #             return redirect(url_for('settings'))
            #         return redirect(url_for('livec'))
            #     else:
            #         return render_template('user_not_present.html')
        except:
            login_user(user)
            session['name'] = user_name
            flash('Logged in successfully.')
            add_init_user(user_id, user_name, email)
            perference = get_user_preference(current_user.id)
            
            # print(perference.language)
            if perference['interviewer']==None or perference['language']==None:
                return redirect(url_for('settings'))
            return redirect(url_for('livec'))
               
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

def dashboard():
    return render_template("dashboard.html", data=[2,1,10,6,3])

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
        try:
            handler = Handler()
            
            # check if a file uploaded
            file = request.files.get('cv')
            # handling the file uploading
            if file:
                if allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    
                    # upload to GCS
                    gcs_uri = handler.upload_to_gcs(file, filename)
                    
                    # download the file from GCS to local temporary storage
                    temp_filepath = f"/tmp/{filename}"
                    client = storage.Client()
                    bucket = client.bucket(BUCKET_NAME)
                    blob = bucket.blob(filename)
                    blob.download_to_filename(temp_filepath)
                    
                    # upload to Gemini
                    mime_type = 'image/jpeg' if filename.lower().endswith(('jpg', 'jpeg')) else 'image/png'
                    uploaded_file = assistant.upload_file_to_gemini(temp_filepath, mime_type)
                    
                    # delete the file from GCS
                    handler.delete_from_gcs(filename)
                    
                    # delete the temporary file
                    os.remove(temp_filepath)
                    
                    responses, score = assistant.conduct_interview(uploaded_file)
                    show_feedback = True
                else:
                    return "NOT ALLOWED"    
                
            elif 'question' in request.form:
                question = request.form['question']
                responses, score = assistant.conduct_interview(question)
                show_feedback = True
        except Exception as e:
            return f"server timeout please wait 5 second for server to respond and retry {e}"
 
    responses = assistant.get_messages(session.get('name'))
    try:
        if responses:
            response_last = assistant.convert_json_string_to_dict(responses[0])
            if response_last['feedback'] == "":
                show_feedback = False
        print(response_last)
        print("TYPE", type(response_last))
    except TypeError:
        print(response_last)
        print("TYPE", type(response_last))
        return "Type Error"


    return render_template('chat_ui.html', responses=response_last, show_feedback=show_feedback, name=session.get('name'), preference=preference, test="4")



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

    preference = get_user_preference(current_user.id)
    if preference['language'] == 'English':
        assistant = DataScienceInterviewAssistant(instruction="instructions.txt", current_user=current_user.id)
    elif preference['language'] == 'Hindi':
        assistant = DataScienceInterviewAssistant(instruction="instructions_hindi.txt", current_user=current_user.id)

    interviewer = session.pop('type', None)
    language = session.pop('language', None)
    assistant.init_chat(language)
    print("CHAT INIT FROM MAIN")
    return render_template('confirmation.html', interviewer=interviewer, language=language)


def store_preference(interviewer, language):
    user_manager = UserManager()
    print(f"Storing preferences: Interviewer - {interviewer}, Language - {language}")
    user_manager.add_or_update_user_setting(user_id = current_user.id,interviewer=interviewer, language=language)
    

@app.route('/tts', methods=['GET', 'POST'])
def tts():
    return render_template('text_to_speech.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/please_login')
def please_login():
    return render_template('please_login.html')


@app.route('/langraph')
def langraph():
    return render_template('agent_lang.html')

@app.route("/chat", methods=["POST"])
def chat():
    """API endpoint to handle streaming chat messages using Server-Sent Events."""
    data = request.json
    user_id = data.get("user_id")
    thread_id = data.get("thread_id")
    user_message = data.get("message")

    if not all([user_id, thread_id, user_message]):
        return jsonify({"error": "Missing user_id, thread_id, or message"}), 400

    print(f"Streaming response for user '{user_id}': {user_message}")

    # This generator function will stream data to the client
    def generate():
        try:
            for event_data in stream_agent_events(user_id, thread_id, user_message):
                # Format as a Server-Sent Event
                yield f"data: {event_data}\n\n"
        except Exception as e:
            print(f"An error occurred during streaming: {e}")
            error_event = '{"type": "error", "content": "An internal error occurred."}'
            yield f"data: {error_event}\n\n"

    # Return a streaming response
    return Response(generate(), mimetype='text/event-stream')

@app.route("/get_profile", methods=["GET"])
def profile_data():
    """API endpoint to fetch the current user profile."""
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400
    
    profile = get_profile(user_id)
    return jsonify(profile.model_dump())

if __name__ == '__main__':
    app.run(debug=True)