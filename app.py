from flask import Flask, request, render_template
from my_assistant import DataScienceInterviewAssistant  # Import your class
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from database.user_manager import UserManager
# from database.firestoreCRUD import FirestoreCRUD
from firebase_admin import credentials, firestore
import firebase_admin

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


class User(UserMixin):
    def __init__(self, email):
        self.id = email

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@app.route('/')
def home():
    return render_template('home.html')

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
        return redirect(url_for('profile'))
    return render_template('login.html', ids = 'random_ID')

@app.route('/profile')
@login_required
def profile():
    return render_template('index.html', user=current_user, name=session.get('name'))

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