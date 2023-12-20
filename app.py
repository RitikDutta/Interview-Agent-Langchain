from flask import Flask, request, render_template
from my_assistant import DataScienceInterviewAssistant  # Import your class

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    responses = None
    score = None

    if request.method == 'POST':
        question = request.form['question']
        assistant = DataScienceInterviewAssistant("You are a helpful assistant")  # Use your instruction
        responses, score = assistant.conduct_interview(question)

    return render_template('index.html', responses=responses, score=score)

if __name__ == '__main__':
    app.run(debug=True)
