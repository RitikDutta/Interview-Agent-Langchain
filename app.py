from flask import Flask, request, render_template
from your_script import DataScienceInterviewAssistant  # Import your class

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form['question']
        assistant = DataScienceInterviewAssistant(instruction)  # Use your instruction
        responses, score = assistant.conduct_interview(question)
        return render_template('result.html', responses=responses, score=score)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
