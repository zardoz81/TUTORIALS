from flask import Flask, render_template
import random

app = Flask(__name__)

@app.route("/")
def index():
	name = 'Roman'
	age = '10'
	return render_template("asdf.html", name=name, age=age)

@app.route("/hello")
def hello():
    return "Hello World!"

@app.route("/members")
def members():
    return "Members"

@app.route("/foo", methods=['GET', 'POST'])
def move_forward():
	print('IT clicked!!!')
	num = random.sample(range(100),1)
	text = 'Lucky number of the day'
	return render_template("asdf.html", name=text, age=str(num))

if __name__ == "__main__":
    app.run()
