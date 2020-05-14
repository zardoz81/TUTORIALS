
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, request

app = Flask(__name__)

@app.route('/query-example')
def query_example():
    return 'Todo...'

@app.route('/')
def hello_world():
    return 'Hello from Flask!'

@app.route('/query-example')
def hello_world2():
    return 'Hello from Flask!'


def query_example():
    language = request.args.get('language') #if key doesn't exist, returns None

    return '''<h1>The language value is: {}</h1>'''.format(language)

if __name__ == 'main':
    app.run(debug=False, port=5000) #run app in debug mode on port 5000