from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import json
from QA_system import QA_system
import threading

qa_system = None
hello = ''



class MyFlaskApp(Flask):
    def on_startup(self):
        self.qa_system = QA_system()
        hello = 'ghello woprold'
        self.qa_system.initialize()
        print("QA_SYSTEM Initialized")

    def run(self, host='0.0.0.0', port='5001', debug=None, load_dotenv=True, **options):
        self.on_startup()
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, load_dotenv=load_dotenv, **options)


app = MyFlaskApp(__name__)

CORS(app, supports_credentials=True)

@app.route("/")
@cross_origin(supports_credentials=True)
def hello_world():
    return render_template('index.html')

@app.route("/query")
@cross_origin(supports_credentials=True)
def handle_query():
    # print(request.args['searchQuery'])
    qa_response = query_qa_system(request.args['searchQuery'])
    response = jsonify([{'answer':(j.data, j.score)} for i, j in enumerate(qa_response['reader']['answers'])])
    # response = jsonify([{'success': True}])
    response.headers.add('Content-Type', 'application/json')
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'PUT, GET, POST, DELETE, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Expose-Headers', 'Content-Type,Content-Length,Authorization,X-Pagination')
    return response

def query_qa_system(query):
   print(query)
   result = app.qa_system.query(query)
   return result


app.run(host='0.0.0.0', port=5001, debug=False)