import flask

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return 'Home page'


@app.route('/areas', methods=['GET'])
def areas():
    return 'Areas page'


app.run()