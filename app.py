import flask
import pickle
import pandas as pd
# Use pickle to load in the pre-trained model.
with open(f'RandomForest.pkl', 'rb') as f:
    model = pickle.load(f)
app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))

    if flask.request.method == 'POST':
        N = flask.request.form['N']
        P = flask.request.form['P']
        K = flask.request.form['K']
        temperature = flask.request.form['temperature']
        humidity = flask.request.form['humidity']
        ph = flask.request.form['ph']
        rainfall = flask.request.form['rainfall']
        input_variables = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                       columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'],
                                       dtype=float)
        prediction = model.predict(input_variables)[0]
        return flask.render_template('main.html',
                                     original_input={
                                                     'N': N,
                                                     'P': P,
                                                     'K': K,
                                                     'temperature':temperature,
                                                     'humidity':humidity,
                                                     'ph':ph,
                                                     'rainfall': rainfall},
                                     result=prediction,
                                     )
if __name__ == '__main__':
    app.run()
