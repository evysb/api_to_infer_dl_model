from flask import request,jsonify,Flask, Response
import argparse
import logging
import classificator
import json
logging.basicConfig(level=logging.DEBUG, format='(%(threadName)-9s) %(message)s',)
import tensorflow

print(tensorflow.__version__)

app = Flask(__name__)




############ INICIO - Pode criar funcoes auxiliares para deixar codigo mais organizado

def do_inference(model, image_string):
    classificator_obj = classificator.Classificator(model, image_string)
        
    class_result, predIdxs = classificator_obj.get_classification()
    return class_result, predIdxs

############ FIM - Pode criar funcoes auxiliares para deixar codigo mais organizado






############ INICIO - Pode editar e criar novos endpoints

@app.route('/api/1.0/inference', methods=['POST']) # @ decorator - em qual link essa informação vai aparecer
def inference():
    data = request.form.to_dict(flat=False)
    image = data['image']
    output1, output2 = do_inference(model, image) # onde de fato faz o processamento
    response = {'class': output1, 'probabilidades': str(output2)}
    return Response(response=json.dumps(response), status=200, mimetype='application/json')

############ FIM - Pode editar e criar novos endpoints






def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.route('/api/1.0/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


if __name__=='__main__':
    parser = argparse.ArgumentParser('Similarity API')
    parser.add_argument('--address', '-a', help='Host address.', type=str, required=False, default='0.0.0.0')
    parser.add_argument('--port', '-p', help='Host port.', type=int, required=False, default=5000)
    parser.add_argument('--model_path', '-f', help='model path', type=str, required=False, default="model.h5")
    args = parser.parse_args()

    model = tensorflow.keras.models.load_model(args.model_path) #onde carrega o modelo em memoria
    
    print(f"Running on 'https://{args.address}:{args.port}'.\nPress Ctrl + C to finish.")
    app.run(debug=False, host=args.address, port=args.port)
