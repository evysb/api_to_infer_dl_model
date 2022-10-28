from flask import request,jsonify,Flask, Response
import argparse
import logging
import infer_novelty
import classificator
import json
logging.basicConfig(level=logging.DEBUG, format='(%(threadName)-9s) %(message)s',)


def do_inference(autoencoder_model, image_string):
    classificator = classificator.Classificator(autoencoder_model, image_string)
        
    class_result = classificator.get_classification()
    return class_result


app = Flask(__name__)


@app.route('/api/1.0/inference', methods=['POST'])
def inference():
    data = request.form.to_dict(flat=False)
    image = data['image']
    output = do_inference(model, image)
    response = {'class': output}
    return Response(response=json.dumps(response), status=200, mimetype='application/json')



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
    parser.add_argument('--model_path', '-f', help='autoencoder model path', type=str, required=True, default="model.h5")
    args = parser.parse_args()

    model = infer_novelty.get_model(args.model_path, depth=3, filters=(32, 64), latentDim= 1024)
    
    print(f"Running on 'https://{args.address}:{args.port}'.\nPress Ctrl + C to finish.")
    app.run(debug=False, host=args.address, port=args.port)