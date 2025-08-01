import numpy as np
import pickle
import gradio as gr

class ActiveLearningInference:
    def __init__(self, model_path):
        self.load_model(model_path)
    
    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.layer = model_data['layer']
        self.N_input = model_data['N_input']
        self.N_output = model_data['N_output']
        self.lb_input = model_data['lb_input']
        self.ub_input = model_data['ub_input']
        self.n_ensemble = model_data['n_ensemble']
        self.weights_and_biases = model_data['weights_and_biases']
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = np.tanh(np.dot(H, W) + b)
        W = weights[-1]
        b = biases[-1]
        Y = self.softplus(np.dot(H, W) + b)
        return Y
    
    def softplus(self, x):
        return np.log(1 + np.exp(np.clip(x, -500, 500)))

    def normalize_input(self, input_data):
        return 2.0 * (input_data - self.lb_input) / (self.ub_input - self.lb_input) - 1.0
    
    def predict_single_model(self, input_data, model_idx):
        normalized_input = self.normalize_input(input_data)
        weights = self.weights_and_biases[model_idx]['weights']
        biases = self.weights_and_biases[model_idx]['biases']
        prediction = self.neural_net(normalized_input, weights, biases)
        return prediction

    def predict(self, input_data):
        input_array = np.array(input_data).reshape(1, -1)
        if input_array.shape[1] != self.N_input:
            raise ValueError(f"Expected {self.N_input} features, got {input_array.shape[1]}")
        
        predictions = []
        for i in range(self.n_ensemble):
            pred = self.predict_single_model(input_array, i)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return float(mean_pred.flatten()[0]), float(std_pred.flatten()[0])

# 实例化模型（模型文件名必须与你上传的一致）
inference_model = ActiveLearningInference("final_model_weights.pkl")

# 用于Gradio界面的预测函数
def gradio_predict(x1, x2, x3):
    try:
        input_features = [x1, x2, x3]
        # 使用predict方法获取均值和标准差，但只返回预测值
        mean_pred, std_pred = inference_model.predict(input_features)
        
        return f"{mean_pred:.6f}"
    except Exception as e:
        return f"错误: {str(e)}"

# 启动 Gradio 接口
iface = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Number(label="Initial void ratio", info="初始孔隙比"),
        gr.Number(label="Liquid limit (%)", info="液限百分比"),
        gr.Number(label="Plastic limit (%)", info="塑限百分比")
    ],
    outputs=[
        gr.Textbox(label="Predicted_Cc")  # 改回单个文本框输出
    ],
    title="Uncertainty predictor based on integrated neural networks",
    description="Input initial void, liquid limit (%), and plastic limit (%) values and the prediction Cc value of the neural network model will be returned.",
)

# 为Hugging Face Spaces部署
if __name__ == "__main__":
    iface.launch()