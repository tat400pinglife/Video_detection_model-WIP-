import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

class DeepfakeGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, inputs):
        self.model.zero_grad()
        output = self.model(*inputs)
        target = output[0]
        target.backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu().numpy(), 0)
        if np.max(heatmap) > 0: heatmap /= np.max(heatmap)
        return heatmap

def visualize_gradcam(rgb_frame, heatmap, title="Grad-CAM Attention"):
    heatmap = cv2.resize(heatmap, (rgb_frame.shape[1], rgb_frame.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    frame_uint8 = np.uint8(255 * rgb_frame)
    overlay = cv2.addWeighted(frame_uint8, 0.6, heatmap_color, 0.4, 0)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.title(title)
    plt.axis("off")
    plt.show()