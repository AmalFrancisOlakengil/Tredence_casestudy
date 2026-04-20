import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import PruningNet  # Import your custom architecture

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PruningNet()
model.load_state_dict(torch.load("self_pruning_model.pth", map_location=device))
model.to(device)
model.eval()

# CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def predict(img):
    # Match the transformation used during training
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        prediction = torch.nn.functional.softmax(outputs[0], dim=0)
        confidences = {classes[i]: float(prediction[i]) for i in range(10)}
    return confidences

# Build Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Self-Pruning Neural Network (CIFAR-10)",
    description="This model learns to prune its own weights during training. Upload an image to test its accuracy!"
)

interface.launch()