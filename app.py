import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models.resnet import Bottleneck
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Architecture 
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBottleneck(Bottleneck):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, 
                 base_width=64, dilation=1, norm_layer=None, se_reduction=16):
        super(SEBottleneck, self).__init__(inplanes, planes, stride, downsample, 
                                         groups, base_width, dilation, norm_layer)
        self.se = SELayer(planes * self.expansion, reduction=se_reduction)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out

def get_seresnext50(num_classes=1, se_reduction=16):
    """Create SE-ResNeXt50 model"""
    model = models.resnext50_32x4d(pretrained=True)
    base_width = model.base_width
    
    def replace_bottlenecks(module, se_reduction_ratio, base_width):
        for name, child_module in module.named_children():
            if isinstance(child_module, Bottleneck):
                inplanes = child_module.conv1.in_channels
                planes = child_module.conv3.out_channels // child_module.expansion
                stride = child_module.stride
                downsample = child_module.downsample
                groups = child_module.conv2.groups
                dilation = child_module.conv2.dilation[0]
                
                new_bottleneck = SEBottleneck(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    groups=groups,
                    base_width=base_width,
                    dilation=dilation,
                    se_reduction=se_reduction_ratio
                )
                
                new_bottleneck.load_state_dict(child_module.state_dict(), strict=False)
                setattr(module, name, new_bottleneck)
            else:
                replace_bottlenecks(child_module, se_reduction_ratio, base_width)
    
    replace_bottlenecks(model, se_reduction, base_width)
    
    # Replace final layer for binary classification (single output)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

# Load the trained model
@torch.no_grad()
def load_model():
    """Load the trained SE-ResNeXt50 model"""
    model_path = 'best_seresnext50_model.pth' # Relative path
    
    # Create model
    model = get_seresnext50(num_classes=1, se_reduction=16)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        
        # Remove 'module.' prefix if present
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model

# Initialize model
print("Loading DeepStroke Model...")
model = load_model()
print(f"Model loaded successfully on {device}")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Optimal threshold from evaluation (Youden's Index)
OPTIMAL_THRESHOLD = 0.4902

def predict_stroke(image):
    """
    Predict stroke probability from brain CT image
    
    Args:
        image: PIL Image or numpy array
    
    Returns:
        tuple: (prediction_text, probability, confidence, detailed_analysis, visualization)
    """
    try:
        # Check if image is None
        if image is None:
            return "No image provided", "N/A", "N/A", "Please upload an image to analyze.", None
        
        print(f"DEBUG: Processing image type: {type(image)}")  # Debug print
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print(f"DEBUG: Image size: {image.size}, mode: {image.mode}")  # Debug print
        
        # Preprocess image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(image_tensor)
            probability = torch.sigmoid(output).item()
        
        # Use optimal threshold for prediction
        prediction = probability >= OPTIMAL_THRESHOLD
        confidence = max(probability, 1 - probability)
        
        # Create prediction text with clinical interpretation
        if prediction:
            risk_level = "HIGH RISK" if probability > 0.8 else "MODERATE RISK" if probability > 0.6 else "ELEVATED RISK"
            prediction_text = f"🚨 **STROKE DETECTED** - {risk_level}"
            color = "#ff4444"
            recommendation = "⚠️ **URGENT**: Immediate medical attention required. Contact emergency services."
        else:
            if probability < 0.2:
                risk_level = "LOW RISK"
                recommendation = "✅ **Low stroke probability detected**. Continue routine medical care."
            elif probability < 0.4:
                risk_level = "MILD RISK"
                recommendation = "⚠️ **Mild concern**. Consider consulting with a neurologist."
            else:
                risk_level = "UNCERTAIN"
                recommendation = "⚠️ **Borderline case**. Additional imaging or clinical assessment recommended."
            
            prediction_text = f"✅ **NO STROKE DETECTED** - {risk_level}"
            color = "#44ff44"
        
        # Detailed analysis
        detailed_analysis = f"""
        **🔬 DETAILED ANALYSIS**
        
        **Prediction:** {prediction_text}
        **Stroke Probability:** {probability:.1%}
        **Model Confidence:** {confidence:.1%}
        **Risk Assessment:** {risk_level}
        
        **Clinical Threshold:** {OPTIMAL_THRESHOLD:.1%} (Optimized using Youden's Index)
        **Model Performance:** ROC-AUC 0.98+ on validation data
        
        **⚕️ Clinical Recommendation:**
        {recommendation}
        
        **📋 Important Notes:**
        - This AI model is for assistance only and should not replace professional medical diagnosis
        - Always consult with qualified medical professionals for definitive diagnosis
        - Consider patient clinical history and symptoms alongside AI predictions
        """
        
        # Create visualization
        visualization = create_prediction_visualization(probability, prediction, confidence)
        
        return prediction_text, f"{probability:.1%}", f"{confidence:.1%}", detailed_analysis, visualization
        
    except Exception as e:
        import traceback
        error_msg = f"Error during prediction: {str(e)}\n\nTraceback: {traceback.format_exc()}"
        print(f"DEBUG: Prediction error - {error_msg}")  # Debug print
        return f"⚠️ **PREDICTION ERROR**\n\n{str(e)}", "Error", "N/A", error_msg, None

def create_prediction_visualization(probability, prediction, confidence):
    """Create a simple and clean visualization with essential information"""
    try:
        # Simple color scheme
        color_safe = '#28a745'      # Green
        color_warning = '#ffc107'   # Yellow  
        color_danger = '#dc3545'    # Red
        
        # Determine colors based on prediction
        gauge_color = color_danger if prediction else color_safe
        
        # Create a single-panel dashboard with just the probability gauge
        fig = go.Figure()
        
        # Main probability gauge - simplified
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=probability * 100,
                title={
                    'text': "Stroke Probability", 
                    'font': {'size': 18, 'family': 'Arial'}
                },
                number={
                    'font': {'size': 32, 'color': gauge_color, 'family': 'Arial Black'},
                    'suffix': '%'
                },
                delta={
                    'reference': OPTIMAL_THRESHOLD * 100,
                    'increasing': {'color': color_danger},
                    'decreasing': {'color': color_safe}
                },
                gauge={
                    'axis': {
                        'range': [0, 100], 
                        'tickfont': {'size': 14}
                    },
                    'bar': {
                        'color': gauge_color, 
                        'thickness': 0.8
                    },
                    'steps': [
                        {'range': [0, OPTIMAL_THRESHOLD * 100], 'color': "rgba(40, 167, 69, 0.2)"},
                        {'range': [OPTIMAL_THRESHOLD * 100, 100], 'color': "rgba(220, 53, 69, 0.2)"}
                    ],
                    'threshold': {
                        'line': {'color': "#000", 'width': 3},
                        'thickness': 0.8,
                        'value': OPTIMAL_THRESHOLD * 100
                    }
                }
            )
        )
        
        # Add simple status annotation
        status_text = "⚠️ STROKE DETECTED" if prediction else "✅ NO STROKE DETECTED"
        status_color = color_danger if prediction else color_safe
        
        fig.add_annotation(
            x=0.5, y=0.1,
            xref="paper", yref="paper",
            text=f"<b style='color:{status_color};font-size:16px'>{status_text}</b><br>" +
                 f"<span style='font-size:12px'>Confidence: {confidence:.0%}</span><br>" +
                 f"<span style='font-size:11px'>Threshold: {OPTIMAL_THRESHOLD:.0%}</span>",
            showarrow=False,
            font={'size': 14, 'color': status_color},
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=status_color,
            borderwidth=2,
            borderpad=10,
            xanchor="center"
        )
        
        # Simple layout configuration
        fig.update_layout(
            height=400,  # Much smaller height
            showlegend=False,
            title={
                'text': "🧠 Stroke Detection Result",
                'x': 0.5,
                'y': 0.95,
                'font': {'size': 24, 'family': 'Arial Black', 'color': '#2C3E50'}
            },
            font={'size': 12, 'family': 'Arial'},
            margin=dict(t=60, b=40, l=40, r=40),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    except Exception as e:
        import traceback
        print(f"DEBUG: Visualization error - {str(e)}\n{traceback.format_exc()}")
        # Return a simple figure on error
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"Visualization Error: {str(e)}",
            showarrow=False,
            font={'size': 16, 'color': 'red'}
        )
        return fig

def create_model_info():
    """Create enhanced model information display with interactive elements"""
    info_html = """
    <div class="model-info">
        <h2>🧠 DeepStroke AI - Advanced Brain CT Stroke Detection</h2>
        <div class="model-specs-grid">
            <div>
                <h3>🏗️ Model Architecture</h3>
                <ul>
                    <li><strong>🔬 Network:</strong> SE-ResNeXt50</li>
                    <li><strong>📐 Input Size:</strong> 224×224 RGB</li>
                    <li><strong>⚙️ Parameters:</strong> ~25M trained</li>
                    <li><strong>🎯 Threshold:</strong> 49.02% (Optimized)</li>
                    <li><strong>🧮 SE Ratio:</strong> 16 (Attention)</li>
                </ul>
            </div>
            <div>
                <h3>📊 Performance Metrics</h3>
                <ul>
                    <li><strong>🎯 ROC-AUC:</strong> 0.98+ (Excellent)</li>
                    <li><strong>🔍 Sensitivity:</strong> High stroke detection</li>
                    <li><strong>✅ Specificity:</strong> Low false alarms</li>
                    <li><strong>📋 Validation:</strong> External datasets</li>
                    <li><strong>⚡ Speed:</strong> <1s inference</li>
                </ul>
            </div>
        </div>
        <div style="margin-top: 20px; padding-top: 15px; border-top: 2px solid rgba(255,255,255,0.3);">
            <p style="text-align: center; margin: 0; font-size: 1.1em;">
                <strong>🚀 Latest Model Version 1.0</strong> | 
                <em>Trained on 10,000+ validated NON-CONTRAST brain CT scans</em>
            </p>
            <p style="text-align: center; margin: 5px 0 0 0; font-size: 0.95em; color: rgba(255,255,255,0.9);">
                ⚠️ <strong>ONLY for non-contrast brain CT imaging</strong>
            </p>
        </div>
    </div>
    """
    return info_html

def create_clinical_guidelines():
    """Create enhanced clinical guidelines display with improved interactivity"""
    guidelines_html = """
    <div class="clinical-guidelines">
        <h3 class="guidelines-title">⚕️ Clinical Usage Guidelines & Safety Protocols</h3>
        
        <div style="margin: 20px 0;">
            <h4 class="critical-section">🚨 CRITICAL SAFETY REMINDERS</h4>
            <ul class="critical-list">
                <li><strong>⚠️ AI ASSISTANCE ONLY</strong> - This tool provides diagnostic support but cannot replace professional medical judgment</li>
                <li><strong>👨‍⚕️ ALWAYS CONSULT PHYSICIANS</strong> - Qualified medical professionals must make final diagnostic and treatment decisions</li>
                <li><strong>⏰ TIME-CRITICAL CASES</strong> - In suspected acute stroke, follow standard emergency protocols regardless of AI output</li>
                <li><strong>🧠 NON-CONTRAST BRAIN CT ONLY</strong> - This model was trained exclusively on non-contrast brain CT scans and will fail on other imaging types</li>
            </ul>
        </div>
        
        <div style="margin: 20px 0;">
            <h4 class="best-practices-section">✅ BEST PRACTICES</h4>
            <ul class="guidelines-list">
                <li><strong>🔍 Image Quality:</strong> Ensure CT scans have adequate contrast and clear anatomical landmarks</li>
                <li><strong>🎯 Threshold:</strong> 49% threshold optimized for balanced sensitivity/specificity</li>
                <li><strong>🔄 Cross-validation:</strong> Compare AI findings with clinical assessments</li>
                <li><strong>👥 Team Approach:</strong> Involve radiologists and neurologists in complex cases</li>
            </ul>
        </div>
        
        <div style="margin: 20px 0; padding: 15px; background: rgba(220,53,69,0.1); border-radius: 10px; border-left: 4px solid #dc3545;">
            <h4 style="color: #dc3545; margin-top: 0;">⚠️ IMAGING LIMITATIONS</h4>
            <p style="margin-bottom: 8px; font-weight: 600; color: #dc3545;">
                🚨 <strong>This model will FAIL on:</strong>
            </p>
            <ul style="margin: 0; color: #495057;">
                <li><strong>Contrast-enhanced CT scans, MRI images, X-rays</strong></li>
                <li><strong>Other organ imaging</strong> (chest, abdomen, spine, etc.)</li>
                <li><strong>Pediatric scans or post-surgical images</strong></li>
            </ul>
            <p style="margin-top: 10px; font-weight: 600; color: #dc3545;">
                ⚡ <strong>Using inappropriate image types will produce unreliable results!</strong>
            </p>
        </div>
    </div>
    """
    return guidelines_html

# Create Gradio Interface
def create_gradio_app():
    """Create the main Gradio application with improved styling"""
    
    # Custom CSS for medical-grade styling with dark mode support and animations
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        max-width: 1400px !important;
        margin: auto;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    @keyframes shimmer {
        0% { background-position: -200px 0; }
        100% { background-position: 200px 0; }
    }
    
    /* Clinical Guidelines Styling - Dark Mode Compatible */
    .clinical-guidelines {
        background: var(--background-fill-primary, #ffffff);
        border: 2px solid var(--border-color-primary, #e0e0e0);
        border-left: 5px solid #007bff;
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        color: var(--body-text-color, #000000);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .clinical-guidelines:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .clinical-guidelines::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #007bff, #28a745, #ffc107, #dc3545);
        border-radius: 15px;
        z-index: -1;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .clinical-guidelines:hover::before {
        opacity: 0.1;
    }
    
    .guidelines-title {
        color: #007bff !important;
        margin-top: 0 !important;
        font-weight: 700;
        font-size: 1.3em;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .critical-section {
        color: #dc3545 !important;
        font-weight: 700;
        margin-bottom: 10px;
        font-size: 1.1em;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .best-practices-section {
        color: #28a745 !important;
        font-weight: 700;
        margin-bottom: 10px;
        font-size: 1.1em;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .requirements-section {
        color: #fd7e14 !important;
        font-weight: 700;
        margin-bottom: 10px;
        font-size: 1.1em;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .critical-list {
        color: #dc3545 !important;
        font-weight: 600;
        line-height: 1.6;
    }
    
    .critical-list li {
        margin-bottom: 8px;
        padding-left: 5px;
        border-left: 3px solid #dc3545;
        padding-left: 10px;
        margin-left: 5px;
    }
    
    .guidelines-list {
        color: var(--body-text-color, #333333) !important;
        opacity: 0.9;
        line-height: 1.6;
    }
    
    .guidelines-list li {
        margin-bottom: 6px;
        padding-left: 5px;
        transition: all 0.2s ease;
    }
    
    .guidelines-list li:hover {
        transform: translateX(5px);
        color: #007bff !important;
    }
    
    /* Model Info Box - Enhanced with gradients and animations */
    .model-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #667eea 100%);
        background-size: 200% 200%;
        animation: gradientShift 6s ease infinite;
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 15px 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .model-info::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: shimmer 2s infinite;
    }
    
    .model-info:hover {
        transform: scale(1.02);
        transition: transform 0.3s ease;
    }
    
    .model-info h2 {
        margin-top: 0;
        text-align: center;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        font-size: 1.8em;
        margin-bottom: 20px;
    }
    
    .model-specs-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 25px;
        margin-top: 20px;
    }
    
    .model-specs-grid h3 {
        margin-bottom: 15px;
        font-size: 1.2em;
        border-bottom: 2px solid rgba(255,255,255,0.3);
        padding-bottom: 8px;
    }
    
    .model-specs-grid ul {
        list-style: none;
        padding: 0;
    }
    
    .model-specs-grid li {
        margin-bottom: 8px;
        padding: 8px;
        background: rgba(255,255,255,0.1);
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .model-specs-grid li:hover {
        background: rgba(255,255,255,0.2);
        transform: translateX(5px);
    }
    
    /* Header gradient with enhanced effects */
    .header-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #667eea 100%);
        background-size: 200% 200%;
        animation: gradientShift 8s ease infinite;
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 6px 25px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
    }
    
    .header-gradient::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 3s infinite;
    }
    
    .header-gradient h1 {
        margin: 0;
        font-size: 2.8em;
        text-shadow: 0 3px 6px rgba(0,0,0,0.3);
        animation: pulse 2s infinite;
    }
    
    /* Enhanced button styling */
    .gradio-button {
        background: linear-gradient(135deg, #007bff, #0056b3) !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 1.1em !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0,123,255,0.3) !important;
    }
    
    .gradio-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,123,255,0.4) !important;
        background: linear-gradient(135deg, #0056b3, #004085) !important;
    }
    
    /* Enhanced input styling */
    .gradio-textbox, .gradio-file {
        border-radius: 10px !important;
        border: 2px solid #e9ecef !important;
        transition: all 0.3s ease !important;
    }
    
    .gradio-textbox:focus, .gradio-file:focus {
        border-color: #007bff !important;
        box-shadow: 0 0 0 3px rgba(0,123,255,0.1) !important;
        transform: scale(1.01) !important;
    }
    
    /* Enhanced plot container */
    .plot-container {
        border-radius: 15px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .plot-container:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15) !important;
    }
    
    /* Footer styling with enhanced effects */
    .app-footer {
        text-align: center;
        margin-top: 40px;
        padding: 25px;
        background: var(--background-fill-primary, #ffffff);
        border-radius: 15px;
        border: 2px solid var(--border-color-primary, #e0e0e0);
        color: var(--body-text-color, #333333);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .app-footer::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, #007bff, #28a745, #ffc107, #dc3545);
        animation: shimmer 3s infinite;
    }
    
    .disclaimer {
        color: #dc3545 !important;
        font-weight: 700;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        animation: pulse 3s infinite;
    }
    
    /* Loading animation */
    .loading {
        position: relative;
        color: transparent;
    }
    
    .loading::after {
        content: 'Processing...';
        position: absolute;
        top: 0;
        left: 0;
        color: #007bff;
        animation: pulse 1.5s infinite;
    }
    
    /* Responsive design with enhanced breakpoints */
    @media (max-width: 1200px) {
        .gradio-container {
            max-width: 95% !important;
        }
    }
    
    @media (max-width: 768px) {
        .model-specs-grid {
            grid-template-columns: 1fr;
            gap: 15px;
        }
        
        .header-gradient h1 {
            font-size: 2.2em;
        }
        
        .clinical-guidelines {
            padding: 20px;
        }
        
        .model-info {
            padding: 20px;
        }
        
        .gradio-container {
            max-width: 100% !important;
            margin: 10px;
        }
    }
    
    @media (max-width: 480px) {
        .header-gradient h1 {
            font-size: 1.8em;
        }
        
        .clinical-guidelines {
            padding: 15px;
        }
        
        .model-info {
            padding: 15px;
        }
    }
    
    /* Dark mode enhancements */
    @media (prefers-color-scheme: dark) {
        .plot-container {
            background: #1e1e1e !important;
        }
        
        .clinical-guidelines {
            box-shadow: 0 4px 15px rgba(255,255,255,0.1);
        }
        
        .model-info {
            box-shadow: 0 6px 20px rgba(255,255,255,0.1);
        }
        
        .app-footer {
            box-shadow: 0 4px 15px rgba(255,255,255,0.1);
        }
    }
    """
    
    with gr.Blocks(css=custom_css, title="DeepStroke AI - Brain CT Analysis") as app:
        
        # Header
        gr.HTML("""
        <div class="header-gradient">
            <h1 style="margin: 0; font-size: 2.5em;">🧠 DeepStroke AI</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em;">Advanced Brain CT Stroke Detection System</p>
            <p style="margin: 5px 0 0 0; opacity: 0.9;">Powered by SE-ResNeXt50 Deep Learning Architecture</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model Information
                gr.HTML(create_model_info())
                
                # Clinical Guidelines
                gr.HTML(create_clinical_guidelines())
                
            with gr.Column(scale=2):
                # Main Interface
                gr.Markdown("## 📤 Upload Brain CT Image")
                
                # Example images section
                gr.Markdown("### 🖼️ Try Example Images")
                gr.Markdown("Click on any example below to test the stroke detection system:")
                
                with gr.Row():
                    with gr.Column():
                        # Image input
                        image_input = gr.Image(
                            label="Brain CT Scan",
                            type="pil",
                            sources=["upload", "clipboard"],
                            height=300
                        )
                        
                        # Analyze button
                        analyze_btn = gr.Button(
                            "🔍 Analyze CT Scan",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column():
                        # Results section
                        gr.Markdown("## 📊 Analysis Results")
                        
                        # Main prediction
                        prediction_output = gr.Markdown(
                            label="Prediction",
                            value="Upload an image to see results..."
                        )
                        
                        # Metrics
                        with gr.Row():
                            probability_output = gr.Textbox(
                                label="🎯 Stroke Probability",
                                interactive=False,
                                container=True
                            )
                            confidence_output = gr.Textbox(
                                label="📈 Model Confidence",
                                interactive=False,
                                container=True
                            )
                
                # Detailed Analysis
                gr.Markdown("## 📋 Detailed Clinical Analysis")
                detailed_analysis_output = gr.Markdown(
                    value="Detailed analysis will appear here after image upload..."
                )
                
                # Visualization
                gr.Markdown("## 📊 Analysis Result")
                visualization_output = gr.Plot(
                    label="Stroke Detection"
                )
                
                # Create examples component for easy clicking
                examples = gr.Examples(
                    examples=[
                        ["ExampleIMG/10189.png"],
                        ["ExampleIMG/10300.png"], 
                        ["ExampleIMG/13447.png"],
                        ["ExampleIMG/14343.png"],
                        ["ExampleIMG/15614.png"],
                        ["ExampleIMG/16760.png"],
                        ["ExampleIMG/17023.png"]
                    ],
                    inputs=[image_input],
                    outputs=[
                        prediction_output,
                        probability_output,
                        confidence_output,
                        detailed_analysis_output,
                        visualization_output
                    ],
                    fn=predict_stroke,
                    cache_examples=False,  # Disable caching to avoid index errors
                    examples_per_page=7
                )
        
        # Footer
        gr.HTML("""
        <div class="app-footer">
            <p>
                <strong>DeepStroke AI v1.0</strong> | 
                Developed for Brain CT Stroke Detection | 
                <span class="disclaimer">For Research Only</span>
            </p>
            <p style="margin: 5px 0 0 0; font-size: 0.9em;">
                Always consult with qualified medical professionals for definitive diagnosis and treatment decisions.
            </p>
        </div>
        """)
        
        # Event handlers
        analyze_btn.click(
            fn=predict_stroke,
            inputs=[image_input],
            outputs=[
                prediction_output,
                probability_output,
                confidence_output,
                detailed_analysis_output,
                visualization_output
            ]
        )
        
        # Auto-analyze on image upload
        image_input.change(
            fn=predict_stroke,
            inputs=[image_input],
            outputs=[
                prediction_output,
                probability_output,
                confidence_output,
                detailed_analysis_output,
                visualization_output
            ]
        )
    
    return app

if __name__ == "__main__":
    print("🚀 Starting DeepStroke AI Application...")
    print(f"📱 Model loaded on device: {device}")
    print(f"🎯 Using optimal threshold: {OPTIMAL_THRESHOLD}")
    
    # Create and launch the app
    app = create_gradio_app()
    
    # Launch with configuration for medical applications
    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True for public sharing (not recommended for medical apps)
        debug=False,
        auth=None,  # Add authentication for production use
        ssl_verify=True,
        favicon_path=None,
        inbrowser=True,
        show_error=True
    )
