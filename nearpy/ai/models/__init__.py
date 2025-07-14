from .audio import AudioCNN, AudioRNN # Also add AudioTransformer
from .encode import TimeAutoEncoder, AEWrapper
# from .pinn import CardiacPressureConverter

from .cnn import CNNClassifier
from .grunet import GRUNet

MODELS = {
    'autoenc': TimeAutoEncoder,
    'audio_cnn': AudioCNN,
    'audio_rnn': AudioRNN,
    # 'audio_transformer': AudioTransformer,
    'grunet': GRUNet
}

def get_model(name): 
    if name not in MODELS.keys(): 
        print(f'Model {name} not implemented. Available models: {MODELS.keys()}')
        return None 
    
    return MODELS[name]

__all__ = [
    'get_model',
    'AEWrapper'
]