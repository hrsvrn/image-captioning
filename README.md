# Image Captioning with CNN-RNN

An image captioning system that generates natural language descriptions for images using a CNN-RNN architecture. The model combines a CNN encoder (Inception v3) for feature extraction and an LSTM decoder for caption generation.

## Architecture

The system consists of two main components:

1. **CNN Encoder**: Uses a pre-trained Inception v3 model to extract image features
2. **RNN Decoder**: Uses an LSTM network to generate sequential captions from the extracted features

## Dataset

The project uses the Flickr8k dataset, which contains:
- 8,091 images
- 40,455 captions (5 captions per image)
- Images with diverse scenes and objects

## Project Structure

```
image-captioning/
├── model.py              # Model definitions (EncoderCNN, DecoderRNN, CNNtoRNN)
├── train.py              # Training script
├── get_loader.py         # Data loading utilities
├── utils.py              # Utility functions (checkpointing, evaluation)
├── dataset.py            # Dataset download script
├── flickr8k/
│   ├── images/           # Dataset images
│   └── captions.txt      # Image-caption pairs
├── test_examples/        # Test images for evaluation
└── runs/                 # TensorBoard logs
```

## Model Components

### EncoderCNN
- Pre-trained Inception v3 backbone
- Custom fully connected layer for embedding
- Dropout for regularization
- Option to fine-tune CNN parameters

### DecoderRNN
- LSTM-based language model
- Word embedding layer
- Linear layer for vocabulary prediction
- Dropout for regularization

### CNNtoRNN
- Complete model combining encoder and decoder
- Inference method for generating captions
- Handles end-to-end training

## Installation

Install required dependencies:

```bash
pip install torch torchvision
pip install pandas pillow spacy
pip install tensorboard kagglehub
python -m spacy download en_core_web_sm
```

## Usage

### 1. Download Dataset

```bash
python dataset.py
```

### 2. Train Model

```bash
python train.py
```

### 3. Generate Captions

The model can generate captions for new images using the `caption_image` method:

```python
from model import CNNtoRNN
import torch

# Load trained model
model = CNNtoRNN(embed_size=256, hidden_size=256, vocab_size=vocab_size, num_layers=1)
model.load_state_dict(torch.load('checkpoint.pth'))

# Generate caption
caption = model.caption_image(image_tensor, vocabulary)
```

## Configuration

### Hyperparameters
- **Embedding size**: 256
- **Hidden size**: 256
- **Number of LSTM layers**: 1
- **Learning rate**: 3e-4
- **Batch size**: 32
- **Number of epochs**: 100

### Data Preprocessing
- Image resize: 356x356 → 299x299 (random crop)
- Normalization: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
- Vocabulary threshold: 5 (minimum word frequency)

## Features

- **Vocabulary Management**: Automatic vocabulary building with special tokens (PAD, SOS, EOS, UNK)
- **Data Loading**: Efficient batch processing with padding
- **Checkpointing**: Model state saving and loading
- **Evaluation**: Sample caption generation on test images
- **Monitoring**: TensorBoard integration for training visualization

## Training Process

1. **Feature Extraction**: CNN encoder extracts visual features from images
2. **Caption Processing**: Text captions are tokenized and converted to numerical sequences
3. **Training**: Model learns to predict next word given image features and previous words
4. **Evaluation**: Periodic testing on sample images during training

## Inference

During inference, the model:
1. Extracts features from the input image
2. Generates captions word by word using beam search or greedy decoding
3. Stops when EOS token is generated or maximum length is reached

## File Descriptions

- `model.py`: Core neural network architectures
- `train.py`: Training loop with loss calculation and optimization
- `get_loader.py`: Dataset class and data loading utilities
- `utils.py`: Helper functions for checkpointing and evaluation
- `dataset.py`: Script to download Flickr8k dataset

## Example Output

The model generates captions like:
- "A child in a pink dress is climbing up a set of stairs"
- "A black dog and a spotted dog are fighting"
- "Two dogs of different breeds looking at each other on the road"

## Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision
- pandas
- Pillow
- spacy
- tensorboard
- kagglehub

## License

This project is for educational purposes. Please respect the Flickr8k dataset license terms.