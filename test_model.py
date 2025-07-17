import torch
import torch.nn as nn
import unittest
import numpy as np
from model import EncoderCNN, DecoderRNN, CNNtoRNN
from get_loader import Vocabulary

class TestImageCaptioningModel(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_size = 256
        self.hidden_size = 256
        self.vocab_size = 1000
        self.num_layers = 1
        self.batch_size = 4
        self.seq_len = 10
        
        # Create mock vocabulary
        self.vocab = Vocabulary(freq_threshold=1)
        # Add some basic tokens
        self.vocab.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.vocab.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        for i in range(4, self.vocab_size):
            self.vocab.itos[i] = f"word_{i}"
            self.vocab.stoi[f"word_{i}"] = i
        
    def test_encoder_cnn_initialization(self):
        """Test EncoderCNN initialization."""
        encoder = EncoderCNN(self.embed_size, train_CNN=False)
        self.assertEqual(encoder.inception.fc.out_features, self.embed_size)
        self.assertIsInstance(encoder.relu, nn.ReLU)
        self.assertIsInstance(encoder.dropout, nn.Dropout)
        
    def test_encoder_cnn_forward(self):
        """Test EncoderCNN forward pass."""
        encoder = EncoderCNN(self.embed_size, train_CNN=False)
        encoder.eval()  # Set to eval mode to avoid auxiliary outputs
        
        # Test with batch of images (batch_size, 3, 299, 299)
        images = torch.randn(self.batch_size, 3, 299, 299)
        features = encoder(images)
        
        self.assertEqual(features.shape, (self.batch_size, self.embed_size))
        self.assertFalse(torch.isnan(features).any())
        
    def test_encoder_cnn_training_mode(self):
        """Test EncoderCNN in training mode handles auxiliary outputs."""
        encoder = EncoderCNN(self.embed_size, train_CNN=False)
        encoder.train()
        
        images = torch.randn(self.batch_size, 3, 299, 299)
        features = encoder(images)
        
        # Should handle auxiliary outputs and return only main features
        self.assertEqual(features.shape, (self.batch_size, self.embed_size))
        
    def test_decoder_rnn_initialization(self):
        """Test DecoderRNN initialization."""
        decoder = DecoderRNN(self.embed_size, self.hidden_size, self.vocab_size, self.num_layers)
        
        self.assertEqual(decoder.embed.num_embeddings, self.vocab_size)
        self.assertEqual(decoder.embed.embedding_dim, self.embed_size)
        self.assertEqual(decoder.lstm.input_size, self.embed_size)
        self.assertEqual(decoder.lstm.hidden_size, self.hidden_size)
        self.assertEqual(decoder.lstm.num_layers, self.num_layers)
        self.assertEqual(decoder.linear.in_features, self.hidden_size)
        self.assertEqual(decoder.linear.out_features, self.vocab_size)
        
    def test_decoder_rnn_forward(self):
        """Test DecoderRNN forward pass."""
        decoder = DecoderRNN(self.embed_size, self.hidden_size, self.vocab_size, self.num_layers)
        
        # Mock features from encoder
        features = torch.randn(self.batch_size, self.embed_size)
        
        # Mock captions (sequence_length, batch_size)
        captions = torch.randint(0, self.vocab_size, (self.seq_len, self.batch_size))
        
        outputs = decoder(features, captions)
        
        # Output should have shape (seq_len + 1, batch_size, vocab_size)
        expected_shape = (self.seq_len + 1, self.batch_size, self.vocab_size)
        self.assertEqual(outputs.shape, expected_shape)
        self.assertFalse(torch.isnan(outputs).any())
        
    def test_cnn_to_rnn_initialization(self):
        """Test CNNtoRNN model initialization."""
        model = CNNtoRNN(self.embed_size, self.hidden_size, self.vocab_size, self.num_layers)
        
        self.assertIsInstance(model.EncoderCNN, EncoderCNN)
        self.assertIsInstance(model.DecoderRNN, DecoderRNN)
        self.assertEqual(model.EncoderCNN.train_CNN, False)
        
    def test_cnn_to_rnn_forward(self):
        """Test CNNtoRNN forward pass."""
        model = CNNtoRNN(self.embed_size, self.hidden_size, self.vocab_size, self.num_layers)
        model.eval()
        
        images = torch.randn(self.batch_size, 3, 299, 299)
        captions = torch.randint(0, self.vocab_size, (self.seq_len, self.batch_size))
        
        outputs = model(images, captions)
        
        expected_shape = (self.seq_len + 1, self.batch_size, self.vocab_size)
        self.assertEqual(outputs.shape, expected_shape)
        self.assertFalse(torch.isnan(outputs).any())
        
    def test_caption_image_functionality(self):
        """Test image captioning functionality."""
        model = CNNtoRNN(self.embed_size, self.hidden_size, self.vocab_size, self.num_layers)
        model.eval()
        
        # Single image
        image = torch.randn(1, 3, 299, 299)
        
        caption = model.caption_image(image, self.vocab, max_length=10)
        
        self.assertIsInstance(caption, list)
        self.assertTrue(len(caption) <= 10)
        
        # Check if caption contains valid words
        for word in caption:
            self.assertIn(word, self.vocab.itos.values())
            
    def test_caption_image_eos_termination(self):
        """Test that caption generation terminates at EOS token."""
        model = CNNtoRNN(self.embed_size, self.hidden_size, self.vocab_size, self.num_layers)
        model.eval()
        
        image = torch.randn(1, 3, 299, 299)
        
        # Mock the model to predict EOS token early
        original_forward = model.DecoderRNN.linear.forward
        def mock_forward(x):
            # Force EOS prediction on second step
            result = original_forward(x)
            if hasattr(mock_forward, 'call_count'):
                mock_forward.call_count += 1
            else:
                mock_forward.call_count = 1
            
            if mock_forward.call_count == 2:
                # Force EOS token (index 2)
                result[0, 0, 2] = 100.0  # High logit for EOS
                result[0, 0, :2] = -100.0  # Low logits for other tokens
                result[0, 0, 3:] = -100.0
            
            return result
        
        model.DecoderRNN.linear.forward = mock_forward
        
        caption = model.caption_image(image, self.vocab, max_length=10)
        
        # Should terminate early due to EOS
        self.assertTrue(len(caption) <= 10)
        
    def test_model_parameters_gradient(self):
        """Test that model parameters have gradients during training."""
        model = CNNtoRNN(self.embed_size, self.hidden_size, self.vocab_size, self.num_layers)
        model.train()
        
        images = torch.randn(self.batch_size, 3, 299, 299)
        captions = torch.randint(1, self.vocab_size, (self.seq_len, self.batch_size))
        
        outputs = model(images, captions)
        
        # Create a simple loss
        loss = outputs.mean()
        loss.backward()
        
        # Check that decoder parameters have gradients
        for param in model.DecoderRNN.parameters():
            self.assertIsNotNone(param.grad)
            
        # Check that encoder FC layer has gradients
        self.assertIsNotNone(model.EncoderCNN.inception.fc.weight.grad)
        self.assertIsNotNone(model.EncoderCNN.inception.fc.bias.grad)
        
    def test_model_device_compatibility(self):
        """Test model works on different devices."""
        model = CNNtoRNN(self.embed_size, self.hidden_size, self.vocab_size, self.num_layers)
        
        # Test CPU
        model = model.to("cpu")
        images = torch.randn(2, 3, 299, 299).to("cpu")
        captions = torch.randint(0, self.vocab_size, (5, 2)).to("cpu")
        
        outputs = model(images, captions)
        self.assertEqual(outputs.device.type, "cpu")
        
        # Test GPU if available
        if torch.cuda.is_available():
            model = model.to("cuda")
            images = images.to("cuda")
            captions = captions.to("cuda")
            
            outputs = model(images, captions)
            self.assertEqual(outputs.device.type, "cuda")
            
    def test_model_evaluation_mode(self):
        """Test model behavior in evaluation mode."""
        model = CNNtoRNN(self.embed_size, self.hidden_size, self.vocab_size, self.num_layers)
        
        # Test training mode
        model.train()
        self.assertTrue(model.training)
        self.assertTrue(model.EncoderCNN.training)
        self.assertTrue(model.DecoderRNN.training)
        
        # Test evaluation mode
        model.eval()
        self.assertFalse(model.training)
        self.assertFalse(model.EncoderCNN.training)
        self.assertFalse(model.DecoderRNN.training)
        
    def test_different_batch_sizes(self):
        """Test model with different batch sizes."""
        model = CNNtoRNN(self.embed_size, self.hidden_size, self.vocab_size, self.num_layers)
        model.eval()
        
        for batch_size in [1, 2, 4, 8]:
            images = torch.randn(batch_size, 3, 299, 299)
            captions = torch.randint(0, self.vocab_size, (self.seq_len, batch_size))
            
            outputs = model(images, captions)
            expected_shape = (self.seq_len + 1, batch_size, self.vocab_size)
            self.assertEqual(outputs.shape, expected_shape)
            
    def test_different_sequence_lengths(self):
        """Test model with different sequence lengths."""
        model = CNNtoRNN(self.embed_size, self.hidden_size, self.vocab_size, self.num_layers)
        model.eval()
        
        for seq_len in [5, 10, 15, 20]:
            images = torch.randn(self.batch_size, 3, 299, 299)
            captions = torch.randint(0, self.vocab_size, (seq_len, self.batch_size))
            
            outputs = model(images, captions)
            expected_shape = (seq_len + 1, self.batch_size, self.vocab_size)
            self.assertEqual(outputs.shape, expected_shape)

if __name__ == '__main__':
    unittest.main()