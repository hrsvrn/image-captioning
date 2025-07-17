import torch
import unittest
import tempfile
import os
import pandas as pd
from PIL import Image
import numpy as np
from get_loader import get_loader, FlickrDataset
from model import CNNtoRNN
from utils import save_checkpoint, load_checkpoint
import torchvision.transforms as transforms

class TestImageCaptioningIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with mock data."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(self.images_dir)
        
        # Create mock images
        self.mock_images = []
        for i in range(5):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img_path = os.path.join(self.images_dir, f"test_image_{i}.jpg")
            img.save(img_path)
            self.mock_images.append(f"test_image_{i}.jpg")
        
        # Create mock captions file
        self.captions_data = {
            "image": self.mock_images * 2,  # Each image has 2 captions
            "caption": [
                "A dog running in the park",
                "A happy dog playing outside",
                "A cat sitting on a chair",
                "A white cat relaxing indoors",
                "A bird flying in the sky",
                "A small bird with colorful feathers",
                "A car driving on the road",
                "A red car parked on the street",
                "A house with a garden",
                "A beautiful house with flowers"
            ]
        }
        
        self.captions_file = os.path.join(self.temp_dir, "captions.txt")
        df = pd.DataFrame(self.captions_data)
        df.to_csv(self.captions_file, index=False)
        
        # Model hyperparameters
        self.embed_size = 128
        self.hidden_size = 128
        self.num_layers = 1
        
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_data_loading(self):
        """Test that data loading works correctly."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        loader, dataset = get_loader(
            root_folder=self.images_dir,
            annotation_file=self.captions_file,
            transform=transform,
            batch_size=2,
            num_workers=0  # Use 0 workers for testing
        )
        
        # Test dataset properties
        self.assertEqual(len(dataset), 10)  # 5 images * 2 captions each
        self.assertGreater(len(dataset.vocab), 4)  # At least special tokens
        
        # Test data loader
        for batch_idx, (images, captions) in enumerate(loader):
            self.assertEqual(images.shape[0], 2)  # batch size
            self.assertEqual(images.shape[1], 3)  # RGB channels
            self.assertEqual(images.shape[2], 224)  # height
            self.assertEqual(images.shape[3], 224)  # width
            
            # Captions should be padded sequences
            self.assertEqual(captions.shape[1], 2)  # batch size
            self.assertGreater(captions.shape[0], 0)  # sequence length
            
            if batch_idx == 0:  # Only test first batch
                break
                
    def test_model_training_step(self):
        """Test a complete training step."""
        transform = transforms.Compose([
            transforms.Resize((299, 299)),  # Inception v3 input size
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        loader, dataset = get_loader(
            root_folder=self.images_dir,
            annotation_file=self.captions_file,
            transform=transform,
            batch_size=2,
            num_workers=0
        )
        
        vocab_size = len(dataset.vocab)
        model = CNNtoRNN(self.embed_size, self.hidden_size, vocab_size, self.num_layers)
        model = model.to(self.device)
        
        criterion = torch.nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        
        for batch_idx, (images, captions) in enumerate(loader):
            images = images.to(self.device)
            captions = captions.to(self.device)
            
            # Forward pass
            outputs = model(images, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Verify loss is finite
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(loss.item(), 0)
            
            break  # Only test one batch
            
    def test_model_inference(self):
        """Test model inference/caption generation."""
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        loader, dataset = get_loader(
            root_folder=self.images_dir,
            annotation_file=self.captions_file,
            transform=transform,
            batch_size=1,
            num_workers=0
        )
        
        vocab_size = len(dataset.vocab)
        model = CNNtoRNN(self.embed_size, self.hidden_size, vocab_size, self.num_layers)
        model = model.to(self.device)
        model.eval()
        
        # Get a single image
        for images, _ in loader:
            images = images.to(self.device)
            
            # Generate caption
            caption = model.caption_image(images, dataset.vocab, max_length=10)
            
            # Verify caption properties
            self.assertIsInstance(caption, list)
            self.assertTrue(len(caption) <= 10)
            
            # Check that all words are in vocabulary
            for word in caption:
                self.assertIn(word, dataset.vocab.itos.values())
                
            break
            
    def test_checkpoint_save_load(self):
        """Test saving and loading model checkpoints."""
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        loader, dataset = get_loader(
            root_folder=self.images_dir,
            annotation_file=self.captions_file,
            transform=transform,
            batch_size=1,
            num_workers=0
        )
        
        vocab_size = len(dataset.vocab)
        model = CNNtoRNN(self.embed_size, self.hidden_size, vocab_size, self.num_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create checkpoint
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": 100,
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.temp_dir, "test_checkpoint.pth.tar")
        save_checkpoint(checkpoint, checkpoint_path)
        
        # Verify file exists
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Create new model and optimizer
        new_model = CNNtoRNN(self.embed_size, self.hidden_size, vocab_size, self.num_layers)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        
        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        step = load_checkpoint(loaded_checkpoint, new_model, new_optimizer)
        
        # Verify step was loaded correctly
        self.assertEqual(step, 100)
        
        # Verify model parameters are the same
        for (name1, param1), (name2, param2) in zip(model.named_parameters(), new_model.named_parameters()):
            self.assertEqual(name1, name2)
            self.assertTrue(torch.equal(param1, param2))
            
    def test_model_with_different_vocab_sizes(self):
        """Test model with different vocabulary sizes."""
        # Create datasets with different frequency thresholds
        for freq_threshold in [1, 2, 3]:
            transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            dataset = FlickrDataset(
                root_dir=self.images_dir,
                captions_file=self.captions_file,
                transform=transform,
                freq_threshold=freq_threshold
            )
            
            vocab_size = len(dataset.vocab)
            model = CNNtoRNN(self.embed_size, self.hidden_size, vocab_size, self.num_layers)
            
            # Test forward pass
            test_image = torch.randn(1, 3, 299, 299)
            test_caption = torch.randint(0, vocab_size, (5, 1))
            
            outputs = model(test_image, test_caption)
            expected_shape = (6, 1, vocab_size)  # seq_len + 1, batch_size, vocab_size
            self.assertEqual(outputs.shape, expected_shape)
            
    def test_model_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        loader, dataset = get_loader(
            root_folder=self.images_dir,
            annotation_file=self.captions_file,
            transform=transform,
            batch_size=1,
            num_workers=0
        )
        
        vocab_size = len(dataset.vocab)
        model = CNNtoRNN(self.embed_size, self.hidden_size, vocab_size, self.num_layers)
        model.train()
        
        # Get batch
        for images, captions in loader:
            # Forward pass
            outputs = model(images, captions[:-1])
            
            # Create loss
            loss = outputs.mean()
            
            # Backward pass
            loss.backward()
            
            # Check gradients exist
            encoder_fc_grad = model.EncoderCNN.inception.fc.weight.grad
            decoder_embed_grad = model.DecoderRNN.embed.weight.grad
            decoder_lstm_grad = model.DecoderRNN.lstm.weight_ih_l0.grad
            decoder_linear_grad = model.DecoderRNN.linear.weight.grad
            
            self.assertIsNotNone(encoder_fc_grad)
            self.assertIsNotNone(decoder_embed_grad)
            self.assertIsNotNone(decoder_lstm_grad)
            self.assertIsNotNone(decoder_linear_grad)
            
            # Check gradients are not zero
            self.assertNotEqual(encoder_fc_grad.abs().sum().item(), 0)
            self.assertNotEqual(decoder_embed_grad.abs().sum().item(), 0)
            self.assertNotEqual(decoder_lstm_grad.abs().sum().item(), 0)
            self.assertNotEqual(decoder_linear_grad.abs().sum().item(), 0)
            
            break

if __name__ == '__main__':
    unittest.main()