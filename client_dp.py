import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import flwr as fl
import logging
from opacus import PrivacyEngine
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
import tenseal as ts

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the model and tokenizer
logging.info("Loading GPT-2 model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")
logging.info("Model and tokenizer loaded successfully.")

# Load client-specific data
client_data = ["Hello, how are you?", "This is a test sentence for training."]  # Replace with actual data

# Initialize TenSEAL context
logging.info("Initializing TenSEAL context...")
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40
context.generate_galois_keys()
logging.info("TenSEAL context initialized.")

# Encrypt function
def encrypt_tensor(tensor, context):
    encrypted_chunks = []
    for chunk in torch.chunk(tensor.flatten(), chunks=10):
        encrypted_chunk = ts.ckks_vector(context, chunk.numpy())
        encrypted_chunks.append(encrypted_chunk.serialize())
    return encrypted_chunks

# Decrypt function
def decrypt_tensor(encrypted_chunks, context):
    decrypted_tensor = []
    for encrypted_chunk in encrypted_chunks:
        ckks_vector = ts.ckks_vector_from(context, encrypted_chunk)
        decrypted_tensor.extend(ckks_vector.decrypt())
    return torch.tensor(decrypted_tensor).reshape(-1)

# Training loop with differential privacy
def train_with_dp(model, train_data, epochs=1, noise_multiplier=0.3, max_grad_norm=1.7, target_delta=1e-5):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=torch.utils.data.DataLoader(
            train_data,
            batch_size=1,  # Replace with a suitable batch size
            shuffle=True,
        ),
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    for epoch in range(epochs):
        total_loss = 0
        for text in train_data:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_data)}")

    epsilon = privacy_engine.get_epsilon(delta=target_delta)
    logging.info(f"Training completed with ε = {epsilon:.2f}, δ = {target_delta}.")
    return model

# Flower client class
class GPT2Client(fl.client.NumPyClient):
    def __init__(self):
        logging.info("Initializing GPT-2 Flower client...")
        self.model = model
        self.data = client_data
        logging.info("Client initialized.")

    def get_parameters(self, config=None):
        logging.info("Getting model parameters...")
        encrypted_params = []
        for _, val in self.model.state_dict().items():
            encrypted_params.extend(encrypt_tensor(val.cpu(), context))
        return encrypted_params

    def set_parameters(self, parameters):
        logging.info("Setting model parameters...")
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {}
        for k, encrypted_chunks in params_dict:
            state_dict[k] = decrypt_tensor(encrypted_chunks, context)
        self.model.load_state_dict(state_dict, strict=True)
        logging.info("Model parameters set.")

    def fit(self, parameters, config):
        logging.info("Starting training round...")
        self.set_parameters(parameters)
        self.model = train_with_dp(self.model, self.data, epochs=1)
        updated_parameters = self.get_parameters()
        logging.info("Training round completed.")
        return updated_parameters, len(self.data), {}

    def evaluate(self, parameters, config):
        logging.info("Evaluating model...")
        self.set_parameters(parameters)
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for text in self.data:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                total_loss += loss.item()
        avg_loss = total_loss / len(self.data)
        logging.info(f"Evaluation complete, average loss: {avg_loss}")
        return avg_loss, len(self.data), {}

# Start the Flower client
if __name__ == "__main__":
    logging.info("Starting Flower client...")
    fl.client.start_numpy_client(
        server_address="localhost:8081",
        client=GPT2Client()
    )
