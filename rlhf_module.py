import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig
from trl.core import LengthSampler
from datasets import Dataset
import torch

class RLHFTrainer:
    def __init__(self, music_rag_system, reward_model_name='OpenAssistant/reward-model-deberta-v3-large', sft_model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.music_rag_system = music_rag_system
        self.reward_model_name = reward_model_name
        self.sft_model_name = sft_model_name
        
        print("🚀 Initializing RLHF Trainer...")
        
        # Initialize reward model and tokenizer
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name)
        self.reward_pipeline = pipeline(
            "sentiment-analysis",
            model=self.reward_model,
            tokenizer=self.reward_tokenizer,
            device=0 if torch.cuda.is_available() else -1, # Use GPU if available
            function_to_apply="none", # We want raw logits
            batch_size=16
        )
        
        # Initialize SFT model (the one to be fine-tuned)
        self.sft_tokenizer = AutoTokenizer.from_pretrained(sft_model_name)
        self.sft_model = self.music_rag_system.embedding_model # Use the existing embedding model
        
        # PPO config
        self.ppo_config = PPOConfig(
            learning_rate=1e-5,
            log_with=None, # No logging for now
            mini_batch_size=4,
            batch_size=16,
            gradient_accumulation_steps=1,
            optimize_cuda_cache=True,
            early_stopping=False,
            target_kl=0.1,
            seed=42,
            init_kl_coef=0.2,
            adap_kl_ctrl=True,
            clip_by_value=True,
            max_grad_norm=1.0,
        )
        
        self.ppo_trainer = None
        print("✅ RLHF Trainer initialized")

    def _prepare_dataset(self, tracks_df):
        # Create a dataset from track descriptions for fine-tuning
        # This dataset will be used to generate responses that are then ranked by the reward model
        data = [{
            "query": f"Recommend music similar to {row['track_name']} by {row['artist_name']}",
            "response": row['description']
        } for _, row in tracks_df.iterrows()]
        
        # Convert to Hugging Face Dataset format
        dataset = Dataset.from_list(data)
        
        # Tokenize queries and responses
        def tokenize_function(examples):
            return {
                "input_ids": self.sft_tokenizer(examples["query"], truncation=True).input_ids,
                "attention_mask": self.sft_tokenizer(examples["query"], truncation=True).attention_mask,
                "response_input_ids": self.sft_tokenizer(examples["response"], truncation=True).input_ids,
                "response_attention_mask": self.sft_tokenizer(examples["response"], truncation=True).attention_mask,
            }

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset.set_format(type="torch")
        
        return tokenized_dataset

    def train_rlhf(self, tracks_df, num_epochs=1):
        print("🏋️‍♂️ Starting RLHF training...")
        
        # Prepare dataset for PPO training
        ppo_dataset = self._prepare_dataset(tracks_df)
        
        # Initialize PPO Trainer
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.sft_model,
            ref_model=None, # No reference model for now
            tokenizer=self.sft_tokenizer,
            dataset=ppo_dataset,
        )
        
        # Define generation settings
        gen_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.sft_tokenizer.eos_token_id,
            "max_new_tokens": 64,
        }
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            for batch in self.ppo_trainer.dataloader:
                query_tensors = batch["input_ids"]
                
                # Generate responses
                response_tensors = self.ppo_trainer.generate(query_tensors, **gen_kwargs)
                
                # Decode responses for reward model
                responses = [self.sft_tokenizer.decode(r.squeeze()) for r in response_tensors]
                
                # Get rewards from reward model
                # The reward model expects pairs of (query, response)
                # For simplicity, we'll just evaluate the response based on its content
                # A more sophisticated approach would involve comparing generated response to ideal response
                rewards = [torch.tensor(self.reward_pipeline(r)[0]['score']) for r in responses]
                
                # Train PPO
                stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
                self.ppo_trainer.log_stats(stats, batch, rewards)
                
        print("✅ RLHF training complete!")
        
        # Update the RAG system's embedding model with the fine-tuned model
        self.music_rag_system.embedding_model = self.sft_model
        print("✅ RAG system's embedding model updated with fine-tuned model.")

    def save_fine_tuned_model(self, path="./fine_tuned_sft_model"):
        print(f"💾 Saving fine-tuned SFT model to {path}...")
        self.sft_model.save_pretrained(path)
        self.sft_tokenizer.save_pretrained(path)
        print("✅ Fine-tuned SFT model saved.")

    def load_fine_tuned_model(self, path="./fine_tuned_sft_model"):
        print(f"📂 Loading fine-tuned SFT model from {path}...")
        self.sft_model = AutoModelForSequenceClassification.from_pretrained(path)
        self.sft_tokenizer = AutoTokenizer.from_pretrained(path)
        self.music_rag_system.embedding_model = self.sft_model
        print("✅ Fine-tuned SFT model loaded and RAG system updated.")


