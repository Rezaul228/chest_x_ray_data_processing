#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class ImageEncoder(Model):
    def __init__(self, embed_dim=256):
        super(ImageEncoder, self).__init__()
        
        # CNN backbone
        self.conv_blocks = [
            self._make_conv_block(32),
            self._make_conv_block(64),
            self._make_conv_block(128)
        ]
        
        self.patch_proj = layers.Dense(embed_dim)
        self.patch_norm = layers.LayerNormalization()
        
    def _make_conv_block(self, filters):
        return [
            layers.Conv2D(filters, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2)
        ]
        
    def call(self, inputs, training=False, verbose=False):
        x = inputs
        if verbose:
            print(f"\nImage Encoder Input shape: {x.shape}")
        
        for i, block in enumerate(self.conv_blocks, 1):
            for layer in block:
                x = layer(x, training=training)
            if verbose:
                print(f"After Conv Block {i}: {x.shape}")
        
        batch_size = tf.shape(x)[0]
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        patches = tf.reshape(x, [batch_size, h*w, -1])
        
        patches = self.patch_proj(patches)
        patches = self.patch_norm(patches)
        patches = tf.nn.l2_normalize(patches, axis=-1)
        
        if verbose:
            print(f"Final patch embeddings: {patches.shape}")
        return patches

class TextEncoder(Model):
    def __init__(self, vocab_size, max_length=30, embed_dim=256):
        super(TextEncoder, self).__init__()
        
        self.embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=True
        )
        
        self.lstm1 = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, dropout=0.5)
        )
        self.lstm2 = layers.Bidirectional(
            layers.LSTM(embed_dim // 2, return_sequences=True, dropout=0.5)
        )
        
        self.token_proj = layers.Dense(embed_dim)
        self.token_norm = layers.LayerNormalization()
        
    def call(self, inputs, training=False, verbose=False):
        if verbose:
            print(f"\nText Encoder Input shape: {inputs.shape}")
        
        x = self.embedding(inputs)
        x = self.lstm1(x, training=training)
        x = self.lstm2(x, training=training)
        
        x = self.token_proj(x)
        x = self.token_norm(x)
        x = tf.nn.l2_normalize(x, axis=-1)
        
        if verbose:
            print(f"Final token embeddings: {x.shape}")
        return x

class HierarchicalCoAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Gate weights
        self.local_image_gate_weights = self.add_weight(
            shape=(embed_dim,),
            initializer="glorot_uniform",
            trainable=True,
            name="local_image_gate_weights"
        )
        self.local_text_gate_weights = self.add_weight(
            shape=(embed_dim,),
            initializer="glorot_uniform",
            trainable=True,
            name="local_text_gate_weights"
        )
        self.global_image_gate_weights = self.add_weight(
            shape=(embed_dim,),
            initializer="glorot_uniform",
            trainable=True,
            name="global_image_gate_weights"
        )
        self.global_text_gate_weights = self.add_weight(
            shape=(embed_dim,),
            initializer="glorot_uniform",
            trainable=True,
            name="global_text_gate_weights"
        )
        
        # Local attention layers
        self.cross_attention1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim//num_heads)
        self.cross_attention2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim//num_heads)
        
        # Global attention layers
        self.global_cross_attention1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim//num_heads)
        self.global_cross_attention2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim//num_heads)
        
        # Layer norms
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.norm3 = layers.LayerNormalization()
        self.norm4 = layers.LayerNormalization()
        
        self.global_norm1 = layers.LayerNormalization()
        self.global_norm2 = layers.LayerNormalization()
        self.global_norm3 = layers.LayerNormalization()
        self.global_norm4 = layers.LayerNormalization()
        
        # FFN layers
        self.ffn1 = tf.keras.Sequential([
            layers.Dense(embed_dim * 4, activation='relu'),
            layers.Dense(embed_dim)
        ])
        self.ffn2 = tf.keras.Sequential([
            layers.Dense(embed_dim * 4, activation='relu'),
            layers.Dense(embed_dim)
        ])
        
        self.global_ffn1 = tf.keras.Sequential([
            layers.Dense(embed_dim * 4, activation='relu'),
            layers.Dense(embed_dim)
        ])
        self.global_ffn2 = tf.keras.Sequential([
            layers.Dense(embed_dim * 4, activation='relu'),
            layers.Dense(embed_dim)
        ])
    
    def call(self, image_tokens, text_tokens):
        # Local attention: image → text
        attended_image = self.cross_attention1(
            query=image_tokens,
            key=text_tokens,
            value=text_tokens
        )
        
        local_image_gate = tf.nn.sigmoid(self.local_image_gate_weights)
        local_image_gate = tf.reshape(local_image_gate, [1, 1, self.embed_dim])
        gated_image = local_image_gate * attended_image + (1 - local_image_gate) * image_tokens
        
        image_tokens = self.norm1(image_tokens + gated_image)
        image_tokens = self.norm2(image_tokens + self.ffn1(image_tokens))
        
        # Local attention: text → image
        attended_text = self.cross_attention2(
            query=text_tokens,
            key=image_tokens,
            value=image_tokens
        )
        
        local_text_gate = tf.nn.sigmoid(self.local_text_gate_weights)
        local_text_gate = tf.reshape(local_text_gate, [1, 1, self.embed_dim])
        gated_text = local_text_gate * attended_text + (1 - local_text_gate) * text_tokens
        
        text_tokens = self.norm3(text_tokens + gated_text)
        text_tokens = self.norm4(text_tokens + self.ffn2(text_tokens))
        
        # Global attention
        global_image_token = tf.reduce_mean(image_tokens, axis=1, keepdims=True)
        global_text_token = tf.reduce_mean(text_tokens, axis=1, keepdims=True)
        
        # Global image → text
        attended_global_image = self.global_cross_attention1(
            query=global_image_token,
            key=global_text_token,
            value=global_text_token
        )
        
        global_image_gate = tf.nn.sigmoid(self.global_image_gate_weights)
        global_image_gate = tf.reshape(global_image_gate, [1, 1, self.embed_dim])
        gated_global_image = global_image_gate * attended_global_image + (1 - global_image_gate) * global_image_token
        
        global_image_token = self.global_norm1(global_image_token + gated_global_image)
        global_image_token = self.global_norm2(global_image_token + self.global_ffn1(global_image_token))
        
        # Global text → image
        attended_global_text = self.global_cross_attention2(
            query=global_text_token,
            key=global_image_token,
            value=global_image_token
        )
        
        global_text_gate = tf.nn.sigmoid(self.global_text_gate_weights)
        global_text_gate = tf.reshape(global_text_gate, [1, 1, self.embed_dim])
        gated_global_text = global_text_gate * attended_global_text + (1 - global_text_gate) * global_text_token
        
        global_text_token = self.global_norm3(global_text_token + gated_global_text)
        global_text_token = self.global_norm4(global_text_token + self.global_ffn2(global_text_token))
        
        # Combine global and local representations
        seq_len_image = tf.shape(image_tokens)[1]
        seq_len_text = tf.shape(text_tokens)[1]
        
        tiled_global_image = tf.tile(global_image_token, [1, seq_len_image, 1])
        tiled_global_text = tf.tile(global_text_token, [1, seq_len_text, 1])
        
        combined_image_tokens = image_tokens + tiled_global_image
        combined_text_tokens = text_tokens + tiled_global_text
        
        return combined_image_tokens, combined_text_tokens

class MultimodalFusion(Model):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim)
        
        # Co-attention layers for synergy and difference branches
        self.synergy_co_attention_layers = [
            HierarchicalCoAttention(embed_dim, num_heads) 
            for _ in range(num_layers)
        ]
        
        self.difference_co_attention_layers = [
            HierarchicalCoAttention(embed_dim, num_heads) 
            for _ in range(num_layers)
        ]
        
        # Final embedding layers
        self.synergy_image_embedding = tf.keras.Sequential([
            layers.Dense(embed_dim),
            layers.LayerNormalization()
        ])
        
        self.synergy_text_embedding = tf.keras.Sequential([
            layers.Dense(embed_dim),
            layers.LayerNormalization()
        ])
        
        self.difference_image_embedding = tf.keras.Sequential([
            layers.Dense(embed_dim),
            layers.LayerNormalization()
        ])
        
        self.difference_text_embedding = tf.keras.Sequential([
            layers.Dense(embed_dim),
            layers.LayerNormalization()
        ])
        
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
    
    def _process_tokens(self, image_tokens, text_tokens, co_attention_layers, image_embedding_layer, text_embedding_layer):
        # Apply co-attention layers
        for co_attn_layer in co_attention_layers:
            image_tokens, text_tokens = co_attn_layer(image_tokens, text_tokens)
        
        # Global pooling
        image_emb = tf.reduce_mean(image_tokens, axis=1)
        text_emb = tf.reduce_mean(text_tokens, axis=1)
        
        # Final embeddings
        image_emb = image_embedding_layer(image_emb)
        text_emb = text_embedding_layer(text_emb)
        
        # L2 normalization
        image_emb = tf.nn.l2_normalize(image_emb, axis=-1)
        text_emb = tf.nn.l2_normalize(text_emb, axis=-1)
        
        return image_emb, text_emb
    
    def call(self, inputs, training=False):
        images, texts = inputs
        
        # Get tokens from encoders
        image_tokens = self.image_encoder(images, training=training)
        text_tokens = self.text_encoder(texts, training=training)
        
        # Process synergy branch
        synergy_image_emb, synergy_text_emb = self._process_tokens(
            image_tokens, text_tokens,
            self.synergy_co_attention_layers,
            self.synergy_image_embedding,
            self.synergy_text_embedding
        )
        
        # Process difference branch
        difference_image_emb, difference_text_emb = self._process_tokens(
            image_tokens, text_tokens,
            self.difference_co_attention_layers,
            self.difference_image_embedding,
            self.difference_text_embedding
        )
        
        # Add loss if in training mode
        if training:
            batch_size = tf.shape(synergy_image_emb)[0]
            dummy_labels = tf.zeros((batch_size,))
            
            synergy_loss = self.contrastive_loss(dummy_labels, (synergy_image_emb, synergy_text_emb))
            difference_loss = self.contrastive_loss(dummy_labels, (difference_image_emb, difference_text_emb))
            
            self.add_loss(synergy_loss + difference_loss)
        
        return synergy_image_emb, synergy_text_emb


class ContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def call(self, y_true, embeddings):
        image_emb, text_emb = embeddings
        batch_size = tf.shape(image_emb)[0]
        
        # Compute similarity matrix
        similarity_matrix = tf.matmul(image_emb, text_emb, transpose_b=True)
        similarity_matrix /= self.temperature
        
        # Labels for matching pairs (diagonal matrix)
        labels = tf.eye(batch_size, dtype=tf.float32)
        
        # Image-to-text direction
        exp_sim = tf.exp(similarity_matrix)
        log_prob = similarity_matrix - tf.math.log(tf.reduce_sum(exp_sim, axis=1, keepdims=True))
        mean_log_prob_pos = tf.reduce_sum(labels * log_prob, axis=1) / tf.reduce_sum(labels, axis=1)
        loss_i2t = -mean_log_prob_pos
        
        # Text-to-image direction
        log_prob = tf.transpose(similarity_matrix) - tf.math.log(tf.reduce_sum(exp_sim, axis=0, keepdims=True))
        mean_log_prob_pos = tf.reduce_sum(labels * log_prob, axis=1) / tf.reduce_sum(labels, axis=1)
        loss_t2i = -mean_log_prob_pos
        
        # Combined bidirectional loss
        loss = (loss_i2t + loss_t2i) / 2.0
        return tf.reduce_mean(loss)

class VisualizationHelper:
    def __init__(self, save_dir='visualizations'):
        self.save_dir = save_dir
        self.history = {'loss': [], 'recall@1': [], 'recall@5': [], 'recall@10': []}
        self.epochs = []
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
    
    def update_history(self, epoch_metrics):
        for metric, value in epoch_metrics.items():
            if metric in self.history:
                self.history[metric].append(value)
    
    def plot_training_progress(self):
        plt.figure(figsize=(18, 6))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'], 'b-', label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Loss')
        
        # Recall plot
        plt.subplot(1, 2, 2)
        for k in [1, 5, 10]:
            key = f'recall@{k}'
            if key in self.history and self.history[key]:
                plt.plot(self.history[key], label=f'Recall@{k}')
        
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        plt.title('Validation Recall')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training_progress.png')
        plt.close()
    
    def plot_similarity_matrix(self, similarity_matrix, k=5):
        plt.figure(figsize=(10, 8))
        
        # Plot heatmap of similarity matrix
        sns.heatmap(similarity_matrix, cmap='viridis')
        plt.title(f'Similarity Matrix (Top {k} shown)')
        plt.xlabel('Text Index')
        plt.ylabel('Image Index')
        
        # Highlight diagonal (matching pairs)
        for i in range(min(len(similarity_matrix), 10)):
            plt.plot(i, i, 'ro')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/similarity_matrix.png')
        plt.close()
    
    def visualize_retrieval_examples(self, model, val_data, num_img_queries=3, num_text_queries=2, top_k=3):
        # Get model embeddings
        image_emb, text_emb = model((val_data['images'], val_data['captions']), training=False)
        
        # Compute similarity matrix
        similarity_matrix = tf.matmul(image_emb, text_emb, transpose_b=True).numpy()
        
        # Save timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create figure for image-to-text retrieval
        self._visualize_image_to_text_retrieval(
            val_data, similarity_matrix, num_img_queries, top_k, timestamp
        )
        
        # Create figure for text-to-image retrieval
        self._visualize_text_to_image_retrieval(
            val_data, similarity_matrix, num_text_queries, top_k, timestamp
        )
        
        # Plot the similarity matrix
        self.plot_similarity_matrix(similarity_matrix, k=top_k)
    
    def _visualize_image_to_text_retrieval(self, val_data, similarity_matrix, num_queries, top_k, timestamp):
        # Select random query images
        query_indices = np.random.choice(len(val_data['images']), num_queries, replace=False)
        
        for idx, query_idx in enumerate(query_indices):
            plt.figure(figsize=(12, 4 * (top_k + 1)))
            
            # Display query image
            plt.subplot(top_k + 1, 3, 2)
            plt.imshow(val_data['images'][query_idx])
            plt.title(f"Query Image {query_idx}")
            plt.axis('off')
            
            # Get top-k text matches
            sim_scores = similarity_matrix[query_idx]
            top_indices = np.argsort(sim_scores)[::-1][:top_k]
            
            # Check correct match rank
            correct_rank = np.where(np.argsort(sim_scores)[::-1] == query_idx)[0][0] + 1
            
            # Display top text matches
            for i, match_idx in enumerate(top_indices):
                is_correct = match_idx == query_idx
                
                plt.subplot(top_k + 1, 1, i + 2)
                plt.text(0.1, 0.5, 
                       f"Match {i+1}: Text #{match_idx} (Score: {sim_scores[match_idx]:.4f})" + 
                       (" - CORRECT MATCH!" if is_correct else ""),
                       fontsize=12, color='green' if is_correct else 'black')
                
                # Tokenizer should be used here to decode captions, but we'll use a placeholder
                caption = f"Caption for text #{match_idx}"
                plt.text(0.1, 0.3, caption, fontsize=10)
                plt.axis('off')
            
            plt.suptitle(f"Image-to-Text Retrieval (Correct match at rank {correct_rank})")
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f'{self.save_dir}/img2text_query{idx}_{timestamp}.png')
            plt.close()
    
    def _visualize_text_to_image_retrieval(self, val_data, similarity_matrix, num_queries, top_k, timestamp):
        # Select random query texts
        query_indices = np.random.choice(len(val_data['captions']), num_queries, replace=False)
        
        for idx, query_idx in enumerate(query_indices):
            plt.figure(figsize=(12, 6))
            
            # Display query text info
            plt.subplot(1, top_k + 1, 1)
            
            # Tokenizer should be used here to decode captions, but we'll use a placeholder
            caption = f"Caption for text #{query_idx}"
            plt.text(0.1, 0.5, f"Query Text: {caption}", fontsize=12)
            plt.text(0.1, 0.3, f"Text #{query_idx}", fontsize=10)
            plt.axis('off')
            
            # Get top-k image matches
            sim_scores = similarity_matrix.T[query_idx]  # Transpose for text-to-image
            top_indices = np.argsort(sim_scores)[::-1][:top_k]
            
            # Check correct match rank
            correct_rank = np.where(np.argsort(sim_scores)[::-1] == query_idx)[0][0] + 1
            
            # Display top image matches
            for i, match_idx in enumerate(top_indices):
                is_correct = match_idx == query_idx
                
                plt.subplot(1, top_k + 1, i + 2)
                plt.imshow(val_data['images'][match_idx])
                plt.title(f"Match {i+1}: Image #{match_idx}\nScore: {sim_scores[match_idx]:.4f}" + 
                        ("\nCORRECT!" if is_correct else ""),
                        color='green' if is_correct else 'black')
                plt.axis('off')
            
            plt.suptitle(f"Text-to-Image Retrieval (Correct match at rank {correct_rank})")
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f'{self.save_dir}/text2img_query{idx}_{timestamp}.png')
            plt.close()
    
    def log_epoch_results(self, epoch, metrics):
        # Format metrics for display
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log to console
        print(f"[{timestamp}] Epoch {epoch}: {metrics_str}")
        
        # Log to file
        with open(f'{self.save_dir}/training_log.txt', 'a') as f:
            f.write(f"[{timestamp}] Epoch {epoch}: {metrics_str}\n")

