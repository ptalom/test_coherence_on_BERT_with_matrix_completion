import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, args):
        super(TransformerModel, self).__init__()
        
        self.config = BertConfig(
            vocab_size=vocab_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size,
            hidden_size=args.hidden_size,
            position_embedding_type=args.position_embedding_type,
            hidden_dropout_prob=args.hidden_dropout_prob,
            attention_probs_dropout_prob=args.attention_probs_dropout_prob,
            output_hidden_states=args.output_hidden_states,
            use_cache=args.use_cache,
        )
        
        self._backbone = BertModel(config=self.config, add_pooling_layer=False)#, use_mlp=args.use_mlp)        
        self.read_out = nn.Linear(self.config.hidden_size, 1)

    def forward(self, X, attention_mask=None, head_mask=None):
        """
        X : (batch_size, seq_len) tokens de la matrice aplatie
        attention_mask : masque pour BERT (1 pour visible, 0 pour masqué)
        head_mask : optionnel, masque pour certaines têtes d'attention
        """
        # Récupère les embeddings de la dernière couche BERT
        embedding = self._backbone(
            input_ids=X,
            attention_mask=attention_mask,
            head_mask=head_mask
        ).last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        output = self.read_out(embedding)  # (batch_size, seq_len, 1)
        output = output.squeeze(-1)        # (batch_size, seq_len)
        
        return output
