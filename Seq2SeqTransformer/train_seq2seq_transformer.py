import sys
sys.path.append('./src/')
sys.path.append('./src/data/')
sys.path.append('./src/models/')

import torch
import torch.nn as nn
import yaml
import metrics
from models import trainer
from data.datamodule import DataManager
from txt_logger import TXTLogger
from seq2seq_transformer import Seq2SeqTransformer 
from bpe_tokenizer import BPETokenizer


EMB_SIZE = 256
NHEAD = 8
FFN_HID_DIM = 256
BATCH_SIZE = 1024
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
RANDOM_STATE = 42
MAX_LEN = 15
LEARNING_RATE = 0.001
SHEDULER_STEP_SIZE = 10
DROPOUT = 0.1
TRY_ONE_BATCH = False
PREFIX_FILTER = None
EPOCH_NUM = 20
FILENAME = '../data/rus.txt'
TRAIN_SIZE = 0.8


torch.manual_seed(RANDOM_STATE)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
    
def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
    

class Seq2SeqTrainer(nn.Module):
    def __init__(self, model, optimizer, scheduler, loss_fn, src_tokenizer, tgt_tokenizer):
        super(Seq2SeqTrainer, self).__init__()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.tgt_tokenizer = tgt_tokenizer
        self.src_tokenizer = src_tokenizer
        
        
    def training_step(self, batch):
        self.optimizer.zero_grad()
        src, tgt = batch
        src = src.transpose(1, 0)
        tgt = tgt.transpose(1, 0)
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        logits = self.model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[1:, :]
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        return loss
        
    def validation_step(self, batch):
        with torch.no_grad():
            src, tgt = batch
            src = src.transpose(1, 0)
            tgt = tgt.transpose(1, 0)
            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            logits = self.model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            tgt_out = tgt[1:, :]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        
        return loss

    def forward(self, src):
        batch_size = src.shape[0]
        src = src.transpose(1, 0)
        tgt_input = torch.tensor([[BOS_IDX] * batch_size], dtype=torch.long, device=DEVICE).view(1, batch_size)
        for i in range(MAX_LEN):
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            logits = self.model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            _, next_items = logits[-1].topk(1)
            next_items = next_items.squeeze().view(1, batch_size)
            tgt_input = torch.cat((tgt_input, next_items), dim=0)
        return tgt_input, None
            
    def eval_bleu(self, predicted_ids_list, target_tensor):
        predicted = predicted_ids_list.squeeze().detach().cpu().numpy().swapaxes(0, 1)[:, 1:]
        actuals = target_tensor.squeeze().detach().cpu().numpy()[:, 1:]
        bleu_score, actual_sentences, predicted_sentences = metrics.bleu_scorer(
            predicted=predicted, actual=actuals, target_tokenizer=self.tgt_tokenizer
        )
        return bleu_score, actual_sentences, predicted_sentences
        
if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'

    config = dict()
    config['batch_size'] = BATCH_SIZE
    config['prefix_filter'] = PREFIX_FILTER
    config['max_length'] = MAX_LEN
    config['epoch_num'] = EPOCH_NUM
    config['try_one_batch'] = TRY_ONE_BATCH
    config['learning_rate'] = LEARNING_RATE
    config['device'] = DEVICE
    config['filename'] = FILENAME
    config['train_size'] = TRAIN_SIZE
    config['embedding_size'] = EMB_SIZE
    config['hidden_size'] = FFN_HID_DIM

    dm = DataManager(config)
    train_dataloader, dev_dataloader = dm.prepare_data()

    SRC_VOCAB_SIZE = dm.source_tokenizer.tokenizer.get_vocab_size()
    TGT_VOCAB_SIZE = dm.target_tokenizer.tokenizer.get_vocab_size()
    PAD_IDX = dm.source_tokenizer.tokenizer.token_to_id("[PAD]")
    BOS_IDX = dm.source_tokenizer.tokenizer.token_to_id("[BOS]")
    EOS_IDX = dm.source_tokenizer.tokenizer.token_to_id("[EOS]")
    print(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)
    
    config['src_vocab_size'] = SRC_VOCAB_SIZE
    config['tgt_vocab_size'] = TGT_VOCAB_SIZE
    
    model = Seq2SeqTransformer(
        NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
        NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM, DROPOUT
    )
    model = model.to(DEVICE)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SHEDULER_STEP_SIZE, gamma=0.99)
    
    model_trainer = Seq2SeqTrainer(model, optimizer, scheduler, loss_fn, dm.source_tokenizer, dm.target_tokenizer)    
    logger = TXTLogger('training_logs')

   
    trainer_cls = trainer.Trainer(model=model_trainer, config=config, logger=logger)

    if config['try_one_batch']:
        train_dataloader = [list(train_dataloader)[0]]
        dev_dataloader = [list(train_dataloader)[0]]

    trainer_cls.train(train_dataloader, dev_dataloader)
