from tqdm import tqdm

class Trainer:
    def __init__(self, model, config, logger):
        self.model = model
        self.epoch_num = config['epoch_num']
        self.logger = logger

        self.logger.log(config)

    def train(self, train_dataloader, val_dataloader):
        try:
            for epoch in tqdm(range(self.epoch_num)):
                train_epoch_loss = 0
                self.model.train()
                for batch in tqdm(train_dataloader):
                    train_loss = self.model.training_step(batch)
                    train_epoch_loss += train_loss
                train_epoch_loss = train_epoch_loss / len(train_dataloader)

                val_epoch_loss, val_epoch_bleu = 0, 0
                self.model.eval()
                for batch in tqdm(val_dataloader):
                    val_loss = self.model.validation_step(batch)
                    val_epoch_loss += val_loss
                val_epoch_loss = val_epoch_loss / len(val_dataloader)

                input_tensor, target_tensor = batch
                predicted_samples, _ = self.model.forward(input_tensor)
                source_sentences = input_tensor.detach().cpu().numpy()
                source_sentences = [" ".join(self.model.src_tokenizer.decode(x)) for x in source_sentences]
                bleu_score, actual_sentences, predicted_sentences = self.model.eval_bleu(predicted_samples, target_tensor)
                print('Current BLEU: ', bleu_score, 'Val Loss: ', val_epoch_loss.item(), 'Train Loss: ', train_epoch_loss.item())
                for a, b, c in zip(source_sentences[:10], actual_sentences, predicted_sentences):
                    print(f"{a} ---> {b} ---> {c}")
                print('##############################')

                self.logger.log({"val_loss": val_epoch_loss.item(),
                                 "train_loss": train_epoch_loss.item() ,
                                 "bleu_score": bleu_score})

        except KeyboardInterrupt:
            pass

        print(f"Last {epoch} epoch train loss: ", train_epoch_loss)
        print(f"Last {epoch} epoch val loss: ", val_epoch_loss)
        print(f"Last {epoch} epoch val bleu: ", bleu_score)
