from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing


class BPETokenizer:
    def __init__(self, sentence_list):
        """
        sentence_list - список предложений для обучения
        """
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.special_tokens = ["[UNK]", "[BOS]", "[EOS]", "[PAD]", "[MASK]"]
        self.trainer = BpeTrainer(vocab_size=4_000, special_tokens=self.special_tokens)
        self.tokenizer.train_from_iterator(sentence_list, trainer=self.trainer)
        self.tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[
                ("[BOS]", self.tokenizer.token_to_id("[BOS]")),
                ("[EOS]", self.tokenizer.token_to_id("[EOS]")),
            ],
        )
        # TODO: Реализуйте конструктор c помощью https://huggingface.co/docs/transformers/fast_tokenizers, обучите токенизатор, подготовьте нужные аттрибуты(word2index, index2word)
        self.pad_ids = self.tokenizer.token_to_id("[PAD]")
        self.max_len = 15

    def __call__(self, sentence):
        """
        sentence - входное предложение
        """
        # TODO: Реализуйте метод токенизации с помощью обученного токенизатора
        ids = self.tokenizer.encode(sentence).ids
        ids = self.pad_sequence(ids)
        return ids


    def decode(self, token_list):
        """
        token_list - предсказанные ID вашего токенизатора
        """
        # TODO: Реализуйте метод декодирования предсказанных токенов

        return self.tokenizer.decode(token_list).split()
        
    def pad_sequence(self, ids):
        ids = ids[:self.max_len] + [self.pad_ids] * max(0, self.max_len - len(ids))
        return ids
