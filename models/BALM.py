# from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    LlamaConfig,
    LlamaModel,
    LlamaTokenizer,
    GPT2Config,
    GPT2Model,
    GPT2Tokenizer,
    BertConfig,
    BertModel,
    BertTokenizer,
)
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class LearnableTemperatureInfoNCELoss(nn.Module):
    def __init__(self, initial_temperature=0.07):
        super().__init__()
        # learnbale temperature
        self.log_temperature = nn.Parameter(torch.tensor(initial_temperature).log())

    def forward(self, query, positive, negative=None):
        """
        Compute InfoNCE loss
        Args:
            query: [batch_size, seq_len, dim] - time series encoder outputs
            positive: [batch_size, seq_len, dim] - LLM outputs (positive samples)
            negative: [batch_size, seq_len, dim] - other time steps' LLM outputs (negative samples)
        """

        temperature = self.log_temperature.exp()

        query = query.mean(dim=1)
        positive = positive.mean(dim=1)
        if negative is None:
            negative = torch.roll(positive, shifts=1, dims=0)
        query = F.normalize(query, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)
        pos_logits = torch.sum(query * positive, dim=-1) / temperature
        neg_logits = torch.matmul(query, negative.t()) / temperature
        logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits], dim=-1)
        labels = torch.zeros(query.size(0), dtype=torch.long, device=query.device)
        loss = F.cross_entropy(logits, labels)
        return loss


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.seed = configs.seed
        self.data = configs.data
        self.llm_model = configs.llm_model

        if configs.llm_model == "LLAMA":
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained("huggyllama/llama-7b")
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    "huggyllama/llama-7b",
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    "huggyllama/llama-7b",
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    "huggyllama/llama-7b",
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except (
                EnvironmentError
            ):  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    "huggyllama/llama-7b",
                    trust_remote_code=True,
                    local_files_only=False,
                )
        elif configs.llm_model == "GPT2":
            self.gpt2_config = GPT2Config.from_pretrained("openai-community/gpt2")

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    "openai-community/gpt2",
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    "openai-community/gpt2",
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    "openai-community/gpt2",
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except (
                EnvironmentError
            ):  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    "openai-community/gpt2",
                    trust_remote_code=True,
                    local_files_only=False,
                )
        elif configs.llm_model == "BERT":
            self.bert_config = BertConfig.from_pretrained(
                "google-bert/bert-base-uncased"
            )

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    "google-bert/bert-base-uncased",
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    "google-bert/bert-base-uncased",
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    "google-bert/bert-base-uncased",
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except (
                EnvironmentError
            ):  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    "google-bert/bert-base-uncased",
                    trust_remote_code=True,
                    local_files_only=False,
                )
        else:
            raise Exception("LLM model is not defined")

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = "[PAD]"
            self.tokenizer.add_special_tokens({"pad_token": pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False
            # param.requires_grad = True

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout
        )

        self.backcast_len = configs.seq_len
        # mask
        self.mask_rate = configs.mask_rate

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.prompt_len = min(
            self.patch_nums, int(self.patch_nums * configs.pred_len / configs.seq_len)
        )
        self.head_nf = self.d_ff * (self.prompt_len + self.patch_nums)

        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            if self.mask_rate == 0:
                self.output_projection = FlattenHead(
                    configs.enc_in,
                    self.head_nf,
                    self.pred_len,
                    head_dropout=configs.dropout,
                )
            else:
                self.output_projection = FlattenHead(
                    configs.enc_in,
                    self.head_nf,
                    self.pred_len + self.backcast_len,
                    head_dropout=configs.dropout,
                )
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

        self.tsprojection = nn.Linear(configs.d_model, configs.llm_dim)

        self.alignment_weight = 1

        # learnable prompt
        self.learnable_prompt = nn.Parameter(torch.randn(self.prompt_len, self.d_llm))

        self.LearnableTemperatureInfoNCELoss = LearnableTemperatureInfoNCELoss(
            initial_temperature=0.07
        )

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        epoch,
        batch,
        mode,
        mask=None,
        batch_y=None,
    ):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out, alignment_loss = self.forecast(
                x_enc, x_mark_enc, x_dec, x_mark_dec, epoch, batch, mode, batch_y
            )
            # Return both the prediction and the alignment loss

            if mode == "train":
                # Return forecast output, alignment loss, trend loss, and adversarial losses
                return dec_out[:, -self.pred_len :, :], alignment_loss
            else:
                # For validation and testing, return forecast output only
                return dec_out[:, -self.pred_len :, :]
        return None

    def forecast(
        self, x_enc, x_mark_enc, x_dec, x_mark_dec, epoch, batch, mode, batch_y=None
    ):
        x_enc = self.normalize_layers(x_enc, "norm")  # torch.Size([24, 96, 7])

        B, T, N = x_enc.size()

        x_enc = (
            x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        )  # torch.Size([192, 512, 1])

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)
        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())

            prompt_ = (
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}, "
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()  # ([24, 96, 7])

        prompt = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(
            prompt.to(x_enc.device)
        )  # (batch, prompt_token, dim)

        batch_size, current_length, dim = prompt_embeddings.shape

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))

        Learn_prompt = self.learnable_prompt.unsqueeze(0).repeat(batch_size, 1, 1)
        prompt_embeddings_all = torch.cat([Learn_prompt, prompt_embeddings], dim=1)
        # Rest of processing remains the same
        llama_enc_out = prompt_embeddings_all

        dec_out = self.llm_model(inputs_embeds=llama_enc_out)

        llm_hidden_states = dec_out.last_hidden_state

        dec_out = llm_hidden_states[:, -self.prompt_len :, :]

        TS = enc_out
        TS = self.tsprojection(TS)

        # Scale-STD
        text_mean = dec_out.mean(dim=1, keepdim=True)  # [batch, 1, d_llm]
        text_std = torch.sqrt(
            ((dec_out - text_mean) ** 2).mean(dim=1, keepdim=True) + 1e-8
        )  # [batch, 1, d_llm]

        ts_mean = TS.mean(dim=1, keepdim=True)  # [batch, 1, d_llm]
        ts_std = torch.sqrt(
            ((TS - ts_mean) ** 2).mean(dim=1, keepdim=True) + 1e-8
        )  # [batch, 1, d_llm]
        scale_factor = text_std / (ts_std + 1e-8)  # [batch, 1, d_llm]

        dec_out = dec_out / scale_factor

        infonce_loss = self.LearnableTemperatureInfoNCELoss(TS, dec_out)
        alignment_loss = self.alignment_weight * infonce_loss

        dec_out = torch.cat(
            (dec_out[:, :, : self.d_ff], TS[:, :, : self.d_ff]), dim=1
        )

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1])
        )
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, :])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, "denorm")

        return dec_out, alignment_loss
    
    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags