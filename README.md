Finetune LLaMA

This is a combination of the following three repos:

- https://github.com/tatsu-lab/stanford_alpaca
- https://github.com/zphang/minimal-llama
- https://github.com/randaller/llama-chat

It finetunes LLaMa using the Alpaca dataset with Peft.

You need to install the packages required by the above libraries.

Steps:

1. Put `alpaca_data.json` to this directory.
2. Run `python3 alpaca.py`; this will create the trainingset and save to
   pickle.
3. Run `. train.sh`
4. Run `chat_alpaca.py` vs `chat_no_alpaca.py`
