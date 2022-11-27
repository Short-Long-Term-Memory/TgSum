from sys import argv
from time import sleep
from collections import defaultdict

from origamibot import OrigamiBot as Bot
from origamibot.listener import Listener

from metric_experiments.lm import LM


class BotsCommands:
    def __init__(self, bot: Bot):  # Can initialize however you like
        self.bot = bot

    def start(self, message):  # /start command
        self.bot.send_message(
            message.chat.id,
            (
                "Hi! Currently supported commands are:\n"
                "/start -- just print this message\n"
                "/echo str -- sends str back to you\n"
                "/gen -- the next non-command message will be considered a prompt for text generation\n"
                "/sum -- the next non-command message will be considered a prompt for text summarization\n"
                "/len x -- set the number of tokens to add when generating\n"
                "/p x -- set typical_p parameter for generation, 0 < p < 1, higher p gives more random text\n"
                "/iter x -- set the number of candidates to consider for summarization\n"
            ),
        )
        settings = self.bot.settings[message.chat.id]
        settings["task"] = "gen"
        settings["len"] = 32
        settings["p"] = 0.5
        settings["iter"] = 3

    def echo(self, message, value: str):
        self.bot.send_message(message.chat.id, value)

    def gen(self, message):
        settings = self.bot.settings[message.chat.id]
        settings["task"] = "gen"

    def sum(self, message):
        settings = self.bot.settings[message.chat.id]
        settings["task"] = "sum"

    def len(self, message, value: int):
        settings = self.bot.settings[message.chat.id]
        settings["len"] = value

    def p(self, message, value: float):
        settings = self.bot.settings[message.chat.id]
        settings["p"] = value


class MessageListener(Listener):
    def __init__(self, bot):
        self.bot = bot

    def on_message(self, message):
        if message.text[:1] == "/":
            return
        print("on_message", message)
        chat = message.chat.id
        if chat not in self.bot.settings:
            return
        settings = self.bot.settings[chat]
        task = settings["task"]
        if task == "gen":
            length, p = settings["len"], settings["p"]
            prompt = message.text
            self.bot.send_message(chat, f"Generating {length} tokens with p={p}...")
            self.bot.send_message(
                chat, self.bot.lm.generate_from_text(prompt, length=length, p=p)
            )
        elif task == "sum":
            length, p, iter = settings["len"], settings["p"], settings["iter"]
            prompt = f"Text: {message.text} Summary:"
            self.bot.send_message(
                chat, f"Generating {length} tokens {iter} times with p={p}..."
            )
            conts = []
            for p in [0.25, 0.5, 0.95]:
                text_length = len(self.bot.lm.text_to_ids(message.text))
                cont = ""
                while not cont:
                    cont = self.bot.lm.generate_from_text(
                        message.text, length=text_length, p=p
                    )[len(message.text) :]
                conts.append(cont)
                self.bot.send_message(chat, f"Continuation with p={p}:\n{cont}")

            for it in range(iter):
                full = self.bot.lm.generate_from_text(prompt, length=length, p=p)
                summary = full[len(prompt) :]
                self.bot.send_message(chat, f"Summary #{it + 1}:")
                self.bot.send_message(chat, summary)
                raw_score = self.bot.lm.loss_str(summary, message.text)
                prompted_score = self.bot.lm.loss_str(
                    f"Summary: {summary} Text:", message.text
                )
                cont_scores = []
                for i, cont in enumerate(conts):
                    cont_score = self.bot.lm.loss_str(f"Summary: {summary} Text:", cont)
                    cont_scores.append(cont_score.item())
                cont_score = sum(cont_scores) / len(cont_scores)
                self.bot.send_message(
                    chat,
                    f"raw_score = {raw_score}, prompted_score = {prompted_score}, cont_score = {cont_score}",
                )
                self.bot.send_message(chat, f"cont_scores = {cont_scores}")
        elif task:
            self.bot.send_message(
                chat, f"Something went wrong, I don't understand this task: {task}"
            )

    def on_command_failure(self, message, err=None):  # When command fails
        if err is None:
            self.bot.send_message(message.chat.id, "Command failed to bind arguments!")
        else:
            self.bot.send_message(message.chat.id, "Error in command:\n{err}")


def main_loop():
    token = "5935410865:AAHT5iX3iVWVogquC9m6uRu8JMZcnBxF9jc"

    bot = Bot(token)
    # bot.checkpoint = "EleutherAI/gpt-neo-125M"
    bot.checkpoint = "../baseline_models/t5-base_ft"
    # bot.checkpoint = "distilgpt2"
    bot.lm = LM.from_pretrained(bot.checkpoint)
    bot.settings = defaultdict(dict)

    bot.add_listener(MessageListener(bot))
    bot.add_commands(BotsCommands(bot))
    bot.start()
    print("started")
    while True:
        sleep(1)


if __name__ == "__main__":
    while True:
        try:
            main_loop()
        except Exception as e:
            print(e)
