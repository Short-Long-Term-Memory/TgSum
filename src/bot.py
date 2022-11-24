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
                "/len x -- set the number of tokens to add when generating\n"
                "/k x -- set the top_k parameter in contrastive search (larger are better, but slower)\n"
                "/alpha x -- set the penalty_alpha parameter in contrastive search (larger are less repetitive, but strange) \n"
            ),
        )

    def echo(self, message, value: str):  # /echo [value: str] command
        self.bot.send_message(message.chat.id, value)

    def gen(self, message):
        settings = self.bot.settings[message.chat.id]
        settings["task"] = "gen"
        settings["len"] = 32
        settings["k"] = 3
        settings["alpha"] = 0.5

    def len(self, message, value: int):
        settings = self.bot.settings[message.chat.id]
        settings["len"] = value
    
    def k(self, message, value: int):
        settings = self.bot.settings[message.chat.id]
        settings["k"] = value

    def alpha(self, message, value: float):
        settings = self.bot.settings[message.chat.id]
        settings["alpha"] = value

class MessageListener(Listener):
    def __init__(self, bot):
        self.bot = bot

    def on_message(self, message):
        if message.text[:1] == '/':
            return
        print("on_message", message)
        chat = message.chat.id
        if chat not in self.bot.settings:
            return
        settings = self.bot.settings[chat]
        if settings["task"] == "gen":
            length, k, alpha = settings["len"], settings["k"], settings["alpha"]
            prompt = message.text
            print("from", prompt)
            self.bot.send_message(chat, f"Generating {length} tokens with k={k}, alpha={alpha}...")
            self.bot.send_message(
                chat, self.bot.lm.generate_from_text(prompt, length=length, k=k, alpha=alpha)
            )
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
    bot.checkpoint = "gpt2"
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
