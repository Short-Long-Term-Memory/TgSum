from sys import argv
from time import sleep

from origamibot import OrigamiBot as Bot
from origamibot.listener import Listener

from lm import LM


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
                "/gen [len] -- the next message will be considered a prompt for text generation\n"
            ),
        )

    def echo(self, message, value: str):  # /echo [value: str] command
        self.bot.send_message(message.chat.id, value)

    def gen(self, message, length: int = 32):
        self.bot.task[message.chat.id] = ("gen", length)


class MessageListener(Listener):  # Event listener must inherit Listener
    def __init__(self, bot):
        self.bot = bot

    def on_message(self, message):  # called on every message
        chat = message.chat.id
        if chat not in self.bot.task:
            self.bot.send_message(
                chat,
                "I don't know what to do with this text, there is no active command.",
            )
            return
        task = self.bot.task[chat]
        if task[0] == "gen":
            length = task[1]
            self.bot.send_message(chat, self.bot.lm.generate(message, length))
        else:
            self.bot.send_message(
                chat, f"Something went wrong, I don't understand this task: {task}"
            )

    def on_command_failure(self, message, err=None):  # When command fails
        if err is None:
            self.bot.send_message(message.chat.id, "Command failed to bind arguments!")
        else:
            self.bot.send_message(message.chat.id, "Error in command:\n{err}")


if __name__ == "__main__":
    token = "5935410865:AAHT5iX3iVWVogquC9m6uRu8JMZcnBxF9jc"

    bot = Bot(token)
    bot.checkpoint = "dummy"
    bot.lm = LM.from_pretrained(bot.checkpoint)
    bot.task = dict()

    bot.add_listener(MessageListener(bot))
    bot.add_commands(BotsCommands(bot))
    bot.start()
    while True:
        sleep(1)
        print("working")
