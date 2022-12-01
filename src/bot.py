from time import sleep
from collections import defaultdict
from typing import Optional

from origamibot import OrigamiBot as Bot
from origamibot.listener import Listener

from metric_experiments.lm import LM
from topic_modeling.topics import TopicsCollector


class TgSumBot(Bot):
    def __init__(self, token: str, topic_model: TopicsCollector, model_name: str = "EleutherAI/gpt-neo-125M"):
        super().__init__(token)
        self.settings = defaultdict(dict)
        self.lm = LM.from_pretrained(model_name)
        self.checkpoint = model_name
        self.topic_model = topic_model


class BotsCommands:
    def __init__(self, bot: TgSumBot):  # Can initialize however you like
        self.bot = bot

    def start(self, message):  # /start command
        self.bot.send_message(
            message.chat.id,
            (
                "Hi! Currently supported commands are:\n"
                "/start -- just print this message\n\n"
                
                "Write or forward messages to the bot, then use one of the main commands :) "
                "Keep in mind that 'topic' commands only work if at least 10 messages are sent to the bot\n\n"

                "Main commands:\n"
                "/collect <topic> -- show all messages with given <topic>\n"
                "/gen -- generate text based on message history\n"
                "/gen_topic <topic> -- generate text based on message with given topic\n"
                "/sum -- summarize sent messages\n"
                "/sum_topic <topic> -- summarize sent messages with given topic\n"
                "/clear -- clear message history\n\n"

                "Configs: \n"
                "/eval -- enable/disable evaluation mode: show continuations and metrics\n"
                "/len x -- set the number of tokens to add when generating, default len=32\n"
                "/p x -- set typical_p parameter for generation, 0 < p < 1,"
                        "higher p gives more random text, default p=0.5\n"
                "/iter x -- set the number of candidates to consider for summarization, default iter=3"
            ),
        )
        settings = self.bot.settings[message.chat.id]
        settings["is_init"] = True
        settings["len"] = 32
        settings["p"] = 0.5
        settings["iter"] = 3
        settings["text"] = []
        settings["eval"] = False

    def collect(self, message, value: str):
        self.bot.send_message(message.chat.id, self.__get_text(message.chat.id, value))

    def sum(self, message):
        self.__sum(message)

    def sum_topic(self, message, value: str):
        self.__sum(message, value)

    def gen(self, message):
        self.__gen(message)

    def gen_topic(self, message, value: str):
        self.__gen(message, value)

    def __gen(self, message, value: Optional[str] = None):
        chat = message.chat.id
        settings = self.bot.settings[chat]

        if not settings.get("is_init", False):
            self.bot.send_message(chat, "Please, run /start command to init bot")
            return

        length, p = settings["len"], settings["p"]
        prompt = self.__get_text(chat, value)

        if prompt == "":
            self.bot.send_message(chat, "Nothing to generate")
            return

        self.bot.send_message(chat, f"Generating {length} tokens with p={p}...")
        self.bot.send_message(
            chat, self.bot.lm.generate_from_text(prompt, length=length, p=p)
        )

    def __sum(self, message, value: Optional[str] = None):
        chat = message.chat.id
        settings = self.bot.settings[chat]

        if not settings.get("is_init", False):
            self.bot.send_message(chat, "Please, run /start command to init bot")
            return

        text = self.__get_text(chat, value)

        if text == "":
            self.bot.send_message(chat, "Nothing to summarize")
            return

        length, p, iterations = settings["len"], settings["p"], settings["iter"]
        prompt = f"Text: {text} Summary:"
        self.bot.send_message(
            chat, f"Generating {length} tokens {iterations} times with p={p}..."
        )

        conts = []
        if settings["eval"]:
            for p in [0.25, 0.5, 0.95]:
                text_length = len(self.bot.lm.text_to_ids(text))
                cont = ""
                while not cont:
                    cont = self.bot.lm.generate_from_text(
                        text, length=text_length, p=p
                    )[len(text):]
                conts.append(cont)
                self.bot.send_message(chat, f"Continuation with p={p}:\n{cont}")

        for it in range(iterations):
            full = self.bot.lm.generate_from_text(prompt, length=length, p=p)
            summary = full[len(prompt):]
            self.bot.send_message(chat, f"Summary #{it + 1}:")
            self.bot.send_message(chat, summary)
            if settings["eval"]:
                raw_score = self.bot.lm.loss_str(summary, text)
                prompted_score = self.bot.lm.loss_str(
                    f"Summary: {summary} Text:", text
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

    def clear(self, message):
        settings = self.bot.settings[message.chat.id]
        settings["text"] = []
        self.bot.send_message(message.chat.id, "Message history cleared")

    def __get_text(self, chat_id: int, topic: str = None) -> str:
        text = self.bot.settings[chat_id]["text"]
        if len(text) >= TopicsCollector.MIN_MSG_NUM and topic is not None and topic != "":
            text = self.bot.topic_model.find_messages_with_topic(text, topic)
            print(len(text))

        return "\n".join(text)


    def len(self, message, value: int):
        settings = self.bot.settings[message.chat.id]
        settings["len"] = value

    def p(self, message, value: float):
        settings = self.bot.settings[message.chat.id]
        settings["p"] = value

    def eval(self, message):
        settings = self.bot.settings[message.chat.id]
        settings["eval"] = not settings.get("eval", False)
        self.bot.send_message(message.chat.id, f"Evaluation mode: {settings['eval']}")


class MessageListener(Listener):
    def __init__(self, bot):
        self.bot = bot

    def on_plain_message(self, message):
        print("on_plain_message", message)
        text = message.text if message.text is not None else message.caption
        user = "@" + (message.forward_from.username if message.forward_from is not None else message.from_user.username)
        if text is not None:
            self.bot.settings[message.chat.id]["text"].append(user + ": " + text)


def main_loop():
    token = "5935410865:AAHT5iX3iVWVogquC9m6uRu8JMZcnBxF9jc"
    topic_collector = TopicsCollector()
    bot = TgSumBot(token, topic_collector)
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
