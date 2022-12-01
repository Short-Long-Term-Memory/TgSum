# TgSum
Telegram bot that can summarize chat history into a concise abstract.
You can try it at @tg_sum_bot. If the server is down and the bot isn't working, run [this notebook](https://colab.research.google.com/drive/1FDwN9RM1uOsBHLsOL9ADHqLtw-iJBzaV?usp=sharing).

- [Ideas to try](ideas.md)
- [What we tried already](logbook/logbook.md)

## Available commands
                
Write or forward messages to the bot, then use one of the main commands :) 
Keep in mind that 'topic' commands only work if at least 10 messages are sent to the bot

### Main commands:
* **/start** : start bot
* **/collect** _\<topic\>_ : show all messages with given topic
* **/gen** generate text based on message history
* **/gen_topic** *\<topic\>* : generate text based on message with given topic
* **/sum** summarize sent messages
* **/sum_topic** *\<topic\>* : summarize sent messages with given topic
* **/clear** : clear message history

### Configs:
* **/eval** : enable/disable evaluation mode: show continuations and metrics
* **/len** _x_ : set the number of tokens to add when generating, default len=32
* **/p** _x_ : set typical_p parameter for generation, 0 < p < 1, higher p gives more random text, default p=0.5
* **/iter** _x_ : set the number of candidates to consider for summarization, default iter=3

## Project Setup

To initialize conda environment, please, run the following script in your terminal

```bash
$ conda env create -f environment.yml 
```

You can then activate new environment:
```bash
$ conda activate tg-sum
```

or add it to pycharm project settings.