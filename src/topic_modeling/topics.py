from typing import List
from bertopic import BERTopic
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords


class TopicsCollector:
    MIN_MSG_NUM = 10
    def __init__(self, language: str = "english", embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.emb_model = embedding_model
        self.lang = language

    def find_messages_with_topic(self, messages: List[str], topic: str) -> List[str]:
        topic_model = BERTopic(language=self.lang, embedding_model=self.emb_model)
        prcsd_msgs = self.__preprocess(messages)
        topics, _ = topic_model.fit_transform(prcsd_msgs)
        relevant_topics, _ = topic_model.find_topics(topic, 2)
        relevant_topic = relevant_topics[0] if relevant_topics[0] != -1 or len(relevant_topics) < 2 else relevant_topics[1]

        relevant_messages_ids = [i for i, t in enumerate(topics) if t == relevant_topic]
        return [messages[i] for i in relevant_messages_ids]

    def __preprocess(self, text: List[str]) -> List[str]:
        return [self.__normalize(sentence) for sentence in text]

    def __normalize(self, sentence: str) -> str:
        eng_stopwords = stopwords.words(self.lang)
        tokens = []
        for token in simple_preprocess(sentence, min_len=1):
            if token not in eng_stopwords:
                tokens.append(token)
        return " ".join(tokens)