import os
import tweepy
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql.functions import col, when

class TwitterStreamApp:
    def __init__(self):
        # Configuração do Spark
        self.conf = SparkConf().setAppName("TwitterStreamApp").setMaster("local[*]")
        self.sc = SparkContext(conf=self.conf)
        self.ssc = StreamingContext(self.sc, 10)
        self.spark = SparkSession.builder.config(conf=self.conf).getOrCreate()

        # Configuração da API do Twitter
        self.consumer_key = os.environ['TWITTER_CONSUMER_KEY']
        self.consumer_secret = os.environ['TWITTER_CONSUMER_SECRET']
        self.access_token = os.environ['TWITTER_ACCESS_TOKEN']
        self.access_token_secret = os.environ['TWITTER_ACCESS_TOKEN_SECRET']

        # Autenticação do Twitter
        auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
        auth.set_access_token(self.access_token, self.access_token_secret)
        self.api = tweepy.API(auth)

    def get_tweets(self):
        class MyStreamListener(tweepy.StreamListener):
            def __init__(self, sc):
                super(MyStreamListener, self).__init__()
                self.rdd_queue = []

            def on_status(self, status):
                self.rdd_queue.append(status.text)

        listener = MyStreamListener(self.sc)
        stream = tweepy.Stream(auth=self.api.auth, listener=listener)
        stream.filter(track=['python', 'spark', 'big data'], is_async=True)
        return listener.rdd_queue

    def process_rdd(self, rdd):
        if not rdd.isEmpty():
            df = self.spark.createDataFrame(rdd.map(lambda x: (x,)), ["text"])

            # Tokenização dos textos
            tokenizer = Tokenizer(inputCol="text", outputCol="words")
            words_data = tokenizer.transform(df)

            # Extração de características
            hashing_tf = HashingTF(inputCol="words", outputCol="features")
            featurized_data = hashing_tf.transform(words_data)

            # Simulação de labels (substitua com lógica real para definir labels)
            labeled_data = featurized_data.withColumn("label", when(col("text").contains("positive"), 1.0).otherwise(0.0))

            # Treinamento do modelo
            lr = LogisticRegression(maxIter=10, regParam=0.01)
            model = lr.fit(labeled_data)

            # Imprime um resumo do modelo treinado
            print(f"Coefficients: {model.coefficients} Intercept: {model.intercept}")

    def run(self):
        # Captura de tweets
        rdd_queue = self.get_tweets()

        # Criação de um DStream de tweets
        tweets = self.ssc.queueStream([self.sc.parallelize(rdd_queue)])

        # Extração do texto dos tweets
        statuses = tweets.map(lambda status: status)

        # Processamento dos dados e treinamento do modelo
        statuses.foreachRDD(lambda rdd: self.process_rdd(rdd))

        # Inicialização do contexto de streaming
        self.ssc.start()
        self.ssc.awaitTermination()

if __name__ == "__main__":
    # Configuração das credenciais do Twitter
    os.environ['TWITTER_CONSUMER_KEY'] = 'CONSUMER_KEY'
    os.environ['TWITTER_CONSUMER_SECRET'] = 'CONSUMER_SECRET'
    os.environ['TWITTER_ACCESS_TOKEN'] = 'ACCESS_TOKEN'
    os.environ['TWITTER_ACCESS_TOKEN_SECRET'] = 'ACCESS_TOKEN_SECRET'

    # Criação e execução da aplicação
    app = TwitterStreamApp()
    app.run()
