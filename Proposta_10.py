import autosklearn.metrics
import pandas as pd
import zipfile
import requests
import io
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, log_loss, precision_recall_curve, roc_curve, auc
from imblearn.combine import SMOTEENN
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from tqdm import tqdm
from autosklearn.classification import AutoSklearnClassifier
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from learn2learn.algorithms import MAML
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from textblob import TextBlob

# Configurar tqdm para pandas
tqdm.pandas()

# Definir parâmetros
BATCH_SIZE = 8
MAX_LEN = 128
LR = 2e-5
EPOCHS = 3
STEPS_PER_TASK = 5

# Usar CPU
device = torch.device("cpu")
print(f"Usando dispositivo: {device}")

# Função de pré-carregamento
def preload(step_name):
    print(f"Preload: {step_name} concluído com sucesso.")

# Baixar e extrair o conjunto de dados
print("Baixando e extraindo o conjunto de dados...")
url = 'https://www.cs.ucsb.edu/~william/data/liar_dataset.zip'
response = requests.get(url)
zip_document = zipfile.ZipFile(io.BytesIO(response.content))
zip_document.extractall('./liar_dataset')
preload("Download e extração do conjunto de dados")

# Carregar o conjunto de dados de treinamento, validação e teste
print("Carregando os conjuntos de dados...")
train_data = pd.read_csv('./liar_dataset/train.tsv', sep='\t', header=None)
valid_data = pd.read_csv('./liar_dataset/valid.tsv', sep='\t', header=None)
test_data = pd.read_csv('./liar_dataset/test.tsv', sep='\t', header=None)

# Adicionar nomes às colunas para facilitar a manipulação
columns = ["id", "label", "statement", "subject", "speaker", "speaker_job",
           "state", "party", "barely_true_count", "false_count", "half_true_count",
           "mostly_true_count", "pants_on_fire_count", "context"]
train_data.columns = columns
valid_data.columns = columns
test_data.columns = columns
preload("Carregamento e renomeação dos conjuntos de dados")

# Eliminação: Remover colunas irrelevantes
print("Removendo colunas irrelevantes...")
train_data = train_data[['label', 'statement']]
valid_data = valid_data[['label', 'statement']]
test_data = test_data[['label', 'statement']]
preload("Remoção de colunas irrelevantes")

# Reduzir o tamanho do conjunto de dados para execução rápida
train_data = train_data.sample(frac=0.05, random_state=42).reset_index(drop=True)
valid_data = valid_data.sample(frac=0.05, random_state=42).reset_index(drop=True)
test_data = test_data.sample(frac=0.05, random_state=42).reset_index(drop=True)
preload("Redução do tamanho dos conjuntos de dados")

# Baixar recursos necessários do NLTK, como stopwords e o lematizador
print("Baixando recursos necessários do NLTK...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
preload("Download de recursos NLTK")

# Inicializar o lematizador e definir as stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Função para pré-processamento de texto (Limpeza e Transformação)
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove caracteres especiais
    text = re.sub(r'\d', ' ', text)  # Remove números
    text = text.lower()  # Converte para minúsculas
    text = re.sub(r'\s+', ' ', text)  # Remove múltiplos espaços em branco
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Aplicar pré-processamento aos dados de treinamento, validação e teste
print("Aplicando pré-processamento aos dados...")
train_data['clean_statement'] = train_data['statement'].progress_apply(preprocess_text)
valid_data['clean_statement'] = valid_data['statement'].progress_apply(preprocess_text)
test_data['clean_statement'] = test_data['statement'].progress_apply(preprocess_text)
preload("Pré-processamento de textos")

# Vetorização TF-IDF dos dados de texto (Transformação)
print("Vetorizando os dados de texto com TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['clean_statement'])
X_valid_tfidf = tfidf_vectorizer.transform(valid_data['clean_statement'])
X_test_tfidf = tfidf_vectorizer.transform(test_data['clean_statement'])
preload("Vetorização TF-IDF")

# Label Encoding das labels (Transformação)
print("Codificando as labels...")
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data['label'])
y_valid = label_encoder.transform(valid_data['label'])
y_test = label_encoder.transform(test_data['label'])
preload("Codificação de labels")

# Balanceamento dos dados de treinamento usando SMOTEENN (Desbalanceamento)
print("Balanceando os dados de treinamento com SMOTEENN...")
smote_enn = SMOTEENN(random_state=42)
X_train_balanced, y_train_balanced = smote_enn.fit_resample(X_train_tfidf, y_train)
preload("Balanceamento dos dados com SMOTEENN")

# Função para treinar e avaliar o modelo AutoML com validação cruzada
def train_auto_ml_model(X_train, y_train):
    automl_classifier = AutoSklearnClassifier(
        time_left_for_this_task=60 * 20,  # 20 minutos
        per_run_time_limit=60,  # 1 minuto por modelo
        initial_configurations_via_metalearning=25,  # Inicializações via metalearning
        metric=autosklearn.metrics.accuracy,  # Métrica de acurácia
    )
    automl_classifier.fit(X_train, y_train)

    # Validação cruzada
    cv_scores = cross_val_score(automl_classifier, X_train, y_train, cv=5)
    return automl_classifier, cv_scores

# Treinamento e avaliação do modelo AutoML
print("Treinando e avaliando o modelo AutoML com validação cruzada...")
auto_ml_model, cv_scores = train_auto_ml_model(X_train_balanced, y_train_balanced)
print(f"Acurácia média na validação cruzada: {np.mean(cv_scores)}")
print(f"Desvio padrão na validação cruzada: {np.std(cv_scores)}")
preload("Treinamento e avaliação do modelo AutoML com validação cruzada")

# Avaliação no conjunto de validação
y_pred_valid = auto_ml_model.predict(X_valid_tfidf)
valid_accuracy = accuracy_score(y_valid, y_pred_valid)
valid_report = classification_report(y_valid, y_pred_valid, zero_division=0)
print(f"Acurácia no conjunto de validação: {valid_accuracy}")
print(f"Relatório de classificação no conjunto de validação:\n{valid_report}")

# Avaliação no conjunto de teste
y_pred_test = auto_ml_model.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_pred_test)
test_report = classification_report(y_test, y_pred_test, zero_division=0)
print(f"Acurácia no conjunto de teste: {test_accuracy}")
print(f"Relatório de classificação no conjunto de teste:\n{test_report}")

# Inicializar o tokenizador DistilBERT
print("Inicializando o tokenizador DistilBERT...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Função para criar DataLoader para os dados (Adequação)
class FakeNewsDataset(Dataset):
    def __init__(self, statements, labels, tokenizer, max_len):
        self.statements = statements
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.statements)

    def __getitem__(self, item):
        statement = self.statements[item]
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            statement,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Função para criar DataLoader (Adequação)
def create_data_loader(statements, labels, tokenizer, max_len, batch_size):
    dataset = FakeNewsDataset(
        statements=statements,
        labels=labels,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)

# Criar DataLoader para os dados de treinamento, validação e teste
print("Criando DataLoader para os dados...")
train_data_loader = create_data_loader(train_data['clean_statement'], y_train, tokenizer, MAX_LEN, BATCH_SIZE)
valid_data_loader = create_data_loader(valid_data['clean_statement'], y_valid, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(test_data['clean_statement'], y_test, tokenizer, MAX_LEN, BATCH_SIZE)
preload("Criação de DataLoader")

# Função para treinar e avaliar o modelo MAML
class FakeNewsModel(nn.Module):
    def __init__(self):
        super(FakeNewsModel, self).__init__()
        self.bert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).logits

# Inicializar modelo MAML
print("Inicializando o modelo MAML...")
base_model = FakeNewsModel()
maml = MAML(base_model, lr=LR, first_order=True).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(maml.parameters(), lr=LR)
preload("Inicialização do modelo MAML")

# Função para treinamento e avaliação de cada tarefa
def train_task(task_data_loader):
    maml.train()
    task_loss = 0
    for batch in task_data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = maml(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        # Backward pass e otimização
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        task_loss += loss.item()
    return task_loss / len(task_data_loader)

# Treinar modelo MAML
print("Treinando o modelo MAML...")
for epoch in range(EPOCHS):
    train_loss = train_task(train_data_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {train_loss:.4f}")

# Função para avaliação
def evaluate(model, data_loader):
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    return y_true, y_pred

# Avaliar no conjunto de validação
print("Avaliando no conjunto de validação...")
y_valid_true, y_valid_pred = evaluate(maml, valid_data_loader)
valid_maml_accuracy = accuracy_score(y_valid_true, y_valid_pred)
valid_maml_report = classification_report(y_valid_true, y_valid_pred, zero_division=0)
print(f"Acurácia no conjunto de validação MAML: {valid_maml_accuracy}")
print(f"Relatório de classificação no conjunto de validação MAML:\n{valid_maml_report}")

# Avaliar no conjunto de teste
print("Avaliando no conjunto de teste...")
y_test_true, y_test_pred = evaluate(maml, test_data_loader)
test_maml_accuracy = accuracy_score(y_test_true, y_test_pred)
test_maml_report = classification_report(y_test_true, y_test_pred, zero_division=0)
print(f"Acurácia no conjunto de teste MAML: {test_maml_accuracy}")
print(f"Relatório de classificação no conjunto de teste MAML:\n{test_maml_report}")

# Análise de sentimento com TextBlob
print("Realizando análise de sentimento com TextBlob...")
train_data['polarity'] = train_data['statement'].apply(lambda x: TextBlob(x).sentiment.polarity)
train_data['subjectivity'] = train_data['statement'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# Exibir frases consideradas como discurso de ódio
hate_speech_threshold = 0.5  # Defina o limiar conforme necessário
hate_speech_statements = train_data[train_data['polarity'] < -hate_speech_threshold][['statement', 'polarity', 'subjectivity']]

print("Frases consideradas como discurso de ódio:")
for index, row in hate_speech_statements.iterrows():
    print(f"Sentença: {row['statement']}\nPolaridade: {row['polarity']}\nSubjectividade: {row['subjectivity']}\n")


# Exibir e salvar frases classificadas como fake news
print("Exibindo e salvando frases classificadas como fake news...")
fake_news = test_data[test_data['label'] == 'pants-fire']
fake_news_statements = fake_news['statement'].tolist()  # Converte para lista

# Exibir as frases de fake news
for i, statement in enumerate(fake_news_statements, 1):
    print(f"Fake news {i}: {statement}")

# Salvar as frases de fake news em um arquivo CSV
fake_news.to_csv('fake_news_statements.csv', index=False)
print("Frases de fake news salvas em 'fake_news_statements.csv'")

# Verificar se sentenças de fake news estão no conjunto de discurso de ódio
print("Verificando se sentenças de fake news estão no conjunto de discurso de ódio...")
fake_news_in_hate_speech_count = 0
for statement in fake_news_statements:  # Itera diretamente sobre a lista
    if statement in hate_speech_statements['statement'].values:
        fake_news_in_hate_speech_count += 1
        print(f"Sentença: {statement}\n")

print(f"Número de sentenças de fake news encontradas no conjunto de discurso de ódio: {fake_news_in_hate_speech_count}")


# Gerar a nuvem de palavras
print("Gerando a nuvem de palavras...")
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(train_data['clean_statement']))

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Nuvem de Palavras - Declarações do Conjunto de Treinamento")
plt.savefig("nuvem_de_palavras.png")
plt.show()
preload("Geração da nuvem de palavras")

print("Pipeline concluído com sucesso!")

