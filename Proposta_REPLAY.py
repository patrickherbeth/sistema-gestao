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
from sklearn.metrics import accuracy_score, classification_report
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
from avalanche.benchmarks import nc_benchmark
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, timing_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.training.supervised import Naive

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
        statement = str(self.statements[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            statement,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'statement_text': statement,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = FakeNewsDataset(
        statements=df.clean_statement.to_numpy(),
        labels=label_encoder.transform(df.label.to_numpy()),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )

# Criar DataLoaders para treinamento, validação e teste
print("Criando DataLoaders para treinamento, validação e teste...")
train_data_loader = create_data_loader(train_data, tokenizer, MAX_LEN, BATCH_SIZE)
valid_data_loader = create_data_loader(valid_data, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(test_data, tokenizer, MAX_LEN, BATCH_SIZE)
preload("Criação dos DataLoaders")

# Inicializar o modelo DistilBERT pré-treinado para classificação
print("Inicializando o modelo DistilBERT pré-treinado para classificação...")
class FakeNewsClassifier(nn.Module):
    def __init__(self, n_classes):
        super(FakeNewsClassifier, self).__init__()
        self.bert = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', num_labels=n_classes
        )

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask)

model = FakeNewsClassifier(n_classes=len(label_encoder.classes_))
model = model.to(device)
preload("Inicialização do modelo DistilBERT")

# Configurar otimizador e função de perda
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss().to(device)

# Função para treinamento do modelo MAML (Treinamento)
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader, desc="Treinando"):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)
        loss = loss_fn(outputs.logits, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

# Função para avaliação do modelo MAML (Avaliação)
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Avaliando"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            loss = loss_fn(outputs.logits, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

# Configurar MAML com REPLAY (Integração)
print("Configurando MAML com REPLAY...")
maml = MAML(model, lr=LR, first_order=False)
optimizer = optim.Adam(maml.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEPS_PER_TASK, gamma=0.1)

interactive_logger = InteractiveLogger()

evaluation_plugin = EvaluationPlugin(
    accuracy_metrics(experience=True, stream=True),
    loss_metrics(experience=True, stream=True),
    timing_metrics(experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    loggers=[interactive_logger]
)

# Política de armazenamento REPLAY
storage_policy = ReplayPlugin(mem_size=100)

strategy = Naive(
    model, optimizer, loss_fn, train_epochs=EPOCHS, eval_every=1,
    evaluator=evaluation_plugin, device=device, plugins=[storage_policy]
)

# Função de treinamento e avaliação usando REPLAY (Treinamento)
def train_and_evaluate_replay():
    benchmark = nc_benchmark(
        train_data=train_data_loader,
        test_data=test_data_loader,
        n_experiences=5,
        task_labels=False,
        shuffle=True
    )

    for experience in benchmark.train_stream:
        strategy.train(experience)
        strategy.eval(benchmark.test_stream)

# Treinamento e avaliação com REPLAY
print("Treinando e avaliando com REPLAY...")
train_and_evaluate_replay()

preload("Treinamento e avaliação com REPLAY")

print("Processo concluído!")
