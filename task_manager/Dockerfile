# Usar uma imagem base do Python
FROM python:3.11-slim

# Definir o diretório de trabalho
WORKDIR /app

# Copiar os arquivos para o diretório de trabalho
COPY . .

# Instalar as dependências
RUN pip install -r requirements.txt

# Comando padrão para rodar a aplicação
CMD ["flask", "run", "--host=0.0.0.0"]
