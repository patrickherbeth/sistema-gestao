# Sistema de Gestão de Tarefas

## Configuração Inicial

1. Clone o repositório:
   ```sh
   git clone <link-do-repositorio>
   cd gestor_tarefas


##  Crie e ative o ambiente virtual:

2. Crie e ative o ambiente virtual:
   ```sh
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   pip install -r requirements.txt

3. Crie e inicie os containers:
   ```sh
   docker-compose up --build

4. Execute os testes:
   ```sh
   python -m unittest discover testes