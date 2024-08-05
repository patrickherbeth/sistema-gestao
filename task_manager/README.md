# Sistema de Gestão de Tarefas

## Objetivo
Desenvolver um sistema simples de gestão de tarefas com suporte para criação, visualização, atualização e exclusão de tarefas. O projeto é containerizado usando Docker.

## Configuração Inicial

1. **Clone o Repositório:**

   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd task_manager

1. **Criar banco de dadps:**

   ```bash
     sudo docker-compose run web python app/init_db.py