import requests

# URL base da sua API
BASE_URL = 'http://localhost:5000/tarefas'


def testar_get():
    print("Testando GET")
    response = requests.get(BASE_URL)
    print("Status Code:", response.status_code)
    print("Resposta:", response.json())


def testar_post():
    print("Testando POST")
    dados = {'titulo': 'Nova Tarefa', 'descricao': 'Descrição da tarefa'}
    response = requests.post(BASE_URL, json=dados)
    print("Status Code:", response.status_code)
    print("Resposta:", response.json())


def testar_put(tarefa_id):
    print("Testando PUT")
    dados = {'titulo': 'Tarefa Atualizada', 'descricao': 'Descrição atualizada'}
    url = f'{BASE_URL}/{tarefa_id}'
    response = requests.put(url, json=dados)
    print("Status Code:", response.status_code)
    print("Resposta:", response.json())


def testar_delete(tarefa_id):
    print("Testando DELETE")
    url = f'{BASE_URL}/{tarefa_id}'
    response = requests.delete(url)
    print("Status Code:", response.status_code)
    print("Resposta:", response.text)


if __name__ == '__main__':
    # Testar GET
    testar_get()

    # Testar POST (crie uma tarefa antes de atualizar ou excluir)
    testar_post()

    # Testar PUT (substitua <id> pelo ID da tarefa criada no POST)
    # Assumindo que você saiba o ID ou que ele foi retornado no POST
    tarefa_id = 1  # Exemplo de ID
    testar_put(tarefa_id)

    # Testar DELETE (substitua <id> pelo ID da tarefa que deseja excluir)
    testar_delete(tarefa_id)
