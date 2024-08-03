from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from transformers import DistilBertTokenizer
from model import FakeNewsClassifier  # Certifique-se de ajustar o caminho do modelo

# Inicializar o aplicativo FastAPI
app = FastAPI()

# Carregar o modelo treinado e o tokenizador
device = torch.device("cpu")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = FakeNewsClassifier(n_classes=2)  # Ajuste o número de classes conforme necessário
model.load_state_dict(torch.load('path_to_saved_model.pt', map_location=device))
model = model.to(device)
model.eval()

# Modelo de dados para a previsão
class PredictionRequest(BaseModel):
    statement: str

# Modelo de dados para tarefas
class Task(BaseModel):
    id: int
    description: str
    completed: bool

# Banco de dados fictício para tarefas
tasks_db = [
    {"id": 1, "description": "Tarefa 1", "completed": False},
    {"id": 2, "description": "Tarefa 2", "completed": True},
]

# Rota de verificação do servidor
@app.get("/")
def read_root():
    return {"message": "Servidor está rodando"}

# Rota para fazer previsões
@app.post("/predict")
def predict(request: PredictionRequest):
    # Pré-processamento do texto de entrada
    inputs = tokenizer.encode_plus(
        request.statement,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Fazer previsão
    with torch.no_agrad():
        outputs = model(input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs.logits, dim=1)

    return {"prediction": prediction.item()}

# Rota para obter todas as tarefas
@app.get("/tarefas", response_model=List[Task])
def get_tasks():
    return tasks_db

# Rota para obter uma tarefa específica pelo ID
@app.get("/tarefas/{task_id}", response_model=Task)
def get_task(task_id: int):
    task = next((task for task in tasks_db if task["id"] == task_id), None)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

# Rota para criar uma nova tarefa
@app.post("/tarefas", response_model=Task)
def create_task(task: Task):
    tasks_db.append(task.dict())
    return task

# Rota para atualizar uma tarefa existente
@app.put("/tarefas/{task_id}", response_model=Task)
def update_task(task_id: int, task: Task):
    task_index = next((index for index, t in enumerate(tasks_db) if t["id"] == task_id), None)
    if task_index is None:
        raise HTTPException(status_code=404, detail="Task not found")
    tasks_db[task_index] = task.dict()
    return task

# Rota para deletar uma tarefa
@app.delete("/tarefas/{task_id}", response_model=Task)
def delete_task(task_id: int):
    task_index = next((index for index, t in enumerate(tasks_db) if t["id"] == task_id), None)
    if task_index is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks_db.pop(task_index)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
