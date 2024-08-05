from app import app, db
from app.models import Tarefa
from flask import request, jsonify

@app.route('/tarefas', methods=['GET'])
def listar_tarefas():
    tarefas = Tarefa.query.all()
    return jsonify([{
        'id': tarefa.id,
        'titulo': tarefa.titulo,
        'descricao': tarefa.descricao,
        'concluida': tarefa.concluida,
        'data_criacao': tarefa.data_criacao,
        'data_conclusao': tarefa.data_conclusao
    } for tarefa in tarefas])

@app.route('/tarefas', methods=['POST'])
def criar_tarefa():
    data = request.get_json()
    nova_tarefa = Tarefa(titulo=data['titulo'], descricao=data['descricao'])
    db.session.add(nova_tarefa)
    db.session.commit()
    return jsonify({'id': nova_tarefa.id}), 201

@app.route('/tarefas/<int:tarefa_id>', methods=['PUT'])
def atualizar_tarefa(tarefa_id):
    data = request.get_json()
    tarefa = Tarefa.query.get_or_404(tarefa_id)
    if 'titulo' in data:
        tarefa.titulo = data['titulo']
    if 'descricao' in data:
        tarefa.descricao = data['descricao']
    db.session.commit()
    return jsonify({'id': tarefa.id, 'titulo': tarefa.titulo, 'descricao': tarefa.descricao, 'concluida': tarefa.concluida, 'data_criacao': tarefa.data_criacao, 'data_conclusao': tarefa.data_conclusao})

@app.route('/tarefas/<int:tarefa_id>', methods=['DELETE'])
def excluir_tarefa(tarefa_id):
    tarefa = Tarefa.query.get_or_404(tarefa_id)
    db.session.delete(tarefa)
    db.session.commit()
    return '', 204
