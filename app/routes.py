from flask import Blueprint, request, jsonify
from .models import db, Tarefa

bp = Blueprint('tarefas', __name__)

@bp.route('/tarefas', methods=['POST'])
def criar_tarefa():
    dados = request.get_json()
    nova_tarefa = Tarefa(titulo=dados['titulo'], descricao=dados['descricao'])
    db.session.add(nova_tarefa)
    db.session.commit()
    return jsonify(nova_tarefa.to_dict()), 201

@bp.route('/tarefas', methods=['GET'])
def listar_tarefas():
    tarefas = Tarefa.query.all()
    return jsonify([tarefa.to_dict() for tarefa in tarefas])

@bp.route('/tarefas/<int:id>', methods=['PUT'])
def atualizar_tarefa(id):
    dados = request.get_json()
    tarefa = Tarefa.query.get_or_404(id)
    tarefa.titulo = dados['titulo']
    tarefa.descricao = dados['descricao']
    db.session.commit()
    return jsonify(tarefa.to_dict())

@bp.route('/tarefas/<int:id>', methods=['DELETE'])
def excluir_tarefa(id):
    tarefa = Tarefa.query.get_or_404(id)
    db.session.delete(tarefa)
    db.session.commit()
    return '', 204

@bp.route('/tarefas/<int:id>/concluir', methods=['PUT'])
def concluir_tarefa(id):
    tarefa = Tarefa.query.get_or_404(id)
    tarefa.concluida = True
    db.session.commit()
    return jsonify(tarefa.to_dict())

def init_app(app):
    app.register_blueprint(bp)
