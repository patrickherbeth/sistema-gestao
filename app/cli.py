import click
from flask.cli import with_appcontext
from . import db
from .models import Tarefa

@click.command(name='criar_tarefa')
@click.argument('titulo')
@click.argument('descricao')
@with_appcontext
def criar_tarefa(titulo, descricao):
    tarefa = Tarefa(titulo=titulo, descricao=descricao)
    db.session.add(tarefa)
    db.session.commit()
    click.echo(f'Tarefa {titulo} criada.')

@click.command(name='listar_tarefas')
@with_appcontext
def listar_tarefas():
    tarefas = Tarefa.query.all()
    for tarefa in tarefas:
        click.echo(f'{tarefa.id}: {tarefa.titulo} - {tarefa.descricao} - {tarefa.data_criacao} - {tarefa.data_conclusao}')
