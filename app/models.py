from . import db
from datetime import datetime

class Tarefa(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    titulo = db.Column(db.String(100), nullable=False)
    descricao = db.Column(db.String(200), nullable=False)
    data_criacao = db.Column(db.DateTime, default=datetime.utcnow)
    data_conclusao = db.Column(db.DateTime, nullable=True)
    concluida = db.Column(db.Boolean, default=False)

    def to_dict(self):
        return {
            'id': self.id,
            'titulo': self.titulo,
            'descricao': self.descricao,
            'data_criacao': self.data_criacao.isoformat(),
            'data_conclusao': self.data_conclusao.isoformat() if self.data_conclusao else None,
            'concluida': self.concluida
        }
