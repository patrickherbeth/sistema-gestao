import unittest
from app import criar_app, db
from app.models import Tarefa

class TarefaTesteCase(unittest.TestCase):

    def setUp(self):
        self.app = criar_app()
        self.app.config['TESTING'] = True
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.client = self.app.test_client()

        with self.app.app_context():
            db.create_all()

    def tearDown(self):
        with self.app.app_context():
            db.session.remove()
            db.drop_all()

    def test_criar_tarefa(self):
        resposta = self.client.post('/tarefas', json={'titulo': 'Tarefa Teste', 'descricao': 'Descrição Teste'})
        self.assertEqual(resposta.status_code, 201)

    def test_listar_tarefas(self):
        self.client.post('/tarefas', json={'titulo': 'Tarefa Teste', 'descricao': 'Descrição Teste'})
        resposta = self.client.get('/tarefas')
        self.assertEqual(resposta.status_code, 200)
        self.assertIn('Tarefa Teste', resposta.get_data(as_text=True))

    def test_atualizar_tarefa(self):
        self.client.post('/tarefas', json={'titulo': 'Tarefa Teste', 'descricao': 'Descrição Teste'})
        resposta = self.client.put('/tarefas/1', json={'titulo': 'Tarefa Atualizada', 'descricao': 'Descrição Atualizada'})
        self.assertEqual(resposta.status_code, 200)

    def test_deletar_tarefa(self):
        self.client.post('/tarefas', json={'titulo': 'Tarefa Teste', 'descricao': 'Descrição Teste'})
        resposta = self.client.delete('/tarefas/1')
        self.assertEqual(resposta.status_code, 200)

if __name__ == '__main__':
    unittest.main()
