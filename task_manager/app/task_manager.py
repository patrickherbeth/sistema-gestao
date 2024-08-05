import json
import os
from datetime import datetime

class GerenciadorDeTarefas:
    def __init__(self, arquivo='tarefas.json'):
        self.arquivo = arquivo
        self.tarefas = self._carregar_tarefas()

    def _carregar_tarefas(self):
        if os.path.exists(self.arquivo):
            with open(self.arquivo, 'r') as f:
                return json.load(f)
        return []

    def _salvar_tarefas(self):
        with open(self.arquivo, 'w') as f:
            json.dump(self.tarefas, f, indent=4)

    def criarTarefa(self, titulo, descricao):
        tarefa = {
            'id': len(self.tarefas) + 1,
            'titulo': titulo,
            'descricao': descricao,
            'concluida': False,
            'dataCriacao': datetime.now().isoformat(),
            'dataConclusao': None
        }
        self.tarefas.append(tarefa)
        self._salvar_tarefas()
        return tarefa

    def listarTarefas(self, apenasNaoConcluidas=False):
        if apenasNaoConcluidas:
            return [tarefa for tarefa in self.tarefas if not tarefa['concluida']]
        return self.tarefas

    def atualizarTarefa(self, tarefa_id, titulo=None, descricao=None):
        for tarefa in self.tarefas:
            if tarefa['id'] == tarefa_id:
                if titulo:
                    tarefa['titulo'] = titulo
                if descricao:
                    tarefa['descricao'] = descricao
                self._salvar_tarefas()
                return tarefa
        return None

    def excluirTarefa(self, tarefa_id):
        self.tarefas = [tarefa for tarefa in self.tarefas if tarefa['id'] != tarefa_id]
        self._salvar_tarefas()

    def marcarTarefaComoConcluida(self, tarefa_id):
        for tarefa in self.tarefas:
            if tarefa['id'] == tarefa_id:
                tarefa['concluida'] = True
                tarefa['dataConclusao'] = datetime.now().isoformat()
                self._salvar_tarefas()
                return tarefa
        return None
