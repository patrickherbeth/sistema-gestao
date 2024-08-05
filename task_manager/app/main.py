import argparse
from app.task_manager import GerenciadorDeTarefas


def main():
    parser = argparse.ArgumentParser(description='Gerenciador de Tarefas')
    parser.add_argument('acao', choices=['criar', 'listar', 'atualizar', 'excluir', 'concluir'],
                        help='Ação a ser realizada')
    parser.add_argument('--titulo', help='Título da tarefa')
    parser.add_argument('--descricao', help='Descrição da tarefa')
    parser.add_argument('--id', type=int, help='ID da tarefa')
    parser.add_argument('--nao-concluidas', action='store_true', help='Listar apenas tarefas não concluídas')

    args = parser.parse_args()

    gerenciador = GerenciadorDeTarefas()

    if args.acao == 'criar':
        if args.titulo and args.descricao:
            tarefa = gerenciador.criarTarefa(args.titulo, args.descricao)
            print(f'Tarefa criada: {tarefa}')
        else:
            print('Título e descrição são obrigatórios para criar uma tarefa.')

    elif args.acao == 'listar':
        tarefas = gerenciador.listarTarefas(args.nao_concluidas)
        for tarefa in tarefas:
            print(tarefa)

    elif args.acao == 'atualizar':
        if args.id and (args.titulo or args.descricao):
            tarefa = gerenciador.atualizarTarefa(args.id, args.titulo, args.descricao)
            if tarefa:
                print(f'Tarefa atualizada: {tarefa}')
            else:
                print('Tarefa não encontrada.')
        else:
            print('ID da tarefa e pelo menos um campo para atualizar são obrigatórios.')

    elif args.acao == 'excluir':
        if args.id:
            gerenciador.excluirTarefa(args.id)
            print(f'Tarefa com ID {args.id} excluída.')
        else:
            print('ID da tarefa é obrigatório para excluir.')

    elif args.acao == 'concluir':
        if args.id:
            tarefa = gerenciador.marcarTarefaComoConcluida(args.id)
            if tarefa:
                print(f'Tarefa marcada como concluída: {tarefa}')
            else:
                print('Tarefa não encontrada.')
        else:
            print('ID da tarefa é obrigatório para marcar como concluída.')


if __name__ == '__main__':
    main()
