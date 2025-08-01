# Internal_LLM
📚 Avaliação de Bases de Dados Técnicas com Modelos de Linguagem de Grande Escala:  Um Estudo Aplicado ao Wi-Fi 7 📚
👨🏽‍💻 Autor: Osiel do Couto Rosa  👨🏽‍💻

Um sistema de perguntas e respostas baseado em seus documentos PDF, comparando diferentes modelos de IA.

────────────────────
🔧 PRÉ-REQUISITOS (ANTES DE COMEÇAR)
────────────────────

1. Hardware
	• Memória RAM de pelo menos 16GB

2. TERMINAL ABERTO:
   • Windows: Pressione Win+R, digite "cmd" e Enter
   • Mac/Linux: Abra "Terminal"

3. PYTHON INSTALADO (3.8 ou superior):
   • Verifique se já tem: digite no terminal:
     python --version
   • Se não tiver, baixe em: python.org/downloads

4. OLLAMA INSTALADO:
   • Baixe em: ollama.ai/download
   
────────────────────
🚀 COMO INSTALAR AS BIBLIOTECAS NECESSÁRIAS (PASSO A PASSO)
────────────────────

1. INSTALE AS BIBLIOTECAS (digite cada linha e aguarde):
   pip install langchain-community langchain-ollama
   pip install pypdf
   pip install huggingface-hub
   pip install pandas
   pip install tabulate
   pip install faiss-cpu

2. BAIXE OS MODELOS DE IA (digite cada um separadamente):
   ollama pull llama2
   ollama pull mistral
   ollama pull tinyllama

3. COLOQUE SEUS DOCUMENTOS PDF na pasta:
   /base_dados/ (crie esta pasta na mesma localização do arquivo .py) -> Ajuste no código principal o caminho da pasta
   
4. ARQUIVO DE PERGUNTAS
	Para o modo "Bateria de testes" é necessário um arquivo .csv com as seguintes caracteristicas
	Linha 1: "Perguntas"
	Linha 2-n: Perguntas que deseja enviar para os LLM
	Última linha: Flag "/end"
	Modelo:
		Perguntas
		Texto pergunta 1
		Texto pergunta 2
		...
		/end

────────────────────
💻 COMO EXECUTAR O PROGRAMA
────────────────────

1. INICIE O SERVIDOR DO OLLAMA:
   1 - Abra o terminal
   2 - Digite o comando abaixo 
		ollama serve
   
2. INICIE O SISTEMA:
   1 - Abra outro terminal em paralelo
   2 - Navegue até sua pasta de trabalho:
		cd caminho_da_sua_pasta
   3 - Execute o programa com o comando abaixo:
		python internal_llm.py
   
3. MENU PRINCIPAL aparecerá com 4 opções:
   1 - Modo normal: Faça perguntas a um modelo específico
   2 - Benchmark: Testa TODOS modelos automaticamente
   3 - Bateria de testes: Executa uma bateria de testes automatizada com perguntas pré-definidas em um arquivo csv.
   4 - Sair

─────────────────────────
📂 ESTRUTURA DE ARQUIVOS
─────────────────────────

/sua_pasta/
   │
   ├── base_dados/       (Coloque seus PDFs aqui)
   ├── internal_llm.py   (Arquivo principal)
   ├── perguntas.csv  	 (Arquivo de perguntas para ser usado na bateria de testes)
   └── faiss_index/      (Pasta criada automaticamente)

────────────────────
❓ PROBLEMAS COMUNS (SOLUÇÕES)
────────────────────

🔴 Erro "ModuleNotFoundError":
   • Significa que falta alguma biblioteca
   • Reinstale todas conforme orientação

🔴 Ollama não responde:
   • Verifique se o serviço está rodando em OUTRO terminal:
     ollama serve
   • Teste manualmente:
	ollama run llama2

🔴 PDFs não são reconhecidos:
   • Verifique se estão na pasta /base_dados/
   • Nomes de arquivos não devem ter acentos ou espaços
